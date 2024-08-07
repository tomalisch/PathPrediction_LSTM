## Introduction

## Dependencies
import os
import math
from typing import Any
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
import matplotlib as mpl
import seaborn as sns
import random
from PIL import Image
import pandas as pd
import scipy.io
import datetime
import hdf5storage
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import lightning as L
from lightning.pytorch.callbacks import RichProgressBar

# Set past frame number from which to predict next coordinate
timesteps = 10

# Get and store current path and load training data (specific syntax depends on input raw)
dirLSTM = os.getcwd()+'/'
data = hdf5storage.loadmat(dirLSTM+'data.mat')['Centroidarray']

# Shape training data array and drop NaNs
trainingData = pd.DataFrame(data.transpose([2,0,1]).reshape(-1, 2))
trainingData = trainingData.dropna()

# Scale coordinates independently to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
trainingData = pd.DataFrame( scaler.fit_transform(trainingData) )

# Cull last values in array to allow reshaping according to timesteps set
overflow = len(trainingData) % (timesteps+1)
trainingData = trainingData.drop( trainingData.iloc[-overflow:,:].index, axis='index' )

# Set up target training data (X, Y coordinates every ('timestep'+1)nth frame)
trainingDataXY = np.array(trainingData).reshape((-1,timesteps+1,2))
trainingDataXY_Next = trainingDataXY[:,timesteps,:].reshape(-1,1,2)
# Remove target values from input training set 
trainingDataXY = np.delete(trainingDataXY, timesteps, axis=1)

## Set up input and output tensors, and wrap in DataLoader function
inputs = torch.tensor(trainingDataXY)
targets = torch.tensor(trainingDataXY_Next)
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset)


## Manually set up recurrent neural net using Pytorchs
class PP_LSTM_manual(L.LightningModule):
    
    # Create and initialize weights and biases tensors
    def __init__(self):
        # superInit from Lightning to inherit log functions
        super().__init__()
        # Save hyperparameters in checkpoints
        self.save_hyperparameters()

        # Set up tensors storing weights and bias means and stds
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # Create weights and bias parameters based on random initialization values (Gaussian)
        # Forget gate parameters
        self.wFG1 = nn.Parameter(  torch.normal(mean,std), requires_grad=True )
        self.wFG2 = nn.Parameter(  torch.normal(mean,std), requires_grad=True )
        self.bFG1 = nn.Parameter(  torch.tensor(0.0), requires_grad=True )
        # Input gate params
        self.wIG1 = nn.Parameter(  torch.normal(mean,std), requires_grad=True )
        self.wIG2 = nn.Parameter(  torch.normal(mean,std), requires_grad=True )
        self.bIG1 = nn.Parameter(  torch.tensor(0.0), requires_grad=True )
        # New information params
        self.wNI1 = nn.Parameter(  torch.normal(mean,std), requires_grad=True )
        self.wNI2 = nn.Parameter(  torch.normal(mean,std), requires_grad=True )
        self.bNI1 = nn.Parameter(  torch.tensor(0.0), requires_grad=True )
        # Output gate params
        self.wOG1 = nn.Parameter(  torch.normal(mean,std), requires_grad=True )
        self.wOG2 = nn.Parameter(  torch.normal(mean,std), requires_grad=True )
        self.bOG1 = nn.Parameter(  torch.tensor(0.0), requires_grad=True )


    # Create an lstm unit
    def lstm_unit(self, inputValue, longMemory, shortMemory):

        # Forget gate remember percent w/ sigmoid activation function
        longMemory_remember = torch.sigmoid( (shortMemory * self.wFG1) + (inputValue * self.wFG2) + self.bFG1 )

        # Input gate (potential longerm memory remember percent & potential long term memory information) w/ sigmoid and tanh activation functions
        potentialMemory_remember = torch.sigmoid( (shortMemory * self.wIG1) + (inputValue * self.wIG2) + self.bIG1 )
        potentialMemory = torch.tanh( (shortMemory * self.wNI1) + (inputValue * self.wNI2) + self.bNI1 )

        # Update old long term term memory with new information
        longMemory_updated = ( (longMemory * longMemory_remember) + (potentialMemory_remember * potentialMemory) )
        
        # Ouput gate (potential shortterm memory remember percent & potential short term memory information) w/ sigmoid and tanh activation functions
        shortMemory_remember = torch.sigmoid( (shortMemory * self.wOG1) + (inputValue * self.wOG2) + self.bOG1 )
        shortMemory_updated = torch.tanh(longMemory_updated) * shortMemory_remember

        # Return unit output (updated long term & short term memory)
        return [longMemory_updated, shortMemory_updated]


    # Perform one step through entire (unrolled) LSTM
    def forward(self, input):

        # Initialize long and short term memory
        longMemory = 0
        shortMemory = 0

        # For each frame, sequentially go through the LSTM unit 
        for frame in range(len(input)):
            longMemory, shortMemory = self.lstm_unit( input[frame], longMemory, shortMemory )

        # Return updated short term memory output (output gate) of last time step 
        return shortMemory
    

    # Function to configure Adam optimizer (e.g., if default LR is not optimal))
    def configure_optimizers(self):

        # Keep Adam optimizer parameters default; change if learning insufficient
        return Adam(self.parameters())


    # Perform one step of model training using batch data and its indices; calculate and log loss
    def training_step(self, batch, batchIDs):

        # Define input and targets based on batch data input
        input_i, target_i = batch
        # Forward through unrolled model, predicting outputs from inputs, weights, and biases
        output_i = self.forward(input_i[0])
        # Compute loss as mean squared difference between output and target
        loss = torch.mean((output_i - target_i)**2)
        # Use Lightning to log training loss
        self.log('train_loss', loss)

        return loss

## Alternative 1: Set up and train PyTorch model with inbuilt Lightning functions
class PP_LSTM_Lightning(L.LightningModule):

    def __init__(self):
        super().__init__()

        # Input and output parameters are X and Y coordinates
        # Single layer for now, expand to check if it increases performance
        self.lstm=nn.LSTM(input_size=2, hidden_size=2, batch_first=True, dropout=0.0)

    def forward(self, input):
        lstm_out, temp = self.lstm(input)
  
        # Prediction is the last unrolled lstm unit output of the model (short term memory value)
        prediction = lstm_out[-1]
        return prediction
    
    def configure_optimizers(self, lr=0.001):

        return Adam(self.parameters(), lr=lr)
    
    # Perform one step of model training using batch data and its indices; calculate and log loss
    def training_step(self, batch, batchIDs):

        # Define input and targets based on batch data input
        input_i, target_i = batch
        # Forward through unrolled model, predicting outputs from inputs, weights, and biases
        output_i = self.forward(input_i[0])
        # Compute loss as mean squared difference between output and target
        loss = torch.mean((output_i - target_i)**2)
        # Use Lightning to log training loss
        self.log('train_loss', loss)

        return loss

## Alternative 2: Set up a stacked keras LSTM layer by layer
modelKeras = Sequential()
modelKeras.add(LSTM(128, input_shape=(trainingDataXY.shape[1], trainingDataXY.shape[2]), return_sequences='True'))
modelKeras.add(LSTM(64, return_sequences='True'))
modelKeras.add(LSTM(32, return_sequences='True'))
modelKeras.add(LSTM(16))
modelKeras.add(Dropout(0.2))
# Add dropout layer after LSTM layers (but cf. Cheng et al., 2017 who propose per frame masks)
# Check dense layer output & shape
modelKeras.add(Dense(trainingDataXY_Next.shape[2], activation='relu'))
modelKeras.compile(optimizer='adam', loss='mse')
modelKeras.summary()
# Fit keras model
history = modelKeras.fit( trainingDataXY, trainingDataXY_Next, epochs=10, batch_size=64, validation_split=0.3, verbose=1 )
# Plot training vs validation loss to check for overfitting
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training loss', 'validation loss'])
## Test Keras model
testCoords = trainingDataXY[random.randint(0,len(trainingDataXY[:,0,0])),:,:]
predCoords = modelKeras.predict(testCoords[:-1,:].reshape((1,len(testCoords[:,0])-1,2)))
predCoords = scaler.inverse_transform( predCoords )
testCoords = scaler.inverse_transform( testCoords )
trueCoords = testCoords[-1,:] 
plt.scatter(testCoords[:,0], testCoords[:,1])
plt.scatter(trueCoords[0], trueCoords[1])
plt.scatter(predCoords[0, 0], predCoords[0, 1])
plt.legend(['base', 'true', 'predicted'])



## Set up and train the Lightning model
modelLightning = PP_LSTM_Lightning()

# Set up Lightning trainer
trainerLightning = L.Trainer(max_epochs=10, log_every_n_steps=1, accelerator='gpu', devices='auto', strategy='auto')
trainerLightning.fit(model=modelLightning, train_dataloaders=dataloader)

# Check model output with random input once
testCoords = torch.tensor( scaler.transform( [[16.271627, 28.229027],
       [15.893902, 28.364021],
       [15.524648, 28.632938],
       [15.184661, 28.88007 ],
       [14.943663, 28.832201],
       [14.151804, 29.382717],
       [14.068769, 29.366947],
       [13.594865, 29.508871],
       [13.239436, 29.688492],
       [12.843507, 29.885359]] ).astype(np.float32) ).to(torch.device("mps"))

predCoords = modelLightning(testCoords[:-1,:])
predCoords = scaler.inverse_transform( predCoords.cpu().detach().reshape([1,-1]) )
testCoordsCPU = scaler.inverse_transform( testCoords.cpu().detach() )
plt.scatter(testCoordsCPU[:-1,0], testCoordsCPU[:-1,1])
plt.scatter(predCoords[0, 0], predCoords[0, 1])
plt.scatter(testCoordsCPU[-1, 0], testCoordsCPU[-1, 1])

# Generate simulated tracks based on starting input frames
# Moving window of f frames, predict next frame, move predicted frame into f-frame long window, repeat

# How many minutes of tracks to generate (note training data was 30fps)
lengthMin = 60

# Starting coordinates of f frames
startCoords = torch.tensor( scaler.transform( [[16.271627, 28.229027],
       [15.893902, 28.364021],
       [15.524648, 28.632938],
       [15.184661, 28.88007 ],
       [14.943663, 28.832201],
       [14.151804, 29.382717],
       [14.068769, 29.366947],
       [13.594865, 29.508871],
       [13.239436, 29.688492],
       [12.843507, 29.885359]] ).astype(np.float32) ).to(torch.device("mps"))

simulatedTracks = np.zeros([30 * 60 * lengthMin, 2])
movingWindow = startCoords

# Main loop
for frame in range(len(simulatedTracks[:,0])):
    # Predict next coordinates from current moving window
    predCoords = modelLightning(movingWindow)
    # Log inverse transformed predicted coordinate for later plotting
    simulatedTracks[frame, :] = scaler.inverse_transform( predCoords.cpu().detach().reshape([1,-1]) )
    # Adjust moving window for next frame, roll elements backwards first, then replace last (rolled first) element with new predicted value
    movingWindow = movingWindow.roll(-1,0)
    movingWindow[-1,:] = predCoords

# Plot generated tracks
plt.scatter( simulatedTracks[:,0], simulatedTracks[:,1] )
