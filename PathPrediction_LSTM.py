## Introduction

## Dependencies
import os
import math
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
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

import lightning as L

# Loading training data (specific syntax depends on input raw)
data = hdf5storage.loadmat(os.getcwd()+'/'+'data.mat')['Turncell']

# Shape training data array
trainingData = pd.DataFrame(data.reshape((108000*1000, -1)))
trainingData = trainingData.iloc[:,0:2]
trainingData.dropna()
trainingDataXY = trainingData[:,0:2]
trainingDataXY = trainingDataXY.reshape((-1,10,2))
trainingDataXY_Next = trainingData[0:len(trainingData[:,0]):10,0:2]

## Set up recurrent neural net using Pytorch

class PP_LSTM_manual(L.LightningModule):
    
    # Create and initialize weights and biases tensors
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # superInit from Lightning to inherit log functions
        super().__init__(*args, **kwargs)

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
    def fwd(self, input):

        # Initialize long and short term memory
        longMemory = 0
        shortMemory = 0

        # For each frame, sequentially go through the LSTM unit 
        for frame in range(len(input)):
            longMemory, shortMemory = self.lstm_unit( input[frame], longMemory, shortMemory )

        # Return updated short term memory output (output gate) of last time step 
        return shortMemory
    

    # Function to configure Adam optimizer (e.g., if default LR is not optimal))
    def configureOptimizers(self):

        # Keep Adam optimizer parameters default; change if learning insufficient
        return Adam(self.parameters)


    # Perform one step of model training using batch data and its indices; calculate and log loss
    def trainingStep(self, batch, batchIDs):

        # Define input and targets based on batch data input
        input_i, target_i = batch
        # Forward through unrolled model, predicting outputs from inputs, weights, and biases
        output_i = self.fwd(input_i[0])
        # Compute loss as mean squared difference between output and target
        loss = (output_i - target_i)**2
        # Use Lightning to log training loss
        self.log('train_loss', loss)

        return loss
