## Introduction

## Dependencies
import os
import math
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

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout

# Loading training data

# Shape training data array

trainingData = pd.DataFrame(dataABM_Lior.reshape((108000*1000, -1)))
trainingData = trainingData.iloc[:,0:2]
trainingData.dropna()
scaler = MinMaxScaler()
trainingData = scaler.fit_transform(trainingData)
trainingDataXY = trainingData[:,0:2]
trainingDataXY = trainingDataXY.reshape((-1,30,2))
trainingDataXY_Next = trainingData[0:len(trainingData[:,0]):30,0:2]

## Set up recurrent neural net

rnn = Sequential()

# Input layer
rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (trainingDataXY.shape[1], 2)))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units = 45, return_sequences = True))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units = 45, return_sequences = True))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units = 45))
rnn.add(Dropout(0.2))

# Output layer
rnn.add(Dense(units = 2))


opt = keras.optimizers.Adam(learning_rate=0.0001)
rnn.compile(optimizer = opt, loss = 'mean_squared_error')
rnn.fit(trainingDataXY, trainingDataXY_Next, epochs = 10, batch_size = 32)