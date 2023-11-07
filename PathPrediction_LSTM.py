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
scaler = MinMaxScaler()
trainingData = scaler.fit_transform(trainingData)
trainingDataXY = trainingData[:,0:2]
trainingDataXY = trainingDataXY.reshape((-1,10,2))
trainingDataXY_Next = trainingData[0:len(trainingData[:,0]):10,0:2]

## Set up recurrent neural net using Pytorch

