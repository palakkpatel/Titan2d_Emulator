import numpy as np
from numpy.core.numeric import indices
import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from torchinfo import summary
from time import time


# Simulation Dataset from Parameter file and Simulation Max Height Data

class SimData(Data.Dataset):
    def __init__(self, parameters_file, sim_dir):
        self.parameters = pd.read_excel(parameters_file,skiprows=1)
        self.parameters = self.parameters.iloc[:,0:7]
        sc = StandardScaler()
        self.parameters.iloc[:,1:7] = sc.fit_transform(self.parameters.iloc[:,1:7])
        self.sim_dir = sim_dir
        
    def __len__(self):
        return len(self.parameters)
    
    def __getitem__(self,idx):
        sim_file = str(self.sim_dir) + '/pileheightrecord_' + str(self.parameters.iloc[idx,0]) + '.txt'
        temp_sim = np.loadtxt(sim_file)
        temp_sim = temp_sim.reshape(1, temp_sim.shape[0], temp_sim.shape[1])
        sim_data = torch.from_numpy(temp_sim)
        sim_parameters = torch.from_numpy(self.parameters.iloc[idx,1:].values)
        return sim_parameters, sim_data
    

# Max Height Model to generate a 2d 300x300 grid from 6 parameters
class MaxHeightModel(nn.Module):
    def __init__(self, input_size):
        super(MaxHeightModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 72)
        self.linear2 = nn.Linear(72, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, 6, 6)
        self.deconv2 = nn.ConvTranspose2d(512, 128, 5, 5)
        self.deconv3 = nn.ConvTranspose2d(128, 32, 5, 5)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 2, 2)
        
    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = x.reshape(x.shape[0],1024,1,1)
        x = nn.functional.relu(self.deconv1(x))
        x = nn.functional.relu(self.deconv2(x))
        x = nn.functional.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x
    

# Specifying Model Training 
if torch.cuda.is_available():
    dev = "cuda"
    print("Using CUDA")
else:
    dev = "cpu"
    print("Using CPU")

device = torch.device(dev)





