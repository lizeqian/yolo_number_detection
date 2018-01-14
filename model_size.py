# Define a model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)  #j=1, r=7
        self.conv2 = nn.Conv2d(32, 64, 7, padding=3) #j=j*s=2, r=r+(k-1)*j=11
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) #j=2, r=15
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #j=4, r=23
        self.conv5 = nn.Conv2d(256, 31, 3, padding=1) #j=16, r=71+16*6=167
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(31)

    def forward(self, x):
        x = self.pool((self.bn1(self.conv1(x))))
        x = self.pool((self.bn2(self.conv2(x))))
        x = self.pool((self.bn3(self.conv3(x))))
        x = self.pool((self.bn4(self.conv4(x))))
        x = self.conv5(x)
        return x

model = Model()

# Estimate Size
from pytorch_modelsize import SizeEstimator

se = SizeEstimator(model, input_size=(16,1,256,256))
print(se.estimate_size())

# Returns
# (size in megabytes, size in bits)
# (408.2833251953125, 3424928768)

print(se.param_bits) # bits taken up by parameters
print(se.forward_backward_bits) # bits stored for forward and backward
print(se.input_bits) # bits for input
