from random import random

import torch
import sys




class CNN(torch.nn.Module):
    def __init__(self, ):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, stride=1)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, stride=2)
        self.fc1 = torch.nn.Linear(7200, 128)
        self.fc2 = torch.nn.Linear(128, 7)


    def forward(self, patches):
        """
        
        """
        x = self.conv1(patches)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        out = self.fc2(x)

        #if not self.training:
        #    out = torch.softmax(out)

        return out


