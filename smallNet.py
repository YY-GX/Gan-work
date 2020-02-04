import torch
import torch.nn as nn
import torch.nn.functional as F

class smallNet(nn.Module):

    def __init__(self):
        super(smallNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv = nn.Conv2d(3, 64, 3)
        # batch         
        self.batch = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # relu
        self.relu = nn.ReLU()
        # max pooling        
        self.maxpool = nn.MaxPool2d(2)
        # fc
        self.fc = nn.Linear(788544, 1)

    def forward(self, x):
#         print(x.size())
        x = self.conv(x)
        
        x = self.batch(x)
        
        x = self.relu(x)
        
        x = self.maxpool(x)
        
        x = x.view(-1, self.num_flat_features(x))
        return self.fc(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

