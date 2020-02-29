import torch
import torch.nn as nn
import torch.nn.functional as F

class smallNet(nn.Module):

    def __init__(self, dropout):
        super(smallNet, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),    
            nn.MaxPool2d(2)
        )  
        
        self.seq2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )          
              
        # fc
        self.fc = nn.Linear(16384, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data,0.1)


        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
#         print('input:', x.size())
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq2(x)
        x = self.seq2(x)
#         x = self.seq2(x)
#         x = self.seq2(x)
#         x = self.seq2(x)
#         x = self.seq2(x)

#         print('output:', x.size())
#         x = x.view(-1, self.num_flat_features(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features


