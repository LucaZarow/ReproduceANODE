import torch
from torch import nn

class VanillaMNIST(nn.Module):
    def __init__(self):
        super(VanillaMNIST, self).__init__()

        self.blockOne = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2)
        )
        
        self.blockTwo = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2)
        )
        
        self.blockThree = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(0.1),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.linear = nn.Sequential(
            nn.Linear(32, 10),
        )
    
    def clear(self, m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            
    def forward(self, x):
        x = self.blockOne(x)
        x = self.blockTwo(x)
        x = self.blockThree(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def reset(self, model):
        model.apply(model.clear)