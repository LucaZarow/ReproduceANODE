import torch
import torch.nn

class AugmentedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, filter_size):
        super(AugmentedConv2d, self).__init__(self, in_channels + 1, out_channels, filter_size):
    
    def forward(self, img, time):
        