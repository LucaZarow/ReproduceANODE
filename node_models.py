import torch
from torch import nn
from torchdiffeq import odeint

class NeuralODE(nn.Module):
    # Note certain parameters are constant throughout paper experiments and so are used directly, namely:
    # time_dependent = True
    # non_linearity = 'relu'
    # adjoint = False
    def __init__(self, device, image_size, num_filters, out_dim, 
               augmented_dim=0, tolerance=1e-3):
        super(NeuralODE, self).__init__()

        flattened_dim = (img_size[0] + augmented_dim) * img_size[1] * img_size[2]

        function = ODEConv(image_size, num_filters, augmented_dim)

        self.block_ODE = ODEBlock(device, function, tolerance)
        self.block_non_linear = nn.Linear(flattened_dim, out_dim)

    def forward(self, x):
        x = self.block_ODE(x)
        x = self.block_non_linear(x)

        return x


class ODEBlock(nn.Module):
    # is_conv = true
    # adjoint = False
    def __init__(self, device, function, tolerance):
        super(ODEBlock, self).__init__()
        self.device = device
        self.function = function
        self.tolerance = tolerance

    # eval_times=None (since not plotting convolution trajectory)
    def forward(self, t, x):
        self.function.nfe = 0

        #Only need final result of convolution for plots
        integration_time = torch.tensor([0, 1]).float().type_as(x)

        #if ANODE
        if self.function.augmented_dim > 0:
            batch_size, channels, height, width = x.shape
            aug = torch.zeros(batch_size, self.function.augment_dim,
                              height, width).to(self.device)
            x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        out = odeint(self.function, x_aug, integration_time,
                    rtol=self.tolerance, atol=self.tolerance, method='dopri5',
                    options={'max_num_steps': MAX_NUM_STEPS})
        return out

class ODEConv(nn.Module):  
    # time_dependent = True
    # non_linearity = 'relu'
    def __init__(self, image_size, num_filters, augmented_dim): 
        super(ODEConv, self).__init__()
        self.nfe = 0  # Number of function evaluations

        channels, height, width = img_size
        channels += augmented_dim
       
        self.conv1 = Conv2dTime(channels, num_filters,
                                kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv2dTime(num_filters, num_filters,
                                kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv2dTime(num_filters, channels,
                                kernel_size=1, stride=1, padding=0)

        self.block_non_linear = nn.ReLU(inplace=True)

    def forward(self, t, x):
        out = self.conv1(t, x)
        x = self.block_non_linear(x)
        x = self.block_conv2(t, x)
        x = self.block_non_linear(x)
        x = self.block_conv3(t, x)

# same as code base
class Conv2dTime(nn.Conv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)
