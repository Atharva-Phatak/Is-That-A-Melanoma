#importing the libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F

class GlobalAttnPool2d(nn.Module):

    ''' Implementation of simple attention mechanism
        Args:
            in_features : The number of input features to be expected.
    '''

    def __init__(self , in_features):

        super(GlobalAttnPool2d , self).__init__()

        self.in_featues = in_features

        self.attn = nn.Sequential(
                    nn.Conv2d(in_channels= in_features,
                    out_channels = 1 , kernel_size= 1,
                    padding=0 , stride = 1,
                    bias = False),

                    nn.ReLU()
        )
    
    def forward(self , x):

        x_a = self.attn(x)
        x = x*x_a
        x = torch.sum(x , dim =[-2 , -1] , keepdim= True)

        return x


class GlobalAvgAttnPool2d(nn.Module):
    ''' Implementation of AvgAttn which is simply the concatentation of GlobalAttn and Avg pooling.
        Args:
            in_features : The number of input features to be expected.
    '''

    def __init__(self , in_features):

        super(GlobalAvgAttnPool2d , self).__init__()

        self.in_features = in_features
        self.attn = GlobalAttnPool2d(in_features = self.in_features)

    def forward(self , x):

        x_a = self.attn(x)
        x_avg = F.adaptive_avg_pool2d(x , 1)

        op = torch.cat([x_a , x_avg] , dim = 1)
        return op