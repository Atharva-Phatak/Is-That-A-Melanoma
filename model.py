import torch
import torch.nn as nn
import torch.nn.functional as F
 
from attention import GlobalAvgAttnPool2d
from efficientnet_pytorch import EfficientNet


class MelanomaModel(nn.Module):
    '''This is simple class implementation which uses efficientnet as the backbone.
       Args:

            backbone : the name of the efficientnet model which is going to be used as a backbone
        
        for example : backone = 'efficientnet-b0'
    '''

    def __init__(self , backbone):

        super(MelanomaModel , self).__init__()

        self.backbone = backbone
        self.model = EfficientNet.from_pretrained(self.backbone)
        self.dropout = nn.Dropout(0.4)

        self.head = nn.Linear(self.model._conv_head.out_channels*2 , 1)
        self.attn = GlobalAvgAttnPool2d(in_features= self.model._conv_head.out_channels)

    def forward(self , x):

        bs , _ , _ , _ = x.shape

        op = self.model.extract_features(x)
        op = self.dropout(op)
        op = self.attn(op).reshape(bs , -1)

        op = self.head(op)
        return op
    
       
