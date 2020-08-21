import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.cuda import amp

from sklearn import metrics 

from engine import TrainModel , EvalModel
from dataset import MelanomaDataset
from model import MelanomaModel
from utils import get_train_val_split

import numpy as np
import pandas as pd

import os
import argparse

parser_args = argparse.ArgumentParser()

parser_args.add_argument('-b' , '--backbone' , default= 'efficientnet-b0' , 
                    help = "Efficientnet model name")

#parser.add_argument('-i' , '--inference' , default = False ,
 #                   help = 'Enter True for Inference or False for Training')
parser_args.add_argument('-trainP' , '--train_path' , help = 'Path to Training Images' , default = 'train/')
#parser.add_argument('-train' , '--test_path' , help = 'Path to Testing Images' , default= ' test/')

parser_args.add_argument('-e' , '--epochs' , default = 10 , help = 'The Number of Training Epochs')
parser_args.add_argument('-bs' , '--batchsize' , default= 8 , help = 'The Batch Size for dataloader')
parser_args.add_argument('-augs' , '--augmentations' , default= False , help = 'Enter True for performing Augmentations or False if not needed')
parser_args.add_argument('-lr' , '--learning_rate' , default= 0.01 ,help = 'The value of learning rate' )

parser = parser_args.parse_args()

def GetDataLoader():

    df = pd.read_csv('train.csv')
    train_df , valid_df = get_train_val_split(df)

    if parser['augmentations']:
        train_transforms =  transforms.Compose([
                                    transforms.ColorJitter(brightness = 0.7 , contrast = 0.3),
                                    transforms.RandomRotation(degrees = 75),
                                    transforms.RandomHorizontalFlip(p = 0.6),
                                    transforms.RandomVerticalFlip(p = 0.7),
                                    
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = (0.485, 0.456, 0.406) ,
                                    std = (0.229, 0.224, 0.225))])
        
        

    else:

        train_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean = (0.485, 0.456, 0.406) ,
                                    std = (0.229, 0.224, 0.225))])
    

    train_dataset = MelanomaDataset(df = train_df,
                                       path = parser['train_path'],
                                       transformations= train_transforms,
                                       is_train = True)
    
    
    trainloader = torch.utils.data.DataLoader(train_dataset , batch_size = parser['bs'],
                                            shuffle = True)
    
    valid_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean = (0.485, 0.456, 0.406) ,
                                    std = (0.229, 0.224, 0.225))])

    valid_dataset = MelanomaDataset(df = valid_df,
                                       path = parser['train_path'],
                                       transformations= valid_transforms,
                                       is_train = True)
    
    validloader = torch.utils.data.DataLoader(valid_dataset , batch_size= parser['batchsize'] , shuffle = False)

    return trainloader , validloader


def run_model():


    trainloader , validloader = GetDataLoader()
    net = MelanomaModel(backbone = parser['backbone'])
    net_optim = torch.optim.SGD(net.parameters() , lr = parser['learining_rate'] , momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(net_optim , T_0 = 10 , T_mult = 2 , eta_min = 3e-7)

    scaler = amp.GradScaler()

    if torch.cuda.is_available():

        net = net.cuda()

    best_auc = 0

    for epoch in range(parser['epochs']):

        train_loss = TrainModel(dataloader = trainloader , optimizer = net_optim , model = net ,
                                scaler = scaler , epoch = epoch , scheduler= scheduler , print_batch= False)
        
        val_loss , preds , targets = EvalModel(dataloader = validloader , model = model)

        preds = np.array(preds)
        targets = np.array(targets)

        auc = metrics.roc_auc_score(targets.ravel() , preds.ravel())

        print(f"Epoch : {epoch} | Train Loss :{train_loss} | Valid Loss : {val_loss}")

        if auc > best_auc:

            print(f"AUC Improved {best_auc} -> {auc}")
            best_auc = auc
    
    print("Saving Model")
    torch.save(model.state_dict() , f'model-eff.pth')

if __name__ == "__main__":

    run_model()