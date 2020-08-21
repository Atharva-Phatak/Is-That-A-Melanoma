import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import pandas  as pd
from model import MelanomaModel
from dataset import MelanomaDataset

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def GetPreds(model_path , img_path , backbone):

    df = pd.read_csv('test.csv')
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean = (0.485, 0.456, 0.406) ,
                                    std = (0.229, 0.224, 0.225))])

    test_data = MelanomaDataset(df = df  , path = img_path , transformations  = test_transforms , is_train = False)
    test_loader = torch.utils.data.DataLoader(test_data , batch_size = 16)

    net = MelanomaModel(backbone=  backbone)
    net.load_state_dict(torch.load(model_path))

    preds = []

    with torch.no_grad():

        for img in test_loader:

            if torch.cuda.is_available():

                img = img.cuda()

            logits = net(img)

            preds.extend(logits.cpu().detach().numpy().tolist())
    
    preds = np.array(preds)
    preds = sigmoid(preds)

    return preds