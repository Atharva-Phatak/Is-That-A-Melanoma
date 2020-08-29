import torch
import torch.nn as nn
from torch.cuda import amp
import numpy as np

from loss import focal_binary_cross_entropy


def TrainModel(dataloader , optimizer , model , scaler , epoch , scheduler = None , print_batch = False):

    '''This function implements the training loop
       Args:
            dataloader : The training dataloader
            optimizer : The selected optimizer for training
            model : The model used for training
            scaler : The gradient Scaler for mixed precision Training(Only Works for Pytorch : 1.6)
            epoch : The current epoch
            scheduler : The learning rate scheduler
            print_batch : Do you want to see loss per 500 batches ?

        Returns: This function returns the training loss per epoch.
    '''

    model.train()

    train_loss = []
    for bi , (img , target) in enumerate(dataloader):

        if torch.cuda.is_available():

            img = img.cuda()
            target = target.cuda()

        optimizer.zero_grad()

        with amp.autocast():

            logits = model(img)

            f_loss = focal_binary_cross_entropy(logits , target.view(-1,1).type_as(logits))

        scaler.scale(f_loss)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step(epoch + i/len(dataloader))
        train_loss.append(f_loss.item())

        if print_batch:

            if bi % 500 == 0:

                print('Epoch : {epoch} || Batch : {bi} || Loss : {f_loss:.4f}')


    return np.mean(train_loss)


def EvalModel(dataloader , model):

    '''This function implements the training loop for evaluation for you validation set
       Args:
            dataloder : The validation Dataloader
            model : The model which will be used for evaluation

       Returns : The validation Loss and Final Predicition Logits
    '''
   
    model.eval()

    val_loss = []
    targets = []
    preds = []

    with torch.no_grad():

        for img , target in dataloader:

            img = img.cuda()
            target = target.cuda()

        
            logits = model(img)
            f_loss = focal_binary_cross_entropy(logits , target.view(-1,1).type_as(logits))

            val_loss.append(f_loss.item())
            preds.extend(logits.cpu().detach().numpy().tolist)
            targets.extend(target.cpu().detach().numpy().tolist())

    
    return np.mean(val_loss) , preds , targets
