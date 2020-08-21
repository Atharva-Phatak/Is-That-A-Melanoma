import torch
import torch.nn as nn

from PIL import Image

class MelanomaDataset(torch.utils.data.Dataset):

    '''This class retrieves the image and corresponding label for the image.
       Args:
            df : The dataframe which contains the image name and the label.
            path : The path where all the images as saved.
            transformations : The augmentations which you want to perform on the Image.
            is_train : Whether the data is training data or testing data 

        Returns: A an Image Tensor and the correspoding Label tensor.   
    '''

    def __init__(self , df , path , transformations = None , is_train = False):

        super(MelanomaDataset , self).__init__()

        self.df = df
        self.path = path
        self.transformations = transformations
        self.is_train = is_train

    def __len__(self):

        return len(self.df.shape[0])

    def __getitem__(self , idx):

        im_path = self.path + '/' + self.df['image_name'][idx] + '.png'
        img = Image.open(im_path)

        if self.transformations is not None:

            img = self.transformations(img)
        
        if self.is_train:

            target = self.df.iloc[idx]['target']
            return img , torch.tensor(target , dtype = torch.long)
        
        else:

            return img