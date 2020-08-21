import pandas as pd
import numpy as np

def get_train_val_split(df):
    #Remove Duplicates
    df = df[df.tfrecord != -1].reset_index(drop=True)
    #We are splitting data based on triple stratified kernel provided here https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526
    train_tf_records = list(range(len(df.tfrecord.unique())))[:12]
    split_cond = df.tfrecord.apply(lambda x: x in train_tf_records)
    train_df = df[split_cond].reset_index()
    valid_df = df[~split_cond].reset_index()
    
    return train_df,valid_df