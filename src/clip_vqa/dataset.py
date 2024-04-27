
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import os 
import pandas as pd
from loguru import logger 
import numpy as np 

class VizWizDataset(Dataset):
    def __init__(self, 
                df_path, 
                filter_answerable=True):
        assert os.path.exists(df_path), f"{df_path} does not exists"
        self.df = pd.read_json(df_path)

        # using only Answerable Instances
        if filter_answerable:
            logger.info("Filtering Answerable Question Only")
            self.df = self.df[self.df.answerable == 1]


        self.img_feats = self.df.img_feat
        self.text_feats = self.df.text_feat
        self.answers = self.df.final_answer
        self.anserable = self.df

    def __len__(self):
        return len(self.df)
                
    def __getitem__(self, index):
        img_feat = torch.load(self.df.img_feat.iloc[index])
        text_feat = torch.load(self.df.text_feat.iloc[index])
        feat = torch.cat((img_feat, text_feat), 1).to(torch.float32)

        answer = self.df.final_answer.iloc[index]
        answerability = self.df.answerable.iloc[index]       
        return index, feat, answer, answerability