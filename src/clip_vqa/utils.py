import torch 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np 
from itertools import combinations


def get_optimizer(model_config, model):
    if model_config.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=False)
        return optimizer, lr_scheduler
    else:
        raise NotImplementedError(f"{model_config.optimizer} not implemented")

def accuracy_vqa(df, index, value):
    if value == None:
        return 0
    ans_list = [elem['answer'] for elem in df.iloc[index]['answers']]
    return np.divide(np.sum(np.minimum(np.count_nonzero(np.array(list(combinations(ans_list, 9))) == value, axis=1), 1)), 10)


class Config:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)