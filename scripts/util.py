import torch
from torch.utils.data import Dataset
import random 
import pandas as pd 
from typing import Tuple
import numpy as np 
import tqdm 
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, roc_curve,  
                             precision_score, recall_score, auc, cohen_kappa_score,
                             balanced_accuracy_score, precision_recall_curve, accuracy_score)

from sklearn.metrics import r2_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

DRUG1_ID_COLUMN_NAME = "Drug1_ID"
DRUG2_ID_COLUMN_NAME= "Drug2_ID"
CELL_LINE_COLUMN_NAME = "Cell_Line_ID"


class DrugCombDataset_customized(Dataset):

    '''Dataset class that returns all model inputs and a target'''

    def __init__(self, drugcomb, cell_lines, chembert_mapping, label_name = 'target'):
        self.drugcomb = drugcomb
        self.chembert_mapping = chembert_mapping 
        self.cell_lines = cell_lines 
        self.targets = torch.from_numpy(drugcomb[label_name].values) 


    def __len__(self):
        return len(self.drugcomb)

    def __getitem__(self, idx):
        sample = self.drugcomb.iloc[idx]

        drug1 = sample[DRUG1_ID_COLUMN_NAME]
        drug2 = sample[DRUG2_ID_COLUMN_NAME]
        drug1_tokens = self.chembert_mapping[drug1]
        drug2_tokens = self.chembert_mapping[drug2]


        cell_line_name = sample[CELL_LINE_COLUMN_NAME]
        cell_line_embeddings = self.cell_lines.loc[cell_line_name].values.flatten()
        cell_line_embeddings = torch.tensor(cell_line_embeddings)

        target = self.targets[idx].unsqueeze(-1).float()
        
        return (drug1_tokens, drug2_tokens, cell_line_embeddings, target)




def split_train_val(i_time, lenth, max_split = 10):
    '''
    randomly select 10% of data as testing data 
    '''
    random.seed(42)
    random_num = random.sample(range(0, lenth), lenth)
    pot = int(lenth / max_split)
    test_num = random_num[pot * i_time:pot * (i_time + 1)]
    train_num = random_num[:pot * i_time] + random_num[pot * (i_time + 1):]
    return train_num, test_num

def get_datasets_simple(cell_lines: pd.DataFrame,
                 fold_number: int,
                 dataset: pd.DataFrame,
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    '''Prepares all input datasets for the model based on the evaluation setup and synergy score'''


    train_num, test_num = split_train_val(fold_number, len(dataset)) 
    train_dataset, test_dataset = dataset.iloc[train_num], dataset.iloc[test_num]


    return dataset, train_dataset, test_dataset, cell_lines


