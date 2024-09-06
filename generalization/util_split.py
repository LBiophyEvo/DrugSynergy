import random 
import pandas as pd 
from typing import Tuple
import numpy as np 


def split_num(dataset,  flag = 'cell', max_paral = 20):
    cells_unique = list(dataset.Cell_Line_ID.unique())
    if flag == 'cell':
        num_train = len(cells_unique)
    elif flag in ['drug', 'one_drug', 'two_drug']:
        num_train = 5
    elif flag == 'comb':
        num_train = 50 
    elif flag == 'tissue':
        num_train = len(dataset.tissue_name.unique())
    run_iter = num_train//max_paral
    if run_iter*max_paral < num_train:
        run_iter += 1
    split_dict = dict(zip(np.arange(run_iter), \
                   np.array_split(np.arange(num_train), run_iter)))

    return num_train, split_dict 
def get_datasets_tissue(fold_number: int,
                 dataset: pd.DataFrame,
                 dataset_split: pd.DataFrame
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    '''Split data into training and test based on tissue'''

    tissue_test = dataset_split['obj'][dataset_split['fold_' + str(fold_number)] == 0].values
    
    train_dataset, test_dataset = dataset[~dataset.tissue_name.isin(tissue_test)], dataset[dataset.tissue_name.isin(tissue_test)]


    return train_dataset, test_dataset


def get_datasets_cells(fold_number: int,
                 dataset: pd.DataFrame,
                 dataset_split: pd.DataFrame
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    '''Split data into training and test based on cell lines'''
    cell_test = dataset_split['obj'][dataset_split['fold_' + str(fold_number)] == 0].values

    train_dataset, test_dataset = dataset[~dataset.Cell_Line_ID.isin(cell_test)], dataset[dataset.Cell_Line_ID.isin(cell_test)]


    return train_dataset, test_dataset


def get_datasets_combinations(fold_number: int,
                 dataset: pd.DataFrame,
                 dataset_split: pd.DataFrame
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    '''Split data into training and test based on drug combinations'''
    dataset['drug_combination'] = dataset.Drug1_ID.astype(str) + '_' + dataset.Drug2_ID.astype(str) 
    drugscomb_test = dataset_split['obj'][dataset_split['fold_' + str(fold_number)] == 0].values
    
    train_dataset, test_dataset = dataset[~(dataset.drug_combination.isin(drugscomb_test))], dataset[dataset.drug_combination.isin(drugscomb_test)]


    return train_dataset, test_dataset

def get_datasets_drugs(fold_number: int,
                 dataset: pd.DataFrame,
                 dataset_split: pd.DataFrame,
                 flag: str,
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    '''Split data into training and test based on drugs'''
    
    drugs_test = dataset_split['obj'][dataset_split['fold_' + str(fold_number)] == 0].values
    drugs_train = dataset_split['obj'][dataset_split['fold_' + str(fold_number)] == 1].values

    if flag == 'one_drug':
        train_dataset = dataset[(dataset.Drug1_ID.isin(drugs_train))&(dataset.Drug2_ID.isin(drugs_train))]
        mask1 = ((dataset.Drug1_ID.isin(drugs_train))&(dataset.Drug2_ID.isin(drugs_test)))
        mask2 = ((dataset.Drug2_ID.isin(drugs_train))&(dataset.Drug1_ID.isin(drugs_test)))
        mask = mask1|mask2
        test_dataset = dataset[mask]
    else:
        train_dataset, test_dataset = dataset[(dataset.Drug1_ID.isin(drugs_train))&(dataset.Drug2_ID.isin(drugs_train))], dataset[(dataset.Drug1_ID.isin(drugs_test))&(dataset.Drug2_ID.isin(drugs_test))]


    return train_dataset, test_dataset

def get_datasets_split(fold_number: int,
                 dataset: pd.DataFrame,
                 dataset_split: pd.DataFrame,
                 flag: str,
                 )-> Tuple[pd.DataFrame, pd.DataFrame]:
    if flag == 'cell':
        train_dataset, test_dataset = get_datasets_cells(fold_number, dataset, dataset_split)
    elif flag == 'one_drug':
        train_dataset, test_dataset = get_datasets_drugs(fold_number, dataset, dataset_split, flag = flag)
    elif flag == 'two_drug':
        train_dataset, test_dataset = get_datasets_drugs(fold_number, dataset, dataset_split, flag = flag)
    elif flag =='comb':
        train_dataset, test_dataset = get_datasets_combinations(fold_number, dataset, dataset_split)
    elif flag == 'tissue':
        train_dataset, test_dataset = get_datasets_tissue(fold_number, dataset, dataset_split)
     
    return train_dataset, test_dataset

