import sys 
sys.path.append('../')
from scripts.model import MLP_drug_cell_permutate
from scripts.evaluation import calculate_accuracy 
from scripts.util import DrugCombDataset_customized 
from scripts.active_learning_utils import get_datasets_simple_constraint_v2, selec_data, get_datasets_simple, func_lambd, calculate_derivative
from scripts.train_model import train_func
from util_split import get_datasets_split, split_num
import joblib 
import numpy as np 
import pandas as pd 
import argparse 
import os 
import torch 
import random 
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed 

class run_one:
    def __init__(self, hparam, config):
        self.hparam = hparam
        self.config = config  
    
    def train_one(self, itme):
        
        model = MLP_drug_cell_permutate(hparam=self.hparam)
        model.to(self.hparam['device'])
        dataset_all = self.hparam['dataset']
        random.seed(42+itme)
        train_dataset, test_dataset = get_datasets_split(fold_number=itme, dataset=dataset_all, 
                                                         dataset_split= self.hparam['data_split_df'],
                                                         flag=self.config.flag)
        print(len(train_dataset), len(test_dataset))
        if len(test_dataset) > 1:
            self.hparam['test_dataset'] = test_dataset
            model, best_result = train_func(model, self.hparam, train_dataset)

            ration_pos = test_dataset.target.sum()/len(test_dataset)
            best_result = best_result + [ration_pos]
        else:
            best_result = []
        return best_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a mlp model')
    # flag : can be cell, one_drug, two_drug, comb, tissue
    parser.add_argument('--flag', type=str, default='one_drug')
    parser.add_argument('--study_name', type=str, default='oneil')
    parser.add_argument('--feature_name', type=str, default='morgan')
    parser.add_argument('--operation', type=str, default='additive')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--cell_feature', type=str, default='ge')
    parser.add_argument('--batch_size', type=int, default=256)


    config = parser.parse_args()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f'-------use operation: {config.operation }------------')
    print(device)
    data_folder_path = "../data/" + str(config.study_name) + '/'
    save_folder = "results_" + str(config.flag) + "/" + str(config.study_name) + '/'
    os.makedirs(save_folder, exist_ok=True)
    synergy_score = "loewe"
    cell_lines = pd.read_feather(data_folder_path + "cell_lines_" + str(config.cell_feature) + ".feather").set_index("cell_line_name")
    cell_lines = cell_lines.astype(np.float32)
    dataset = pd.read_feather(data_folder_path + f"{synergy_score}.feather")
    cells_unique = list(dataset.Cell_Line_ID.unique())
    mol_mapping = joblib.load(data_folder_path + 'mol_' + str(config.feature_name) +'_dic.joblib')
    
    drug_dim = mol_mapping['shape']
    print(drug_dim)

    data_split_guide = pd.read_csv('../data/splits/' + str(config.study_name) + '/' + 'leave_' + str(config.flag) + '.csv')

    num_train, split_dict = split_num(dataset=dataset, flag=config.flag, max_paral=5)
   
    hparam = {
    'drug_dim': drug_dim,
    'cell_dim': 908,
    'hid_dim_1': 2**8,
    'hid_dim': 2**7,
    'lr': 1e-4,
    'wd': 1e-6,
    'max_epoch': 100,
    'device': device,
    'dataset': dataset,
    'cell_lines': cell_lines, 
    'mol_mapping': mol_mapping,
    'drugs_feature': config.feature_name,
    'operation': config.operation,
    'data_split_df': data_split_guide, 
    'batch_size': config.batch_size
    }

    
    iter_run_one = run_one(hparam, config=config)
    for id_run, el in enumerate(list(split_dict.keys())):
        run_arange = split_dict[el]
        
        r = Parallel(n_jobs=-1)(delayed(iter_run_one.train_one)(iter) for iter in run_arange)
    
        joblib.dump(r, save_folder + 'mlp_' + str(config.operation) + '_' + str(config.feature_name) + '_' + str(id_run) + '.joblib')