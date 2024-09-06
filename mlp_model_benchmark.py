# load model
from scripts.model import MLP_drug_cell_permutate
from scripts.util import DrugCombDataset_customized, get_datasets_simple
from scripts.evaluation import calculate_accuracy
from scripts.train_model import train_func
from torch.utils.data import DataLoader
from torch import optim, nn, no_grad
import scanpy as sc 
import random 
from sklearn import metrics
import numpy as np 
import joblib 
from joblib import Parallel, delayed 
import torch 
import pandas as pd 
import random 
import argparse 
import os 


class run_one:
    def __init__(self, hparam):
        self.hparam = hparam
        

    def train_one(self, frac, itme):
        random.seed(42 + itme) 
        model = MLP_drug_cell_permutate(hparam=self.hparam)
        model.to(self.hparam['device'])
        train_dataset_tmp = self.hparam['train_dataset']
        # select fraction of traning data 
        select = random.choices(range(len(train_dataset_tmp)), k = int(len(train_dataset_tmp)*frac))
        train_dataset = train_dataset_tmp.iloc[select]
        test_dataset = self.hparam['test_dataset']
        print(len(train_dataset), len(test_dataset))
        model, best_result = train_func(model, self.hparam, train_dataset)

        return best_result

if __name__ == '__main__':
    # drug_feature can choose among  ['morgan', 'onehot', 'chembert_384', 'rdkit', 'map4', 'maccs' ]
    # operation can choose among  ['bilinear', 'additive' , 'max', 'multiply', 'sort', 'other'] 
    # cell_feature can choose among ['onehot', 'ge]
    # frac: define the percentage of data to train the model
    # num_run: number of replicates of running code 
    parser = argparse.ArgumentParser(description='Train a mlp model')
    parser.add_argument('--study_name', type=str, default='oneil')
    # 10 random cross validation 
    parser.add_argument('--num_run', type=int, default=50)
    # how much percentage of data for training 
    parser.add_argument('--frac', type=float, default=0.11)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--folder', type=str, default='results')
    parser.add_argument('--drug_feature', type=str, default='morgan')
    parser.add_argument('--cell_feature', type=str, default='ge')
    parser.add_argument('--operation', type=str, default='additive')
    parser.add_argument('--batch_size', type=int, default=256)

    config = parser.parse_args()
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f'device {device}; drug feature: {config.drug_feature}; cell_feature: {config.cell_feature}; operation: {config.operation}')
    print(device)

    # ----------------------- step 1: load data-----------------------
    data_folder_path = "./data/" + str(config.study_name) + '/'
    save_folder = config.folder + "/" + str(config.study_name) + '/'
    os.makedirs(save_folder, exist_ok=True)

    # ---- load cell line and drug features -----------------
    cell_lines = pd.read_feather(data_folder_path + "cell_lines_" + str(config.cell_feature) + ".feather").set_index("cell_line_name")
    cell_lines = cell_lines.astype(np.float32)
    mol_mapping = joblib.load(data_folder_path + 'mol_' + str(config.drug_feature) +'_dic.joblib')

    dataset = pd.read_feather(data_folder_path + f"loewe.feather")
    dataset, train_dataset_tmp, test_dataset, cell_lines = get_datasets_simple(cell_lines, fold_number = 0, dataset = dataset)
    print('percentage of positive samples in testing data', test_dataset.target.sum()/len(test_dataset))
    drug_dim = mol_mapping['shape']
    cell_dim = cell_lines.shape[1]
    print('drug_dim and cell_dim: ', drug_dim, cell_dim)



    # ----------------------- step 2: define hyperparameter and train -----------------------


    hparam = {
        'drug_dim': drug_dim, 
    'cell_dim': cell_dim,
    'hid_dim_1': 2**8,
    'hid_dim': 2**7,
    'lr': 1e-4,
    'wd': 1e-6,
    'max_epoch': 100,
    'device': device,
    'train_dataset': train_dataset_tmp,
    'test_dataset': test_dataset,
    'cell_lines': cell_lines, 
    'mol_mapping': mol_mapping,
    'drugs_feature': config.drug_feature,
    'cell_feature': config.cell_feature,
    'operation': config.operation,
    'batch_size': config.batch_size
    }

    iter_run_one = run_one(hparam=hparam)

    frac = config.frac 
    results_leave = {} 
    r = Parallel(n_jobs=-1)(delayed(iter_run_one.train_one)(frac, iter) for iter in range(config.num_run))
    for id in range(len(r)):
        results_leave[(frac, id)] = r[id]
    joblib.dump(results_leave, save_folder + str(config.operation) + '_' + str(config.drug_feature) + '_cell_' + str(config.cell_feature) + '_frac_' + str(frac) + '.joblib')

    print(results_leave)
 