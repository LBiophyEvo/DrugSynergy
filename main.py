# run models for several times 
# this method first try exploration then exploitation 
from scripts.model import MLP_drug_cell_permutate
from scripts.evaluation import calculate_accuracy 
from scripts.util import DrugCombDataset_customized 
from scripts.active_learning_utils import get_datasets_simple_constraint_v2, selec_data, get_datasets_simple, func_lambd, calculate_derivative
from scripts.train_model import run_one 
import joblib 
import numpy as np 
import pandas as pd 
import argparse 
import os 
import torch 
import random 
import matplotlib.pyplot as plt 

    
parser = argparse.ArgumentParser(description='Train a mlp model')
parser.add_argument('--num_step_limit', type=int, default=10)
parser.add_argument('--study_name', type=str, default='oneil')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_drugs', type=int, default=50)
parser.add_argument('--flag_stop', type = bool, default=True)
# flag_acquisition: exploitation, exploration, random 
parser.add_argument('--flag_acquisition', type = str, default='exploitation')
parser.add_argument('--batch_size', type = int, default= 32)
parser.add_argument('--drug_feature', type=str, default='morgan')
parser.add_argument('--cell_feature', type=str, default='ge')
parser.add_argument('--operation', type=str, default='additive')


config = parser.parse_args()
device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

data_folder_path = "./data/" + str(config.study_name) + '/'
save_folder = "results/sample_" + str(config.num_drugs) + "/" + str(config.study_name) + '/'


os.makedirs(save_folder, exist_ok=True)
synergy_score = "loewe"
# ------------------------- Load data -----------------------------------
# load data, cell feature and drug features 

dataset = pd.read_feather(data_folder_path + f"/{synergy_score}.feather")
cell_lines = pd.read_feather(data_folder_path + "cell_lines_" + str(config.cell_feature) + ".feather").set_index("cell_line_name")
cell_lines = cell_lines.astype(np.float32)
mol_mapping = joblib.load(data_folder_path + 'mol_' + str(config.drug_feature) +'_dic.joblib')

train_dataset_tmp, val_dataset, flag =  get_datasets_simple(study_name = config.study_name, dataset =dataset)

drug_dim = mol_mapping['shape']
cell_dim = cell_lines.shape[1]

print('drug_dim and cell_dim: ', drug_dim, cell_dim)
print(device)
print(config.study_name, config.flag_stop, config.flag_acquisition)

# ------------------------- define hyperparameters -----------------------------------

hparam = {
    'drug_dim': drug_dim, 
'cell_dim': cell_dim,
'hid_dim_1': 2**8,
'hid_dim': 2**7,
'lr': 1e-4,
'wd': 1e-6,
'max_epoch': 100,
'device': device,
'test_dataset': val_dataset,
'num_drug': config.num_drugs,
'batch_size': config.batch_size,
'cell_lines': cell_lines, 
'mol_mapping': mol_mapping,
'drugs_feature': config.drug_feature,
'cell_feature': config.cell_feature,
'operation': config.operation 
}
# initial training 
final_results = {}
# ------------------------- run the active learning for 5 times -------------------------------------
for iter in range(5):
    left_data, train_dataset_initial, flag = get_datasets_simple_constraint_v2(fold_number = 0, dataset = train_dataset_tmp, step_add= config.num_drugs, study_name=config.study_name)
    max_fold = 5 
    sigma_pred, results, mean_pred = run_one(train_dataset_initial, left_data, max_fold, hparam)
    step_results = results + [len(train_dataset_initial)] + [train_dataset_initial.target.sum(), train_dataset_initial.target.sum()]
    print(step_results)
    results_all = [step_results]
    step = 1 
    lambd =  func_lambd(np.nan, np.nan, config.flag_acquisition)
    while flag:
        if config.flag_acquisition != 'random':
            pred_comb = lambd*sigma_pred + (1- lambd) * mean_pred 
        else:
            pred_comb = [random.random() for el in range(len(left_data))] 

        
        left_data , train_dataset_add, flag = selec_data(pred_comb, left_data,  config.num_drugs, flag_early_stop=config.flag_stop, num_step = step, num_step_limit =config.num_step_limit)
        

        train_dataset_initial = pd.concat([train_dataset_initial, train_dataset_add])
        sigma_pred, results, mean_pred = run_one(train_dataset_initial, left_data, max_fold, hparam, flag = flag)
        step_results = results + [len(train_dataset_initial)]+ [train_dataset_initial.target.sum(), train_dataset_add.target.sum()]
        print(step_results)
        print('add N data: ', len(train_dataset_add))
        results_all += [step_results]
        d_pos, d_current = calculate_derivative(results_all)  
        lambd = func_lambd(d_pos, d_current, config.flag_acquisition)
        print(d_pos, d_current, lambd)
        step += 1

    
    final_results[iter] = results_all
joblib.dump(final_results, save_folder + str(config.flag_acquisition) + '.joblib')







    
