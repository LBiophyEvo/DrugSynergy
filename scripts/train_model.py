

import sys
sys.path.append('../') 
from scripts.model import MLP_drug_cell_permutate
from scripts.evaluation import calculate_accuracy 
from scripts.util import DrugCombDataset_customized 
from torch.utils.data import DataLoader
import numpy as np 
from torch import optim, nn , no_grad
from joblib import Parallel, delayed 

def eval_func(model_dict, hparam, val_dataset):
    '''
    Evaluate the models in model_dict on validation dataset 
    '''

    device = hparam['device']
    data_test = DrugCombDataset_customized(val_dataset, hparam['cell_lines'], hparam['mol_mapping'])
    batch_size = hparam['batch_size']
    if len(data_test) < batch_size:
        batch_size = len(data_test)
    load_data_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    all_prediction = []
    for model in model_dict:
        predictions = []
        model.to(device)
        model.eval()
        for data in load_data_test:
            data = [tensor.to(device) for tensor in data]
            drug_a, drug_b, cell, label = data[0], data[1], data[2], data[3]
            test_pred_fit_prob = model(drug_a, drug_b, cell)
            predictions.append(list(test_pred_fit_prob.cpu().detach().numpy()))
        
        test_pred_fit = np.concatenate(predictions)
        all_prediction.append(test_pred_fit)
    pred_prob = np.concatenate(all_prediction, axis = 1) 
    std_pred = pred_prob.std(axis = 1) 
    mean_pred = pred_prob.mean(axis = 1) 
    return std_pred, mean_pred

def train_func(model: MLP_drug_cell_permutate, hparam, train_dataset):

    '''
    Train MLP model 
    '''

    device = hparam['device']
    batch_size = hparam['batch_size']
    if len(train_dataset) < batch_size:
        batch_size = len(train_dataset)
    data_train = DrugCombDataset_customized(train_dataset, hparam['cell_lines'], hparam['mol_mapping'])
    data_test = DrugCombDataset_customized(hparam['test_dataset'], hparam['cell_lines'], hparam['mol_mapping'])
    load_data_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    load_data_test = DataLoader(data_test, batch_size=256, shuffle=True)
    optimizer  = optim.Adam(model.parameters(), lr=hparam['lr'], weight_decay=hparam['wd'])
    loss_func = nn.BCELoss()
    best_aoc = 0
    model.to(device)
    for epoch in range(hparam['max_epoch']):
        train_loss = []
        for data in load_data_train:
            data = [tensor.to(device) for tensor in data]
            model.train()
            drug_a, drug_b, cell, label = data[0], data[1], data[2], data[3]
            label_pred = model(drug_a, drug_b, cell)
            loss = loss_func(label_pred, label.view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        with no_grad():
            model.eval()
            predictions = []
            real_labels = []
            for data in load_data_test:
                data = [tensor.to(device) for tensor in data]
                drug_a, drug_b, cell, label = data[0], data[1], data[2], data[3]
                test_pred_fit_prob = model(drug_a, drug_b, cell)
                predictions.append(list(test_pred_fit_prob.cpu().detach().numpy()))
                real_labels.append(list(label.cpu().detach().numpy()))
            test_pred_fit = np.concatenate(predictions)
            test_fit = np.concatenate(real_labels)
            test_res = calculate_accuracy(test_pred_fit, test_fit)
   
        print(f'epoch = {epoch} with training loss: {np.mean(train_loss)}, test accuracy: {test_res}')
        if test_res[1] > best_aoc:
            best_aoc = test_res[1] 
            best_result = test_res
    return model, best_result



def run_one(train_dataset_initial, left_data, max_fold, hparam, flag = True):
    '''
    Parallel calculation: Ensemble MLP 
    '''
    model_dict = []
    best_result_l = []
    model = MLP_drug_cell_permutate(hparam=hparam)
    model.to(hparam['device'])
    r = Parallel(n_jobs=-1)(delayed(train_func)(model, hparam, train_dataset_initial)  for fold_name in range(max_fold))
    for id in range(len(r)):
        model_dict.append(r[id][0])
        best_result = [el for el in r[id][1]]
        best_result_l.append([best_result])
    if flag:
        pred_prob, mean_pred = eval_func(model_dict, hparam, left_data)
    else:
        pred_prob, mean_pred = None, None

    best_result_f = [el for el in np.array(best_result_l).mean(axis = 0)]
    return pred_prob, best_result_f, mean_pred