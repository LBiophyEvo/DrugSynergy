

import sys
sys.path.append('../') 
from scripts.model import MLP_drug_cell_permutate
from scripts.evaluation import eval_accuracy 
from scripts.util import DrugCombDataset_customized 
from torch.utils.data import DataLoader
import numpy as np 
from torch import optim, nn , no_grad
from joblib import Parallel, delayed 
from tqdm import tqdm 

def eval_func(model_dict, hparam, val_dataset):
    '''
    Evaluate the models in model_dict on validation dataset 
    '''

    device = hparam['device']
    data_test = DrugCombDataset_customized(val_dataset, hparam['cell_lines'], hparam['mol_mapping'], label_name = hparam['target'])
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

def train_func(model: MLP_drug_cell_permutate, hparam, train_dataset, get_loss = False):

    '''
    Train MLP model 
    '''

    device = hparam['device']
    batch_size = hparam['batch_size']
    if len(train_dataset) < batch_size:
        batch_size = len(train_dataset)
    data_train = DrugCombDataset_customized(train_dataset, hparam['cell_lines'], hparam['mol_mapping'], label_name = hparam['target'])
    data_test = DrugCombDataset_customized(hparam['test_dataset'], hparam['cell_lines'], hparam['mol_mapping'], label_name = hparam['target'])
    load_data_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    load_data_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)
    optimizer  = optim.Adam(model.parameters(), lr=hparam['lr'], weight_decay=hparam['wd'])
    if hparam['task'] == 'reg':
        loss_func = nn.MSELoss()
        best_aoc = -np.inf
    else:
        loss_func = nn.BCELoss()
        best_aoc = 0
    model.to(device)
    best_model = model 
    best_result = None 
    patience = 0 
    loss_dict = {
        'train_loss':[],
        'test_loss':[],
        'epoch': [],
        'test_pred': []
    }
    for epoch in tqdm(range(hparam['max_epoch'])):
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
            test_loss = []
            for data in load_data_test:
                data = [tensor.to(device) for tensor in data]
                drug_a, drug_b, cell, label = data[0], data[1], data[2], data[3]
                test_pred_fit_prob = model(drug_a, drug_b, cell)
                predictions.append(list(test_pred_fit_prob.cpu().detach().numpy()))
                real_labels.append(list(label.cpu().detach().numpy()))
                loss = loss_func(test_pred_fit_prob, label.view(-1,1))
                test_loss.append(loss.item())
            test_pred_fit = np.concatenate(predictions)
            test_fit = np.concatenate(real_labels)
            test_res = eval_accuracy(test_pred_fit, test_fit, hparam['task'])

        loss_dict['epoch'].append(epoch)
        loss_dict['train_loss'].append(np.mean(train_loss))
        loss_dict['test_loss'].append(np.mean(test_loss))
        print(f'epoch = {epoch} with training loss: {np.mean(train_loss)}, test accuracy: {test_res}')
        if test_res[1] > best_aoc:
            best_aoc = test_res[1] 
            best_result = test_res
            best_model = model
            patience = 0
        else:
            patience += 1
        if patience > hparam['patience_max']:
            print('early stop with metric: ', best_result[1])
            break 
    loss_dict['test_pred'] = test_pred_fit 
    loss_dict['test_real_label'] = test_fit
    if not get_loss:
        return best_model, best_result
    else:
        return best_model, best_result, loss_dict



def run_one(train_dataset_initial, left_data, max_fold, hparam, flag = True):
    '''
    Parallel calculation: Ensemble MLP 
    '''
    model_dict = []
    best_result_l = []
    model = MLP_drug_cell_permutate(config = hparam)
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