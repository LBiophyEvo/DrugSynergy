
# evaluate the accuracy 
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, roc_curve,  
                             precision_score, recall_score, auc, cohen_kappa_score,
                             balanced_accuracy_score, precision_recall_curve, accuracy_score)

from sklearn.metrics import r2_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr


import numpy as np
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def calculate_accuracy(test_prob, test_label, thred = 0.75):
    '''
    calculate the accuracy of model: ['ROC-AUC', 'PR-AUC', 'ACCURACY', 'PRECISION', 'RECALL', 'RECALL when PRECISION if 0.75']
    '''
    pred_label = np.array([1 if x > 0.5 else 0 for x in test_prob])
    ACC = accuracy_score(test_label, pred_label)
    if test_label.sum() > 0: 
        if pred_label.sum() >0:
            PREC = precision_score(test_label, pred_label)
            TPR = recall_score(test_label, pred_label)
            

            precision, recall, threshold2 = precision_recall_curve(test_label, test_prob)
            idx = find_nearest(precision, thred)
            return [roc_auc_score(test_label, test_prob), auc(recall, precision), ACC, PREC, TPR, recall[idx]]
        else:
            precision, recall, threshold2 = precision_recall_curve(test_label, test_prob)
            return [roc_auc_score(test_label, test_prob), auc(recall, precision), ACC, 0, 0, 0]
        
    else:
        return [0, 0, ACC, 0, 0, 0]



def calculate_accuracy_deepdds(test_prob, test_label, pred_label, thred = 0.75):
    '''
    calculate the accuracy of model: ['ROC-AUC', 'PR-AUC', 'ACCURACY', 'PRECISION', 'RECALL', 'RECALL when PRECISION if 0.75']
    '''
    ACC = accuracy_score(test_label, pred_label)
    if test_label.sum() > 0: 
        if pred_label.sum() >0:
            PREC = precision_score(test_label, pred_label)
            TPR = recall_score(test_label, pred_label)
            

            precision, recall, threshold2 = precision_recall_curve(test_label, test_prob)
            idx = find_nearest(precision, thred)
        
            return [roc_auc_score(test_label, test_prob), auc(recall, precision), ACC, PREC, TPR, recall[idx]]
        else:
            precision, recall, threshold2 = precision_recall_curve(test_label, test_prob)
            return [roc_auc_score(test_label, test_prob), auc(recall, precision), ACC, 0, 0, 0]
        
    else:
        return [0, 0, ACC, 0, 0, 0]
    

def calculate_accuracy_regression(predict_fit, test_fit):
    '''
    calculate the accuracy of model on regression task: 
    ['Pearson', 'Spearman', 'R2 score']
    '''
    pear = pearsonr(predict_fit.flatten(), test_fit.flatten())[0]
    spear = spearmanr(predict_fit.flatten(), test_fit.flatten())[0]
    r2 = r2_score(predict_fit.flatten(), test_fit.flatten())
    return [pear, spear, r2]