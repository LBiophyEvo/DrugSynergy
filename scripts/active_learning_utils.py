import pandas as pd 
import random 
import numpy as np 
import joblib 
def calculate_derivative(results):
    '''
    Get the derivative of PR-AUC 
    '''
    derev_l = []
    for id,res in enumerate(results[:-1]):
        derev = results[id+1][0][1] - results[id][0][1]
        derev_l.append(derev)
    if len(derev_l) >1:
        derev_l_two = [np.mean(derev_l[:(id+1)]) for id in range(len(derev_l)-1) ]
        d_pos = np.nan 
        for id in range(len(derev_l_two)-1):
            if derev_l_two[id+1] < derev_l_two[id]:
                d_pos = derev_l_two[id]
                break
        d_current = derev_l_two[-1] 
    else:
        d_pos = np.nan 
        d_current = np.mean(derev_l)
    return d_pos, d_current 



def func_lambd(d_pos, d_current, flag):
    '''
    Define lambda in the acquitance function: \alpha  \sigma(S)  + (1-\alpha) S
    '''
    if flag == 'exploration':
        lambd = 1
    elif flag == 'exploitation':
        lambd = 0 
    elif flag == 'hybrid':
        if np.isnan(d_pos):
            lambd = 1
        else:
            lambd = 1/d_pos*d_current
    else:
        lambd = 1
    return lambd 



def sample_constrain(index_all, dataset, step_add):
    '''
    Select the samples such that the number of cells is no more than 5, samples in each cell lines is no more than step_add. 
    '''
    dict_cell = {}
    count = 0 
    for idx in index_all:
        cell = dataset['Cell_Line_ID'].iloc[idx]
        if count < step_add:
            if (len(dict_cell)<5):
                if cell not in dict_cell:
                    dict_cell[cell] = []
                    dict_cell[cell].append(idx)
                    count += 1 
                else:
                    dict_cell[cell].append(idx)
                    count += 1
            else:
                if cell in dict_cell:
                    dict_cell[cell].append(idx)
                    count += 1


    test_num = [x for el in dict_cell for x in dict_cell[el]]

    train_num = [id for id in index_all if id not in test_num]
    return test_num, train_num

def sample_constrain_strong(index_all, dataset, step_add, cells):
    dict_cell = dict([(cell, []) for cell in cells])
    num = int(step_add/5)
    for idx in index_all:
        cell = dataset['Cell_Line_ID'].iloc[idx]
        if cell in dict_cell:
            if len(dict_cell[cell])<num:
                dict_cell[cell].append(idx)

    test_num = [x for el in dict_cell for x in dict_cell[el]]

    train_num = [id for id in index_all if id not in test_num]
    return test_num, train_num

def split_train_val(i_time, lenth, pot):
    random.seed(42 + i_time)
    random_num = random.sample(range(0, lenth), lenth)
    if pot <= lenth:
        test_num = random_num[:pot]
        train_num = random_num[pot:]
    else:
        test_num = random_num
        train_num = None 
    return train_num, test_num

def get_datasets_simple(study_name: str, dataset: pd.DataFrame):

    data_folder_path = "./data/" + str(study_name) + '/' 
    split_dict = joblib.load(data_folder_path + 'train_test_index.joblib')
    train_num, test_num = split_dict['train_index'], split_dict['test_index']

    
    train_dataset, test_dataset, flag = dataset.iloc[train_num], dataset.iloc[test_num], True

    return train_dataset, test_dataset, flag 

def get_datasets_simple_constraint(fold_number: int,
                 dataset: pd.DataFrame,
                 step_add: int, 
                 ):

    lenth = len(dataset)
    random.seed(42 + fold_number)
    random_num = random.sample(range(0, lenth), lenth)
    test_num, train_num = sample_constrain(random_num, dataset, step_add)
    if train_num != None:
        train_dataset, test_dataset, flag = dataset.iloc[train_num], dataset.iloc[test_num], True
    else:
        train_dataset, test_dataset, flag = None, dataset.iloc[test_num], False

    return train_dataset, test_dataset, flag 

def get_datasets_simple_constraint_v2(fold_number: int,
                 dataset: pd.DataFrame,
                 step_add: int,
                 study_name: str):
    if study_name == 'oneil':
        fda_approved_drugs = ['metformin', 'methotrexate', 'temozolomide', 'mitomycin C', '5-Fluorouracil', 'Lapatinib', 'Erlotinib',\
                        'paclitaxel', 'vinblastine', 'vinorelbine', 'Sorafenib', 'Dasatinib', 'Sunitinib', 'dexamethasone', 'Bortezomib', \
                            'gemcitabine', 'topotecan', 'doxorubicin', 'etoposide', 'Eloxatin (TN)', 'Vorinostat', '7-Ethyl-10-hydroxycamptothecin']
    elif study_name == 'dream':
        fda_approved_drugs = ['Azacytidine', 'Azacytidine', 'Carboplatin', 'CarboTaxol', 'CarboTaxol', 'Chloroquine', 'Chloroquine', \
                              'Cisplatin', 'Cisplatin', 'Docetaxel', 'EGFR', 'EGFR_2', 'EGFR_2', 'ESR1_1', 'ESR1_1', 'ESR1_2', 'ESR1_3', \
                                'FNTA_FNTB', 'FOLFIRI', 'FOLFIRI', 'FOLFIRI', 'FOLFOX', 'FOLFOX', 'FOLFOX', 'Gemcitabine', 'HDAC_4', 'HDAC_4', \
                                    'MAP2K', 'MAP2K', 'MAP2K_1', 'MET_ALK', 'MTOR_2', 'MTOR_2', 'MTOR_3', 'MTOR_3', 'Oxaliplatin', 'PARP1', \
                                        'PI3KCD', 'PRKC', 'Proteasome', 'Proteasome_2', 'Proteasome_2', 'TKI', 'TKI', 'TKI', 'TKI_2', 'TKI_2',\
                                              'TOP2_2', 'Topotecan', 'TYMS', 'TYMS', 'Vinorelbine']

    elif study_name == 'almanac':
        fda_approved_drugs = ['allopurinol', 'celecoxib', 'CYTARABINE HYDROCHLORIDE', 'Sorafenib', 'IMIQUIMOD', 'letrozole', 'Imatinib', 'methotrexate', 'Procarbazine hydrochloride', 'Carboplatinum', \
                     'Tamoxifen citrate', 'Azacytidine, 5-', 'raloxifene', 'anastrozole', 'Retinoic acid', 'Eloxatin (TN) (Sanofi Synthelab)', 'paclitaxel','Fulvestrant', 'Nilotinib', 'Pemetrexed', 'Pralatrexate',\
                     'actinomycin D', 'mitoxantrone', 'MITHRAMYCIN', '158798-73-3', '122111-05-1', 'ifosfamide', 'bleomycin', 'temozolomide', 'Vincristine sulfate', 'lomustine', 'Bendamustine hydrochloride', 'TOPOTECAN HYDROCHLORIDE',\
                     'thiotepa', 'ADM hydrochloride', 'mitotane', 'NSC733504', 'Sunitinib', 'Dexrazoxane', 'Melphalan hydrochloride', '5-Aminolevulinic acid hydrochloride', 'busulfan', '55-86-7', '6-Thioguanine',
                     'thalidomide', 'docetaxel', 'amifostine', 'Zoledronic acid', 'Vinblastine sulfate', 'dacarbazine', 'hydroxyurea', 'Vepesid J', '6-Mercaptopurine', 'Gefitinib', 'NSC-127716', 'teniposide', 'mitomycin C', 'methoxsalen',\
                     '5-Fluorouracil', 'cyclophosphamide', 'Lenalidomide', 'Zanosar', 'carmustine', 'chlorambucil', 'Erlotinib hydrochloride', 'MEGESTROL ACETATE', 'EXEMESTANE', 'altretamine', 'Bortezomib', 'Emcyt (Pharmacia)', \
                     'Ixabepilone', 'cis-Platin', 'Pazopanib hydrochloride', 'Vemurafenib', 'Abiraterone', 'CABAZITAXEL', 'Vismodegib', 'Crizotinib', 'Axitinib', 'Ruxolitinib', 'Vandetanib', 'Trisenox', 'NSC256439']
    else:
        fda_approved_drugs = []
    dataset = dataset.reset_index()
    if len(fda_approved_drugs) >0:
        dataset_sub_index1 = dataset[(dataset.Drug1_ID.isin(fda_approved_drugs))&(~dataset.Drug2_ID.isin(fda_approved_drugs))].index
        dataset_sub_index2 = dataset[(dataset.Drug2_ID.isin(fda_approved_drugs))&(~dataset.Drug1_ID.isin(fda_approved_drugs))].index
        index_initial = list(dataset_sub_index1) + list(dataset_sub_index2)
        test_num, _ = sample_constrain(index_initial, dataset, step_add)
        train_num = [inde for inde in range(len(dataset)) if inde not in test_num]
    else:
        lenth = len(dataset)
        random.seed(42 + fold_number)
        random_num = random.sample(range(0, lenth), lenth)
        test_num, train_num = sample_constrain(random_num, dataset, step_add)
    if train_num != None:
        train_dataset, test_dataset, flag = dataset.iloc[train_num], dataset.iloc[test_num], True
    else:
        train_dataset, test_dataset, flag = None, dataset.iloc[test_num], False

    return train_dataset, test_dataset, flag 
 


def selec_data_index(pred_prob, index_sort, left_data, step_add):
    def get_ratio(values):
        index_sort_t = np.argsort(values)[::-1]
        num = int(step_add/5)
        if len(values) < num:
            num = -1
        return np.mean(values.iloc[index_sort_t[:num]])
    left_data['pred'] = pred_prob
    group_data = left_data.groupby('Cell_Line_ID', as_index = False)['pred'].agg(get_ratio)
    group_data = group_data.sort_values(by = 'pred', ascending=False)
    if len(group_data) < 5:
        end_id_c = -1
    else:
        end_id_c = 5 
    cells = list(group_data['Cell_Line_ID'].iloc[:end_id_c])
    print(cells)
    left_data = left_data.drop(columns=['pred'])
    index_selec, index_no_selec = sample_constrain_strong(index_sort, left_data, step_add, cells)
    
       
    return index_selec, index_no_selec 

def selec_data(pred_prob, left_data, step_add, flag_early_stop = False, num_step = 1, num_step_limit = 4):
    pred_prob = np.array(pred_prob).flatten()
    left_counts = left_data.groupby('Cell_Line_ID', as_index = False)['Drug1_ID'].count()
    if not flag_early_stop:
        if (left_counts.Drug1_ID.all() < int(step_add/5))&(len(left_counts)<5):
            data_select = left_data
            data_no_selec = None 
            flag = False
        else:
            index_sort = np.argsort(pred_prob)[::-1]
            index_selec, index_no_selec = selec_data_index(pred_prob, index_sort, left_data, step_add)
            data_select = left_data.iloc[index_selec]
            data_no_selec = left_data.iloc[index_no_selec]
            flag = True 
    else:
        index_sort = np.argsort(pred_prob)[::-1]
        index_selec, index_no_selec = selec_data_index(pred_prob, index_sort, left_data, step_add)
        data_select = left_data.iloc[index_selec]
        data_no_selec = left_data.iloc[index_no_selec]
        if num_step > num_step_limit:
            flag = False
        else:
            flag = True

    return data_no_selec, data_select, flag 
