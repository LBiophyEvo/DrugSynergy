
import pandas as pd 
import random 
import numpy as np
import joblib  
import os 
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

def get_datasets_simple(fold_number: int,
                 dataset: pd.DataFrame,
                 step_add: int, 
                 ):


    train_num, test_num = split_train_val(fold_number, len(dataset), step_add) 


    return train_num, test_num

def get_dict(study_name, max_split = 10):
    data_folder_path = "../data/" + str(study_name) + '/'
        
    synergy_score = "loewe"
    
    dataset = pd.read_feather(data_folder_path + f"{synergy_score}.feather")
    step_add = int(len(dataset)/max_split)
    train_index, test_index = get_datasets_simple(fold_number = 0,
                 dataset = dataset,
                 step_add = step_add)
    dict_study= {
        'train_index': train_index,
        'test_index': test_index
    }
    print(study_name, len(test_index))
    print('save data in ', data_folder_path, 'train_test_index.joblib')
    joblib.dump(dict_study, data_folder_path + 'train_test_index.joblib')
    
def subsample_alamanc(dataset, flag_subsample = False, sample_id = 0):
    if flag_subsample:
        cells_unique = list(dataset['Cell_Line_ID'].unique())
        drugs_unique = list(pd.concat([dataset.Drug1_ID, dataset.Drug2_ID]).unique())
        if sample_id == 0:
            cell_selec = cells_unique[:29]
            drugs_selec = drugs_unique[:38]
        else:
            random.seed(sample_id*100)
            cell_selec = random.sample(cells_unique, k = 29)
            drugs_selec = random.sample(drugs_unique, k= 38)
        # cell_selec = cells_unique[-30:-1]
        # drugs_selec = drugs_unique[38:75]
        dataset_sub = dataset[(dataset.Cell_Line_ID.isin(cell_selec))&(dataset.Drug1_ID.isin(drugs_selec))&(dataset.Drug2_ID.isin(drugs_selec))]
    else:
        dataset_sub = dataset
    return dataset_sub

if __name__ == '__main__':

    for sample_id, study_name in enumerate(['oneil']):
        if study_name not in ['oneil', 'almanac']:
            data_folder_path = "../data/preprocessed/almanac/"
            synergy_score = "loewe"
            cell_lines = pd.read_feather(data_folder_path + "cell_lines.feather").set_index("cell_line_name")
            cell_lines = cell_lines.astype(np.float32)
            dataset = pd.read_feather(data_folder_path + f"{synergy_score}/{synergy_score}.feather")
            dataset = subsample_alamanc(dataset, flag_subsample = True, sample_id=sample_id)
            save_folder = "../data/preprocessed/" + str(study_name) + f"/{synergy_score}/"
            os.makedirs(save_folder, exist_ok=True)
            dataset.to_feather(save_folder + f"{synergy_score}.feather")
        if study_name == 'almanac':
            max_split = 100
        else:
            max_split = 10
        dict_study = get_dict(study_name=study_name, max_split = max_split)



    


