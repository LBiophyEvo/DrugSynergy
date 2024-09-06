
import pandas as pd
import numpy as np 
import random 
import os 
if __name__ == '__main__':
    # oneil, almanac
    for study_name in ['oneil']:
        data_folder_path = "../data/" + str(study_name) + '/'
        synergy_score = "loewe"
        dataset = pd.read_feather(data_folder_path + f"{synergy_score}.feather")
        for split in ['comb', 'cell', 'one_drug', 'two_drug', 'tissue']:

            if split in ['one_drug', 'two_drug']:
                obj_unique = list(set(list(dataset.Drug1_ID.unique()) + list(dataset.Drug2_ID.unique())))
                max_split = 5
            elif split == 'cell':
                obj_unique = list(dataset.Cell_Line_ID.unique())
                max_split = len(obj_unique)
            elif split == 'comb':
                dataset['drug_combination'] = dataset.Drug1_ID.astype(str) + '_' + dataset.Drug2_ID.astype(str) 
                obj_unique = dataset['drug_combination'].unique()
                max_split = 50 
            elif split == 'tissue':
                obj_unique = list(dataset.tissue_name.unique())
                max_split = len(obj_unique)




            lenth = len(obj_unique)
            random.seed(42)
            
            random_num = random.sample(range(0, lenth), lenth)
            table_split = pd.DataFrame(np.ones((lenth, max_split)), columns= ['fold_' + str(iter) for iter in range(max_split)])
            table_split['obj'] = obj_unique
            # 0 for testing, 1 for training 
            for i_time in range(max_split):
                pot = int(lenth / max_split)
                test_num = random_num[pot * i_time:pot * (i_time + 1)]
                table_split['fold_' + str(i_time)].iloc[test_num] = 0
            path = '../data/splits/' + str(study_name) + '/'
            os.makedirs(path, exist_ok=True)
            table_split.to_csv(path + 'leave_' + str(split) + '.csv', index = False)