# From drugcomb data to extract oneil and almanac data 

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import os

def add_cosmic_ids(drugcomb: pd.DataFrame,
                   read_path: str) -> pd.DataFrame:
    cosmic_ids = pd.read_csv(read_path + 'cellosaurus_cosmic_ids.txt', sep=',', header=None)
    
    cosmic_ids = cosmic_ids.dropna()
    
    mapping_cosmic = dict(zip(cosmic_ids[0], cosmic_ids[1]))
    drugcomb["cosmicId"] = drugcomb['cell_line_name'].map(mapping_cosmic)
    
    return drugcomb

def get_smiles(drugcomb: pd.DataFrame,
               read_path: str
              ) -> pd.DataFrame:

    drugs = pd.read_json(read_path + 'drugs.json')
    
    drugs = drugs[drugs['smiles']!='NULL']
    drugs = drugs[~drugs['smiles'].str.contains("Antibody")]
    
    drugs['smiles'] = drugs['smiles'].apply(lambda x: x.split('; ')[-1])
    drugs['smiles'] = drugs['smiles'].apply(lambda x: x.split(';')[-1])
    
    mapping_smiles = dict(zip(drugs['dname'], drugs['smiles']))

    drugcomb["drug_row_smiles"] = drugcomb['drug_row'].map(mapping_smiles)
    drugcomb["drug_col_smiles"] = drugcomb['drug_col'].map(mapping_smiles)
    drugcomb = drugcomb.dropna(subset=['drug_row_smiles', 'drug_col_smiles'])
    
    return drugcomb

def get_target(drugcomb: pd.DataFrame, synergy_score: str,
               synergistic_threshold: float) -> pd.DataFrame:
    
    def synergy_threshold(value):
        if (value >= synergistic_threshold):
            return 1
        else:
            return 0 
    drugcomb['target'] = drugcomb[f'synergy_{synergy_score}'].apply(synergy_threshold)
    drugcomb = drugcomb.dropna(subset=['target'])
    
    return drugcomb

def formatting(drugcomb: pd.DataFrame) -> pd.DataFrame:
    drugcomb = drugcomb[drugcomb['synergy_loewe']!='\\N']
    drugcomb = drugcomb.dropna(subset=['drug_col','cosmicId'])
    drugcomb['synergy_loewe'] = drugcomb['synergy_loewe'].astype(float)
    drugcomb['cosmicId'] = drugcomb['cosmicId'].astype(int)
    
    return drugcomb

def rename_and_crop(drugcomb: pd.DataFrame, synergy_score = 'loewe') -> pd.DataFrame:
    
    drugcomb = drugcomb.rename(
        columns={"drug_row": "Drug1_ID",
                 "drug_col": "Drug2_ID",
                 "cell_line_name": "Cell_Line_ID",
                 "drug_row_smiles": "Drug1",
                 "drug_col_smiles": "Drug2"})
    
    cols_to_keep = ["Drug1_ID", "Drug2_ID", "Cell_Line_ID", "Drug1", "Drug2", "target", "synergy_" + str(synergy_score), "tissue_name"]
    drugcomb = drugcomb[cols_to_keep].reset_index(drop=True)
    
    return drugcomb



def process_cell_lines(drugcomb: pd.DataFrame,
                      read_path: str
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    rma = pd.read_csv(read_path + 'Cell_line_RMA_proc_basalExp.txt', sep='\t')
    
    cosmic_found = set(drugcomb.cosmicId)
    cosmic_intersect = list(set(["DATA."+str(c) for c in cosmic_found]).intersection(set(rma.columns)))
    
    drugcomb = drugcomb[drugcomb.cosmicId.isin([int(c[len("DATA."):]) for c in cosmic_intersect])]
    
    landmark_genes = pd.read_csv(read_path + 'L1000genes.txt', sep='\t')
    landmark_genes = list(landmark_genes.loc[landmark_genes['Type']=="landmark",'Symbol'])
    
    rma_landm = rma[rma['GENE_SYMBOLS'].isin(landmark_genes)]
    rma_landm = rma_landm.drop(['GENE_SYMBOLS','GENE_title'], axis=1).T
    
    cell_line_mapping = dict(zip('DATA.'+drugcomb['cosmicId'].astype(str),drugcomb['cell_line_name']))
    rma_landm.index = rma_landm.index.map(cell_line_mapping)
    rma_landm = rma_landm[~rma_landm.index.isna()]
    
    scaler = StandardScaler()
    scaler.fit(rma_landm)

    cell_line_feats = pd.DataFrame(
        scaler.transform(rma_landm),
        columns=[f'feat_{i}' for i in range(rma_landm.shape[1])],
        # index=rma_landm.index
    )
    cell_line_feats["cell_line_name"] = rma_landm.index
    return drugcomb, cell_line_feats

def preprocess_drugcomb(synergy_score: str,
                        synergistic_thresh: float,
                        save_cell_lines: bool,
                        read_path: str,
                        save_path: str,
                        study_name: str,
                       ) -> None:

    # get data from https://drugcomb.org/download/ 
    drugcomb = pd.read_csv('../data/init/summary_v_1_5.csv')
    if study_name == "almanac":
        study_name = 'ALMANAC'
    elif study_name == 'oneil':
        study_name = 'ONEIL'

    drugcomb = drugcomb[drugcomb.study_name == study_name]
    print('original data size', len(drugcomb))
    drugcomb = add_cosmic_ids(drugcomb, read_path)
    drugcomb = formatting(drugcomb)
    drugcomb = get_smiles(drugcomb, read_path)
    drugcomb = get_target(drugcomb, synergy_score, synergistic_thresh)

    
    drugcomb, cell_line_feats = process_cell_lines(drugcomb, read_path)
    drugcomb = rename_and_crop(drugcomb, synergy_score = synergy_score)

    drugcomb = drugcomb.reset_index(drop=True)

    directory_name = save_path 
    os.makedirs(directory_name, exist_ok=True)
    dic = dict(((el,'first') if el != 'target' else (el, np.mean) for el in  list(drugcomb.columns)))
    drugcomb = drugcomb.groupby(['Drug1_ID', 'Drug2_ID', 'Cell_Line_ID'], as_index = False).agg(dic)

    drugcomb.to_feather(f"{directory_name}/{synergy_score}.feather")
    print('filtered data size', len(drugcomb))
    if save_cell_lines:
        cell_line_feats = cell_line_feats.reset_index(drop=True)
        cell_line_feats.to_feather(save_path + f"cell_lines.feather")

if __name__ == "__main__":
    print('Preprocessing DrugComb')
    study_name = 'oneil'
    parser = argparse.ArgumentParser(description='Preprocessing Drug Comb')
    parser.add_argument('--synergy_score', type=str, default="loewe")
    parser.add_argument('--synergistic_thresh', type=float, default=10.0)
    parser.add_argument('--save_cell_lines', type=bool, default=True)
    parser.add_argument('--read_path', type=str, default="../data/init/")
    parser.add_argument('--save_path', type=str, default="../data/" + str(study_name) + '/')
    args = parser.parse_args()
    print(args)

    preprocess_drugcomb(
        args.synergy_score,
        args.synergistic_thresh,
        args.save_cell_lines,
        args.read_path,
        args.save_path,
        study_name=study_name
    )

    print("Done")
 