
import pandas as pd 
import sys
sys.path.append('../') 
from tqdm import tqdm 
from congfu.utils import get_drug_chembert_384
# "maccs", "morgan-3", "rdkit-7", "map4"
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit import DataStructs
import numpy as np 
import torch
import joblib 
from typing import Union, Callable, Generator
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from map4 import MAP4Calculator
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer


chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

DICT_FP_NAMES = dict(
    maccs = MACCSkeys.GenMACCSKeys,
    morgan = rdFingerprintGenerator.GetMorganGenerator,
    rdkit = rdFingerprintGenerator.GetRDKitFPGenerator,
    map4 = MAP4Calculator
)
DICT_DEFAULT_VISION = dict(
    maccs = None,
    morgan = 2,
    rdkit = 5,
    map4 = 2,
)

def get_drug_chembert_384(smiles: str):
    chemberta._modules["lm_head"] = torch.nn.Identity() 

    chemberta.eval()
    with torch.no_grad():
       
        encoded_input = tokenizer(smiles, return_tensors="pt",padding=True,truncation=True)
        model_output = chemberta(**encoded_input)
        
        emb = torch.mean(model_output[0],1)
            
    return emb.flatten()

def get_feature_dict(df: pd.DataFrame, feature_name: str) -> dict:

    '''Returns the mapping between IDs and molecular graphs'''

    mols = pd.concat([
        df.rename(columns={'Drug1_ID': 'id', 'Drug1': 'drug'})[['id', 'drug']],
        df.rename(columns={'Drug2_ID': 'id', 'Drug2': 'drug'})[['id', 'drug']]
    ],
        axis=0, ignore_index=True
    ).drop_duplicates(subset=['id'])

    dct = {}
    if feature_name not in ['onehot', 'chembert_384']:
        fp_func = _get_fpgen(feature_name)
    elif feature_name == 'onehot':
        fp_func = OneHotEncoder(handle_unknown='ignore')
        fp_func.fit((mols['id'].values).reshape(-1,1))
    for _, x in tqdm(mols.iterrows(), total=len(mols)):
        if feature_name not in ['onehot', 'chembert_384']:
            dct[x['id']] = torch.tensor(np.array(fp_func(Chem.MolFromSmiles(x['drug']))), dtype = torch.float32)
        elif feature_name == 'chembert_384':
            dct[x['id']] = get_drug_chembert_384(x['drug'])
        else:
            dct[x['id']] = torch.tensor(fp_func.transform([[x['id']]]).toarray().flatten(), dtype = torch.float32)

    dct['shape'] = dct[x['id']].shape[0]
    print(dct[x['id']].shape[0])
        
    return dct

def get_drug_features_dic(study_name: str, feature_name: str):
    data_folder_path = "../data/" + str(study_name) + '/'
    synergy_score = 'loewe'
    dataset = pd.read_feather(data_folder_path + f"{synergy_score}.feather")
    mol_mapping = get_feature_dict(dataset, feature_name)

    return mol_mapping 

def _get_fpgen(name) -> Callable:
    """Gather the desired RDkit fingerprint generator.

    Returns:
        Callable: Fingerprint generator
    """
    vision = DICT_DEFAULT_VISION[name]
    length = 1024
    if name == "maccs":
        length, vision = 167, None
        return DICT_FP_NAMES[name]
    if name == "morgan":
        return DICT_FP_NAMES[name](
            radius=vision, fpSize=length
        ).GetFingerprint
    elif name == "rdkit":
        return DICT_FP_NAMES[name](
            maxPath=vision, fpSize=length
        ).GetFingerprint
    elif name == "map4":
        return DICT_FP_NAMES[name](
            radius=vision, dimensions=length,
            is_folded=True
        ).calculate
    
def get_cell_feature_dict(df: pd.DataFrame) -> dict:

    '''Returns the mapping between IDs and molecular graphs'''

    cells = df['Cell_Line_ID'].unique()
    func = OneHotEncoder(handle_unknown='ignore')
    cell_line_feats = pd.DataFrame(
        func.fit_transform((cells).reshape(-1,1)).toarray(),
        columns=[f'feat_{i}' for i in range(len(cells))],
        # index=rma_landm.index
    )
    cell_line_feats["cell_line_name"] = cells
        
    return cell_line_feats
if __name__ == '__main__':
    # -------- get drug fingerprints ------------------------
    for study_name in ['oneil']:
      for feature_name in list(DICT_FP_NAMES.keys()) + ['onehot', 'chembert_384']:
         print(feature_name)
         mol_mapping =get_drug_features_dic(study_name=study_name, feature_name=feature_name)
         data_folder_path = "../data/" + str(study_name) + '/'
         joblib.dump(mol_mapping, data_folder_path + 'mol_' + str(feature_name) +'_dic.joblib')



    # -------- get cell one hot representation ------------------------

    for study_name in ['oneil']:
        data_folder_path = "../data/" + str(study_name) + '/'
        synergy_score = 'loewe'
        dataset = pd.read_feather(data_folder_path + f"{synergy_score}.feather")
        cell_line_feats = get_cell_feature_dict(dataset)
        cell_line_feats.to_feather("./data/" + str(study_name) + "/" + "cell_lines_onehot.feather")

