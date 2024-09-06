
import torch 
from torch import nn, cat as tcat, tensor, optim, LongTensor, flatten
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

class Block_nn(torch.nn.Module):
    """
    Represents a neural network block consisting of a sequence of linear layers
    and ReLU activations. The number of layers is configurable.

    Attributes:
        fit (nn.ModuleList): A list of layers in the block.

    Args:
        dim_in (int): The input dimension.
        dim_out (int): The output dimension.
        hid_dim (int): The hidden dimension.
    Methods:
        forward(x): Defines the forward pass of the block.
    """
    def __init__(self, dim_in, hid_dim, dim_out):
        super(Block_nn, self).__init__()

        self.fit = nn.Sequential(
            nn.Linear(dim_in, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim//2),
            nn.BatchNorm1d(hid_dim//2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, dim_out),
        )
    
    def forward(self, x):
        return self.fit(x)
class MLP_drug_cell_permutate(nn.Module):
    """
    MLP model for prediction of the probability of being synergy or synergy score.

    Attributes:
        fit_pred  (nn.ModuleList): A list of layers in the block. This block takes the embedding of drugs.
        and cells, then predict the synergy probability or synergy score. 
        drug_emb (nn.ModuleList): A list of layers in the block. This block is to reduce the dimension of drug features.
        cell_emb (nn.ModuleList): A list of layers in the block. This block is to reduce the dimension of cell features.

    Args:
        hparam (dic): A dictionary for hyperparameters.
        - drug_dim (int): The dimension of drug features.
        - cell_dim (int): The dimension of cell features.
        - hid_dim (int): The dimension for drug&cell embedding. 
        - hid_dim_1 (int): The hidden dimension in dimension reducer block. 
        - cell_feature (str): The type of cell feature, i.e., ['ge', 'onehot']
        - drug_feature (str): The type of drug feature, i.e., ['morgan', 'onehot', 'chembert_384', 'map4', 'maccs' ]
        - operation (str) : The type of invariant permutation function, i.e., ['bilinear', 'additive' , 'max', 'other'] 
        regress (bool): A variable to define the task: if regress = False, output will be  
        probability of synergy, otherwise, it is the synergy score predcition. 

    Methods:
        permute_operation(emb_1, emb_2): Define the permutation function for drug embeddings. 
        forward(drug_a, drug_b, cell): Defines the forward pass of model. 
    """
    def __init__(self, hparam, regress = False):
        super(MLP_drug_cell_permutate, self).__init__()
        drug_dim = hparam['drug_dim']
        cell_dim = hparam['cell_dim']
        hid_dim = hparam['hid_dim']
        hid_dim_1 = hparam['hid_dim_1']
        self.operation = hparam['operation']

        try:
            self.drugs_feature = hparam['drugs_feature']
        except:
            self.drugs_feature = 'chembert'
        try:
            self.cell_feature = hparam['cell_feature']
        except:
            self.cell_feature = 'ge'
    
        if self.drugs_feature == 'onehot':
            self.drug_emb = nn.Embedding(drug_dim, hid_dim)
        else:
            self.drug_emb = Block_nn(drug_dim, hid_dim_1, hid_dim)

        if self.cell_feature == 'ge':
            self.cell_emb = Block_nn(cell_dim, hid_dim_1, hid_dim)
        else:
            self.cell_emb = nn.Embedding(cell_dim, hid_dim)
        if self.operation in ['sort', 'other']:
            input_dim = 3*hid_dim
        else:
            input_dim = 2*hid_dim 
        self.fit_pred = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid_dim, hid_dim_1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid_dim_1, 1),

        )

        self.nn_out = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.regress = regress
        if self.operation == 'bilinear':
            self.bilinear_weights = nn.Parameter(
                1 / 100 * torch.randn((hid_dim, hid_dim, hid_dim))
                + torch.cat([torch.eye(hid_dim)[None, :, :]] * hid_dim, dim=0)
            )
            self.bilinear_offsets = nn.Parameter(1 / 100 * torch.randn((hid_dim)))

        
    def permute_operation(self, emb_1, emb_2):
        # additive , max, multiplication, sort and concate, bilinear
        if self.operation == 'additive':
            lat = emb_1+emb_2
        elif self.operation == 'max':
            lat = torch.max(emb_1, emb_2)
            
        elif self.operation == 'bilinear':
            # compute <W.h_1, W.h_2> = h_1.T . W.T.W . h_2
            h_1 = self.bilinear_weights.matmul(emb_1.mT).T
            h_2 = self.bilinear_weights.matmul(emb_2.mT).T

            # "Transpose" h_1
            h_1 = h_1.permute(0, 2, 1)

            # Multiplication
            lat = (h_1 * h_2).sum(1)

            # Add offset
            lat += self.bilinear_offsets
        else:
            lat = torch.cat([emb_1, emb_2], dim = 1)
        return lat 


    def forward(self, drug_a, drug_b, cell):
        if self.drugs_feature == 'onehot':
            emb_1, emb_2 = drug_a@self.drug_emb.weight, drug_b@self.drug_emb.weight
        else:
            emb_1, emb_2 = self.drug_emb(drug_a).squeeze(1), self.drug_emb(drug_b).squeeze(1)
        
        if self.cell_feature == 'ge':
            emb_3 = self.cell_emb(cell).squeeze(1)
        else:
            emb_3 = cell@self.cell_emb.weight    
            
        lat = self.permute_operation(emb_1, emb_2)
        emb = tcat((lat, emb_3), dim=1)
        pred = self.fit_pred(emb)
        if self.regress:
            return pred 
        else:
            label = self.nn_out(pred)
            return label 