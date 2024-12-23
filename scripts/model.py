import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
# The model can refer to 
# Paper : RECOVER identifies synergistic drug combinations in vitro through sequential model optimization
# https://doi.org/10.1016/j.crmeth.2023.100599 
########################################################################################################################
# Modules
########################################################################################################################


class LinearModule(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearModule, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class ReLUModule(nn.ReLU):
    def __init__(self):
        super(ReLUModule, self).__init__()

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]

class SigmoidModule(nn.Sigmoid):
    def __init__(self):
        super(SigmoidModule, self).__init__()

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]

class DropoutModule(nn.Dropout):
    def __init__(self, p):
        super(DropoutModule, self).__init__(p)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class FilmModule(torch.nn.Module):
    def __init__(self, num_cell_lines, out_dim):
        super(FilmModule, self).__init__()
        film_init = 1 / 100 * torch.randn(num_cell_lines, 2 * out_dim)
        film_init = film_init + torch.Tensor([([1] * out_dim) + ([0] * out_dim)])

        self.film = Parameter(film_init)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [
            self.film[cell_line][:, : x.shape[1]] * x
            + self.film[cell_line][:, x.shape[1]:],
            cell_line]


class FilmWithFeatureModule(torch.nn.Module):
    def __init__(self, num_cell_line_features, out_dim):
        super(FilmWithFeatureModule, self).__init__()

        self.out_dim = out_dim

        self.condit_lin_1 = nn.Linear(num_cell_line_features, num_cell_line_features)
        self.condit_relu = nn.ReLU()
        self.condit_lin_2 = nn.Linear(num_cell_line_features, 2 * out_dim)

        # Change initialization of the bias so that the expectation of the output is 1 for the first columns
        self.condit_lin_2.bias.data[: out_dim] += 1

    def forward(self, input):
        x, cell_line_features = input[0], input[1]

        # Compute conditioning
        condit = self.condit_lin_2(self.condit_relu(self.condit_lin_1(cell_line_features)))

        return [
            condit[:, :self.out_dim] * x
            + condit[:, self.out_dim:],
            cell_line_features
        ]


class LinearFilmWithFeatureModule(torch.nn.Module):
    def __init__(self, num_cell_line_features, out_dim):
        super(LinearFilmWithFeatureModule, self).__init__()

        self.out_dim = out_dim

        self.condit_lin_1 = nn.Linear(num_cell_line_features, 2 * out_dim)

        # Change initialization of the bias so that the expectation of the output is 1 for the first columns
        self.condit_lin_1.bias.data[: out_dim] += 1

    def forward(self, input):
        x, cell_line_features = input[0], input[1]

        # Compute conditioning
        condit = self.condit_lin_1(cell_line_features)

        return [
            condit[:, :self.out_dim] * x
            + condit[:, self.out_dim:],
            cell_line_features
        ]


########################################################################################################################
# Bilinear MLP
########################################################################################################################


class MLP_drug_cell_permutate(torch.nn.Module):
    def __init__(self, config):

        super(MLP_drug_cell_permutate, self).__init__()

        self.device = config['device']
        predictor_layers = config['predictor_layers']
        self.layer_dims = config['predictor_layers']
        self.task =  config['task']
        self.operation = config['operation']

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]

        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        layers_before_merge = []
        layers_after_merge = []

        # Build early layers (before addition of the two embeddings)
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        # Build last layers (after addition of the two embeddings)
        for i in range(
            len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
            len(self.layer_dims) - 1,
        ):

            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        # Initialize weights close to identity
        if self.operation == 'bilinear':
            self.bilinear_weights = Parameter(
                1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
                + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
            )
            self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))

            self.allow_neg_eigval = config["allow_neg_eigval"]
            if self.allow_neg_eigval:
                self.bilinear_diag = Parameter(1 / 100 * torch.randn((self.merge_dim, self.merge_dim)) + 1)
    
    def permute_operation(self, emb_1, emb_2):
        # additive , max, multiplication, sort and concate, bilinear
        if self.operation == 'additive':
            lat = emb_1+emb_2
        elif self.operation == 'max':
            lat = torch.max(emb_1, emb_2)
        elif self.operation == 'multiply':
            lat = emb_1*emb_2
        elif self.operation == 'sort':
            cat_lat = torch.cat([emb_1, emb_2], dim = 1)

            lat, _ = torch.sort(cat_lat) 
            
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
    def forward(self, h_drug_1, h_drug_2, cell_lines ):
    
        # Apply before merge MLP
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

       
        h_1_scal_h_2 = self.permute_operation(h_1, h_2)
        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        return comb

    def add_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if self.task == 'clf':
            if i != len(self.layer_dims) - 2:
                layers.append(ReLUModule())
            else:
                layers.append(SigmoidModule())
        else:
            if i != len(self.layer_dims) - 2:
                layers.append(ReLUModule())

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]


