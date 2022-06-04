# all imports here
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from torch.autograd import Variable

from datetime import datetime
from tqdm import tqdm
import sklearn
from copy import deepcopy

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, bidirectional = False):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = SEQ_LENGTH
        self.bidrectional = bidirectional
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional = bidirectional)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).cuda()
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).cuda()
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        #h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(ula)
        
        return out

import torch.nn as nn
import math

device = 'cuda'

class MultiHeadAttention(nn.Module):
    '''Multi-head self-attention module'''
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H # number of heads
        self.D = D # dimension
        
        self.wq = nn.Linear(D, D*H)
        self.wk = nn.Linear(D, D*H)
        self.wv = nn.Linear(D, D*H)

        self.dense = nn.Linear(D*H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H*D))   # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):

        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q)  # (B, H, S, D)
        k = self.split_heads(k)  # (B, H, S, D)
        v = self.split_heads(v)  # (B, H, S, D)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) #(B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)

        # add the mask to the scaled tensor.
        if mask is not None:
            attention_scores += (mask * -1e9)
        
        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        scaled_attention = torch.matmul(attention_weights, v)  # (B, H, S, D)
        concat_attention = self.concat_heads(scaled_attention) # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    '''Multi-head self-attention module'''
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H # number of heads
        self.D = D # dimension
        
        self.wq = nn.Linear(D, D*H)
        self.wk = nn.Linear(D, D*H)
        self.wv = nn.Linear(D, D*H)

        self.dense = nn.Linear(D*H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H*D))   # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):

        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q)  # (B, H, S, D)
        k = self.split_heads(k)  # (B, H, S, D)
        v = self.split_heads(v)  # (B, H, S, D)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) #(B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)

        # add the mask to the scaled tensor.
        if mask is not None:
            attention_scores += (mask * -1e9)
        
        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        scaled_attention = torch.matmul(attention_weights, v)  # (B, H, S, D)
        concat_attention = self.concat_heads(scaled_attention) # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, attention_weights

class MultiHeadAttentionCosformerNew(nn.Module):
    '''Multi-head self-attention module'''
    def __init__(self, D, H):
        super(MultiHeadAttentionCosformerNew, self).__init__()
        self.H = H # number of heads
        self.D = D # dimension
        
        self.wq = nn.Linear(D, D*H)
        self.wk = nn.Linear(D, D*H)
        self.wv = nn.Linear(D, D*H)

        self.dense = nn.Linear(D*H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H*D))   # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):

        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q).permute(0,2,1,3)  # (B, S, H, D)
        k = self.split_heads(k).permute(0,2,1,3)  # (B, S, H, D)
        v = self.split_heads(v).permute(0,2,1,3)  # (B, S, H, D)
        B = q.shape[0]
        S = q.shape[1]

        q = torch.nn.functional.elu(q) + 1 # Sigmoid torch.nn.ReLU()
        k = torch.nn.functional.elu(k) + 1 # Sigmoid torch.nn.ReLU()

        # q, k, v -> [batch_size, seq_len, n_heads, d_head]
        cos = (torch.cos(1.57*torch.arange(S)/S).unsqueeze(0)).repeat(B,1).cuda()
        sin = (torch.sin(1.57*torch.arange(S)/S).unsqueeze(0)).repeat(B,1).cuda()
        # cos, sin -> [batch_size, seq_len]
        q_cos = torch.einsum('bsnd,bs->bsnd', q, cos)
        q_sin = torch.einsum('bsnd,bs->bsnd', q, sin)
        k_cos = torch.einsum('bsnd,bs->bsnd', k, cos)
        k_sin = torch.einsum('bsnd,bs->bsnd', k, sin)
        # q_cos, q_sin, k_cos, k_sin -> [batch_size, seq_len, n_heads, d_head]

        kv_cos = torch.einsum('bsnx,bsnz->bnxz', k_cos, v)
        # kv_cos -> [batch_size, n_heads, d_head, d_head]
        qkv_cos = torch.einsum('bsnx,bnxz->bsnz', q_cos, kv_cos)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        kv_sin = torch.einsum('bsnx,bsnz->bnxz', k_sin, v)
        # kv_sin -> [batch_size, n_heads, d_head, d_head]
        qkv_sin = torch.einsum('bsnx,bnxz->bsnz', q_sin, kv_sin)
        # qkv_sin -> [batch_size, seq_len, n_heads, d_head]

        # denominator
        denominator = 1.0 / (torch.einsum('bsnd,bnd->bsn', q_cos, k_cos.sum(axis=1))
                             + torch.einsum('bsnd,bnd->bsn',
                                            q_sin, k_sin.sum(axis=1))
                             + 1e-5)
        # denominator -> [batch_size, seq_len, n_heads]

        O = torch.einsum('bsnz,bsn->bsnz', qkv_cos +
                              qkv_sin, denominator).contiguous()
        # output -> [batch_size, seq_len, n_heads, d_head]

        concat_attention = self.concat_heads(O.permute(0,2,1,3)) # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, None

class MultiHeadAttentionCosSquareformerNew(nn.Module):
    '''Multi-head self-attention module'''
    def __init__(self, D, H):
        super(MultiHeadAttentionCosSquareformerNew, self).__init__()
        self.H = H # number of heads
        self.D = D # dimension
        
        self.wq = nn.Linear(D, D*H)
        self.wk = nn.Linear(D, D*H)
        self.wv = nn.Linear(D, D*H)

        self.dense = nn.Linear(D*H, D)

    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H*D))   # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x, mask):

        q = self.wq(x)  # (B, S, D*H)
        k = self.wk(x)  # (B, S, D*H)
        v = self.wv(x)  # (B, S, D*H)

        q = self.split_heads(q).permute(0,2,1,3)  # (B, S, H, D)
        k = self.split_heads(k).permute(0,2,1,3)  # (B, S, H, D)
        v = self.split_heads(v).permute(0,2,1,3)  # (B, S, H, D)
        B = q.shape[0]
        S = q.shape[1]

        q = torch.nn.functional.elu(q) + 1 # Sigmoid torch.nn.ReLU()
        k = torch.nn.functional.elu(k) + 1 # Sigmoid torch.nn.ReLU()

        # q, k, v -> [batch_size, seq_len, n_heads, d_head]
        cos = (torch.cos(3.1415*torch.arange(S)/S).unsqueeze(0)).repeat(B,1).cuda()
        sin = (torch.sin(3.1415*torch.arange(S)/S).unsqueeze(0)).repeat(B,1).cuda()
        # cos, sin -> [batch_size, seq_len]
        q_cos = torch.einsum('bsnd,bs->bsnd', q, cos)
        q_sin = torch.einsum('bsnd,bs->bsnd', q, sin)
        k_cos = torch.einsum('bsnd,bs->bsnd', k, cos)
        k_sin = torch.einsum('bsnd,bs->bsnd', k, sin)
        # q_cos, q_sin, k_cos, k_sin -> [batch_size, seq_len, n_heads, d_head]

        kv_cos = torch.einsum('bsnx,bsnz->bnxz', k_cos, v)
        # kv_cos -> [batch_size, n_heads, d_head, d_head]
        qkv_cos = torch.einsum('bsnx,bnxz->bsnz', q_cos, kv_cos)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        kv_sin = torch.einsum('bsnx,bsnz->bnxz', k_sin, v)
        # kv_sin -> [batch_size, n_heads, d_head, d_head]
        qkv_sin = torch.einsum('bsnx,bnxz->bsnz', q_sin, kv_sin)
        # qkv_sin -> [batch_size, seq_len, n_heads, d_head]

        kv = torch.einsum('bsnx,bsnz->bnxz', k, v)
        # kv -> [batch_size, n_heads, d_head, d_head]
        qkv = torch.einsum('bsnx,bnxz->bsnz', q, kv)
        # qkv_cos -> [batch_size, seq_len, n_heads, d_head]

        # denominator
        denominator = 1.0 / (torch.einsum('bsnd,bnd->bsn', q, k.sum(axis=1)) + torch.einsum('bsnd,bnd->bsn', q_cos, k_cos.sum(axis=1))
                             + torch.einsum('bsnd,bnd->bsn',
                                            q_sin, k_sin.sum(axis=1))
                             + 1e-5)
        # denominator -> [batch_size, seq_len, n_heads]

        O = torch.einsum('bsnz,bsn->bsnz', qkv + qkv_cos +
                              qkv_sin, denominator).contiguous()
        # output -> [batch_size, seq_len, n_heads, d_head]

        concat_attention = self.concat_heads(O.permute(0,2,1,3)) # (B, S, D*H)
        output = self.dense(concat_attention)  # (B, S, D)

        return output, None

# Positional encodings
def get_angles(pos, i, D):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(D))
    return pos * angle_rates


def positional_encoding(D, position=20, dim=3, device=device):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(D)[np.newaxis, :],
                            D)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    if dim == 3:
        pos_encoding = angle_rads[np.newaxis, ...]
    elif dim == 4:
        pos_encoding = angle_rads[np.newaxis,np.newaxis,  ...]
    return torch.tensor(pos_encoding, device=device)

class TransformerLayer(nn.Module):
    def __init__(self, D, H, hidden_mlp_dim, dropout_rate, attention_type='cosine_square'):
        super(TransformerLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.mlp_hidden = nn.Linear(D, hidden_mlp_dim)
        self.mlp_out = nn.Linear(hidden_mlp_dim, D)
        self.layernorm1 = nn.LayerNorm(D, eps=1e-9)
        self.layernorm2 = nn.LayerNorm(D, eps=1e-9)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        if attention_type == 'cosine':
          self.mha = MultiHeadAttentionCosformerNew(D, H)
        elif attention_type == 'cosine_square':
          self.mha = MultiHeadAttentionCosSquareformerNew(D, H)
        else:
          self.mha = MultiHeadAttention(D,H)

    def forward(self, x, look_ahead_mask):
        
        attn, attn_weights = self.mha(x, look_ahead_mask)  # (B, S, D)
        attn = self.dropout1(attn) # (B,S,D)
        attn = self.layernorm1(attn + x) # (B,S,D)

        mlp_act = torch.relu(self.mlp_hidden(attn))
        mlp_act = self.mlp_out(mlp_act)
        mlp_act = self.dropout2(mlp_act)
        
        output = self.layernorm2(mlp_act + attn)  # (B, S, D)

        return output, attn_weights
  
class Transformer(nn.Module):
    '''Transformer Decoder Implementating several Decoder Layers.
    '''
    def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate, attention_type='cosine_square', SL=20):
        super(Transformer, self).__init__()
        self.attention_type = attention_type
        self.sqrt_D = torch.tensor(math.sqrt(D))
        self.num_layers = num_layers
        self.input_projection = nn.Linear(inp_features, D) # multivariate input
        self.output_projection = nn.Linear(D, out_features) # multivariate output
        self.pos_encoding = positional_encoding(D, position=SL)
        self.dec_layers = nn.ModuleList([TransformerLayer(D, H, hidden_mlp_dim, 
                                        dropout_rate=dropout_rate, attention_type=self.attention_type
                                       ) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        B, S, D = x.shape
        # attention_weights = {}
        x = self.input_projection(x)
        x *= self.sqrt_D
        
        x += self.pos_encoding[:, :S, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, _ = self.dec_layers[i](x=x,
                                          look_ahead_mask=mask)
            # attention_weights['decoder_layer{}'.format(i + 1)] = block
        
        x = self.output_projection(x)
        
        return x, None # attention_weights # (B,S,S)

class TransLSTM(nn.Module):
    '''Transformer Decoder Implementating several Decoder Layers.
    '''
    def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate, LSTM_module, attention_type='regular'):
        super(TransLSTM, self).__init__()
        self.attention_type = attention_type
        self.sqrt_D = torch.tensor(math.sqrt(D))
        self.num_layers = num_layers
        self.input_projection = nn.Linear(inp_features, D) # multivariate input
        self.output_projection = nn.Linear(D, 4) # multivariate output
        self.fc = nn.Linear(4*2, out_features)
        self.pos_encoding = positional_encoding(D)
        self.dec_layers = nn.ModuleList([TransformerLayer(D, H, hidden_mlp_dim, 
                                        dropout_rate=dropout_rate, attention_type=self.attention_type
                                       ) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.LSTM = LSTM_module

    def forward(self, x, mask):
        x_l = self.LSTM(x)
        B, S, D = x.shape
        attention_weights = {}
        x = self.input_projection(x)
        x *= self.sqrt_D
        
        x += self.pos_encoding[:, :S, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block = self.dec_layers[i](x=x,
                                          look_ahead_mask=mask)
            attention_weights['decoder_layer{}'.format(i + 1)] = block
        
        x = self.output_projection(x)

        x = torch.cat((x,x_l),axis=2)

        x = self.fc(x)
        
        return x, attention_weights # (B,S,S)