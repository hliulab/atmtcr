import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import globalvar as gl
device = gl.get_value('cuda')
from transformer import Transformer

"""
预训练模型
transformer
"""
class TransformerCon(torch.nn.Module):
    def __init__(self, n_output=128, output_dim=128, dropout=0.2, encoder1=None, encoder2=None):

        super(TransformerCon, self).__init__()
        self.n_output = n_output
        self.dropout = dropout
        self.output_dim = output_dim

        self.tranformer1 = encoder1
        self.tranformer2 = encoder2

        # predict head
        self.pre_head = nn.Sequential(
            nn.Linear(output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.n_output)
        )
        # self.reduction = nn.Linear(self.num_features_xt, self.output_dim).to(device)
        # org_drug featrues


    def forward(self, data1, vocab_dict):
        x, edge_index1, batch1, drug_org, smiles = data1.x, data1.edge_index, data1.batch, data1.drug_org, data1.smiles


        # out1 = None

        # self.BOS_IDX = vocab_dict.get('<bos>')
        # self.EOS_IDX = vocab_dict.get('<eos>')
        # self.MASK = vocab_dict.get('<mask>')
        self.PAD_IDX = vocab_dict.get('<pad>')
        # encoder drug_smiles
        smile_seqs = []
        tgt_seqs = []
        # 随机mask idx
        len_smiles = len(smiles)
        # 将一个batch的smiles 转化为 torch向量
        mask_idx = get_mask_idx(len_smiles, 0.2)

        for k, smile in enumerate(smiles):
            smile_seq = [int(vocab_dict.get(i)) for i in smile]
            tgt = torch.LongTensor(smile_seq)
            tgt_seqs.append(tgt)
            if k in mask_idx:
                smile_seq[random.randint(0, len(smile) - 1)] = self.PAD_IDX
            smile_seqs.append(torch.LongTensor(smile_seq))

        #统一序列长度
        src_seq = pad_sequence(smile_seqs, batch_first=True, padding_value=self.PAD_IDX).transpose(0, 1).to(device)
        tgt_seq = pad_sequence(tgt_seqs, batch_first=True, padding_value=self.PAD_IDX).transpose(0, 1).to(device)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(self, src_seq, tgt_seq,device)

        x1 = self.tranformer1(src_seq, src_seq, src_mask, tgt_mask, src_key_padding_mask=src_padding_mask,
                             tgt_key_padding_mask=tgt_padding_mask).transpose(0, 1)
        x2 = self.tranformer2(tgt_seq, tgt_seq, src_mask, tgt_mask, src_key_padding_mask= src_padding_mask, tgt_key_padding_mask=tgt_padding_mask).transpose(0, 1)

        self.num_features_xt = x1.size(1) * x1.size(2)

        x1 = torch.reshape(x1, (x1.size(0), -1)).to(device)

        # self.reduction1 = nn.Linear(self.num_features_xt, self.output_dim).to(device)
        x1 = nn.Linear(self.num_features_xt, self.output_dim).to(device)(x1)

        # predict drug_org layers1
        out1 = self.pre_head(x1)

        #训练结果 匹配与gnn大小一致
        x2 = torch.reshape(x2, (x2.size(0), -1)).to(device)

        # self.reduction2 = nn.Linear(self.num_features_xt, self.output_dim).to(device)
        x2 = nn.Linear(self.num_features_xt, self.output_dim).to(device)(x2)

        # predict drug_org layers
        out2 = self.pre_head(x2)

        return x1, out1, x2, out2

def get_mask_idx(len_s, p):
    # smiles_tmp = smiles
    all_idx = list(range(len_s))
    num_mask = int(len_s * p)
    np.random.seed(num_mask)
    np.random.shuffle(all_idx)
    mask_idx = all_idx[:num_mask]

    return mask_idx

def create_mask(self, src, tgt, device):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]
        tgt_mask = Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)  # [tgt_len,tgt_len]
        # Decoder的注意力Mask输入，用于掩盖当前position之后的position，所以这里是一个对称矩阵
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool).to(device)
        # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的，所以这里全是0
        src_padding_mask = (src == self.PAD_IDX).transpose(0, 1).to(device)
        # 用于mask掉Encoder的Token序列中的padding部分,[batch_size, src_len]
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1).to(device)
        # 用于mask掉Decoder的Token序列中的padding部分,batch_size, tgt_len
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask