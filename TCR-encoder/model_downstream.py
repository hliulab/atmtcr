import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torch.nn.utils.rnn import pad_sequence
import globalvar as gl
device = gl.get_value('cuda')

"""
下游任务 gnn模型
"""
class Model(nn.Module):
    def __init__(self, n_output=1, cell_dim=954, output_dim=128, dropout=0.2, encoder=None):
        super(Model, self).__init__()

        self.n_output = n_output
        self.gnn = encoder

        # cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(cell_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )

        # predict
        self.pre = nn.Sequential(
            nn.Linear(output_dim * 3, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.n_output)
        )

    def forward(self, data1, data2,drug_dict):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell

        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # encoder drug1
        x1 = self.gnn(x1, edge_index1, batch1)

        # encoder drug2
        x2 = self.gnn(x2, edge_index2, batch2)


        # encoder drug_org
        cell_vector = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell_vector)

        # predict drug_org layers
        xc = torch.cat((x1, x2, cell_vector), 1)
        xc = F.normalize(xc, 2, 1)
        out = self.pre(xc)

        return out

"""
下游任务 tranformer模型
"""
class Model_tranformer(nn.Module):
    def __init__(self, n_output=1, cell_dim=954, output_dim=128, dropout=0.2, encoder1=None,encoder2=None):
        super(Model_tranformer, self).__init__()

        self.n_output = n_output
        self.dropout = dropout
        self.output_dim = output_dim
        self.tranformer1 = encoder1
        # self.tranformer2 = encoder2

        # cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(cell_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )

        # predict
        self.pre = nn.Sequential(
            nn.Linear(output_dim * 3, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.n_output)
        )

    def forward(self, data1, data2,vocab_dict):

        #x1 药物1 x2 药物2
        x1, edge_index1, batch1, cell, smile1, y = data1.x, data1.edge_index, data1.batch, data1.cell, data1.smiles, data1.y

        x2, edge_index2, batch2, smile2 = data2.x, data2.edge_index, data2.batch, data2.smiles

        src_seq1, tgt_seq1 =get_smiles(smile1,vocab_dict)
        src_seq2, tgt_seq2 =get_smiles(smile2,vocab_dict)

        x1 = self.tranformer1(src_seq1, tgt_seq1)
        x2 = self.tranformer1(src_seq2, src_seq2)

        num_features_xt1 = x1.size(1)*x1.size(2)
        x1 = torch.reshape(x1, (x1.size(0), x1.size(1) * x1.size(2))).to(device)

        num_features_xt2 = x2.size(1)*x2.size(2)
        x2 = torch.reshape(x2, (x2.size(0), x2.size(1) * x2.size(2))).to(device)

        reduction1 = nn.Sequential(
            nn.Linear(num_features_xt1, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.output_dim)
        ).to(device)

        reduction2 = nn.Sequential(
            nn.Linear(num_features_xt2, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.output_dim)
        ).to(device)

        x1 = reduction1(x1)
        x2 = reduction2(x2)

        # encoder drug_org
        cell_vector = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell_vector)


        # predict drug_org layers
        xc = torch.cat((x1, x2, cell_vector), 1)
        xc = F.normalize(xc, 2, 1)
        out = self.pre(xc)

        return out, y

def get_smiles(smiles,vocab_dict):
    smile_seqs = []
    src_lens = []
    for smile in smiles:
        smile_seq = [int(vocab_dict.get(i)) for i in smile]
        smile_seqs.append(torch.LongTensor(smile_seq))
        src_len = len(smile)
        src_lens.append(src_len)

    src_seq = pad_sequence(smile_seqs, batch_first=True).to(device)
    tgt_seq = src_seq

    return src_seq,tgt_seq