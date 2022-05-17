# -*- coding: UTF-8 -*-
import argparse
import math
import random
import time
from MyDataSet_smiles import MyDataSet
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader as DL
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from sklearn.manifold import TSNE
import os
os.environ['CUDA_VISIBLE_DEVICES']= '0,1'
import globalvar as gl
# os.environ['CUDA_VISIBLE_DEVICES']= '1'
gl._init()
if torch.cuda.is_available():
    device = torch.device('cuda')
    gl.set_value('cuda', device)
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    gl.set_value('cuda', device)
    print('The code uses CPU!!!')

# from transformer import Transformer
from transformer_smiles import Transformer
from transformer_smiles import Encoder
from utils_pretrain import *
from model_pretrain import Net
from nt_xent import NT_Xent
from encoder_gnn import GCNet
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
plt.switch_backend('agg')
"""
模型预训练

"""

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, vocab_dict):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    feature_graph = torch.Tensor()
    feature_org = torch.Tensor()
    top_3_tokens = []
    for tem in train_bar:
        graph1, out_1, org2, out_2,attn_score = net(tem.to(device))
        top_3_tokens.extend(calculate(tem,attn_score))
        feature_graph = torch.cat((feature_graph, torch.Tensor(graph1.to('cpu').data.numpy())), 0)
        feature_org = torch.cat((feature_org, torch.Tensor(org2.to('cpu').data.numpy())), 0)
        criterion = NT_Xent(out_1.shape[0], temperature, 1)
        loss = criterion(out_1, out_2)
        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        # break
    return total_loss / total_num, feature_graph, feature_org, attn_score,top_3_tokens

def calculate(inputs,attn_score):
    """
    统计计算attention分数最高的前三
    1. 计算attention score 最大的前三个以及索引
    2. 通过索引去找对应输入ids的位置的字符标志
    3. 通过ids去映射到字
    """
    # print(inputs.shape,len(attn_score),attn_score[0].shape)
    attn_score = torch.tensor(attn_score[0])
    scores, indexs = torch.sort(attn_score,descending=True)
    top_3_token = []
    for i in range(indexs.shape[0]):
        ids = inputs[i,indexs[i,:3]]
        # print(ids)
        top_3_token.append(ids.cpu().numpy().tolist())
    return top_3_token
def get_dict(datafile):

    # smiles 字典 统计所有smiles 字符出现频率 有高到低字典排序 1- 43
    src_dict = {}
    with open("data/pretrain/data/" + datafile + "_dict.txt", 'r') as f:
        for line in f.readlines():
            line = line.strip()
            k = line.split(' ')[0]
            v = line.split(' ')[1]
            src_dict[k] = int(v)
    f.close()
    sort_dict = {key: rank for rank, key in enumerate(sorted(src_dict.values(), reverse=True), 1)}
    vocab_dict = {k: sort_dict[v] for k, v in src_dict.items()}

    vocab_dict['<pad>'] = 0
    return vocab_dict

def compute_rsquared(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    r2=round((SSR / SST) ** 2,3)
    return r2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ATMTCR')
    parser.add_argument('--datafile', default='data', help='orginal data for input in-vitro tryout now')
    parser.add_argument('--path', default='pretrain', help='orginal data for input')
    parser.add_argument('--feature_dim', default=32, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=1024, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--downtask', default='model_downstream.py', help='Number of sweeps over the dataset to train')
    # d_ff, d_k, d_v, n_heads = 64, 64, 64, 2

    #目前最佳batch_size1024
    d_ff, d_k, d_v, n_heads = 128, 128, 128, 2
    n_layers = 2
    precet = 0.25
    dropout = 0.2


    # args parse
    args = parser.parse_args()
    print(args)
    feature_dim, temperature, k, datafile = args.feature_dim, args.temperature, args.k, args.datafile

    batch_size, epochs = args.batch_size, args.epochs

    train_datas = []
    # data prepare
    train_data = TestbedDataset(root=args.path, dataset=args.datafile)


    vocab_dict = get_dict(datafile)
    vl = len(vocab_dict)
    PAD_IDX = vocab_dict.get('<pad>')
    # encoder drug_smiles
    smile_seqs = []
    # 将一个batch的smiles 转化为 torch向量
    for smile in train_data:
        smile_seq = [int(vocab_dict.get(i)) for i in smile]

        tgt = torch.LongTensor(smile_seq)
        # tgt = torch.cat([torch.tensor([self.BOS_IDX]), torch.LongTensor(smile_seq), torch.tensor([self.EOS_IDX])], dim=0)

        # smile_seq[random.randint(0, len(smile) - 1)] = 0
        smile_seqs.append(torch.LongTensor(smile_seq))

    # 统一序列长度
    src_seq = pad_sequence(smile_seqs, batch_first=True, padding_value=PAD_IDX)
    src_seq_len = src_seq.shape[1] # 序列长度
    train_data = MyDataSet(src_seq)


    # model setup and optimizer config


    # encoder
    # model_encoder1 = GCNet().cuda()
    coder = Encoder(src_vocab_size=vl, d_model=feature_dim, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads, n_layers=n_layers).to(device)
    model = Transformer(src_vocab_size=vl, tgt_vocab_size=None, d_model=feature_dim, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads, n_layers=n_layers, precet=precet, seq_len=src_seq_len, dropout=dropout, trans_encoder=coder).cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1])  # 在这里使用了2个GPU

    model.to(device)
    from collections import Counter
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)
    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}'.format(batch_size, epochs,datafile)
    if not os.path.exists('results/'+save_name_pre):
        os.mkdir('results/'+save_name_pre)
    tsne = TSNE()
    AUCs = ('Epoch\tloss\tr2\ttime')
    for epoch in range(1, epochs + 1):
        start = time.time()
        # train_loader = DL(train_data, batch_size=batch_size, shuffle=True)
        train_loader = DL(train_data, batch_size=batch_size, shuffle=False)
        train_loss, features, org, attn_score,top_3_tokens = train(model, train_loader, optimizer, vocab_dict)
        # 取频率最大
        top_3_tokens = [j for i in top_3_tokens for j in i]
        counts = {k:top_3_tokens.count(v) for k,v in vocab_dict.items()}
        counts = sorted(counts.items(),key=lambda x:x[1],reverse=True)
        with open(f"counts/count_{epoch}_feq.json",'w',encoding='utf8') as f:
            import json
            json.dump(counts,f,ensure_ascii=False,indent=2)
            # print(counts)
        # exit(0)
    torch.save(model.state_dict(), 'results/model_transformer_state_dict.pkl')
    torch.save(model, 'results/model_transformer.pkl')

















