#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import globalvar as gl
gl._init()
if torch.cuda.is_available():
    device = torch.device('cuda')
    gl.set_value('cuda', device)
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    gl.set_value('cuda', device)
    print('The code uses CPU!!!')
from utils_dowmstream import *
from encoder_gnn import GCNet
from model_downstream import Model, Model_tranformer
import random
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer import Transformer
import deepchem as dc

""""
药物下游任务
"""

def compute_mae_mse_rmse(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    mae=sum(absError)/len(absError)  # 平均绝对误差MAE
    mse=sum(squaredError)/len(squaredError)  # 均方误差MSE
    RMSE= mse ** 0.5
    return mae, mse, RMSE

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

# train for one epoch to learn unique features
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch,drug_dict):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).float().to(device)
        y = y.squeeze(1)
        output, y_ = model(data1, data2, drug_dict)
        # loss = loss_fn(output, y.unsqueeze(1))
        loss = loss_fn(output, F.normalize(y.unsqueeze(1), 2, 1))
        # print('loss', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test,drug_dict):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output, y_ = model(data1, data2,drug_dict)
            ys = output.to('cpu').data.numpy()
            # predicted_labels = list(map(lambda x: np.argmax(x), ys))
            total_preds = torch.cat((total_preds, torch.Tensor(ys)), 0)
            # total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            y = data1.y.view(-1, 1).cpu()
            y_norm = F.normalize(y, 2, 1)
            total_labels = torch.cat((total_labels, y_norm), 0)
    return total_preds.numpy().flatten(), total_labels.numpy().flatten()


def get_dict(drug_name):
    src_dict = {}
    # with open("down_task/processed/dict_" + drug_name + ".txt", 'r') as f:
    with open("pretrain/processed/dict.txt", 'r') as f:
        for line in f.readlines():
            line = line.strip()
            k = line.split(' ')[0]
            v = line.split(' ')[1]
            src_dict[k] = int(v)
    f.close()
    sort_dict = {key: rank for rank, key in enumerate(sorted(src_dict.values(),reverse=True), 1)}
    drug_vocab_dict = {k: sort_dict[v] for k,v in src_dict.items()}

    return drug_vocab_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--combfile', default='score_leave', help='orginal data for input')
    parser.add_argument('--cellfile', default='cell_features_954', help='orginal data for input')
    parser.add_argument('--path', default='down_task', help='orginal data for input')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=50, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--encoder', default=2, type=int, help='Number of encoder 1 2 ')

    # args parse
    args = parser.parse_args()
    print(args)
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k

    batch_size, epochs, encoder = args.batch_size, args.epochs, args.encoder

    LOG_INTERVAL = 50


    # data prepare
    drug1_data = TestbedDataset(root=args.path, dataset=args.combfile + '_drug1', drugdata=args.combfile, drug_name='smile1', celldata=args.cellfile)
    drug2_data = TestbedDataset(root=args.path, dataset=args.combfile + '_drug2', drugdata=args.combfile, drug_name='smile2', celldata=args.cellfile)


    # 10 k-fold 处理
    lenth = len(drug1_data)
    pot = int(lenth / 10)
    print('lenth', lenth)
    print('pot', pot)
    random_num = random.sample(range(0, lenth), lenth)
    for i in range(10):
        print("k-fold {}".format(i+1))
        test_num = random_num[pot * i:pot * (i + 1)]
        train_num = random_num[:pot * i] + random_num[pot * (i + 1):]

        drug1_data_train = drug1_data[train_num]
        drug1_data_test = drug1_data[test_num]
        drug1_loader_train = DataLoader(drug1_data_train, batch_size=batch_size, shuffle=None)
        drug1_loader_test = DataLoader(drug1_data_test, batch_size=batch_size, shuffle=None)

        drug2_data_test = drug2_data[test_num]
        drug2_data_train = drug2_data[train_num]
        drug2_loader_train = DataLoader(drug2_data_train, batch_size=batch_size, shuffle=None)
        drug2_loader_test = DataLoader(drug2_data_test, batch_size=batch_size, shuffle=None)


        drug_dict = get_dict("dict")

        encoder_file = '{}_{}_{}'.format(encoder,batch_size, epochs)
        """
        读取预训练模型的数据
        使用 GNN encoder or transformer encoder 来预测药物下游任务
        """
        if encoder == 1:
        #GNN
            model_encoder = GCNet().cuda()
            model_encoder.load_state_dict(torch.load('results/model/model_encoder'+encoder_file+'.pkl'))
            model = Model(encoder=model_encoder).to(device)
        if encoder == 2:
        # transformer
            model_encoder_transformer = Transformer(len(drug_dict))
            model_encoder_transformer.load_state_dict(torch.load('results/model/model_encoder'+encoder_file+'.pkl'))
            model = Model_tranformer(encoder1=model_encoder_transformer, encoder2=None).to(device)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)

        if not os.path.exists('results/model_encoder'+encoder_file):
            os.makedirs('results/model_encoder'+encoder_file)
        result_file_name = 'results/model_encoder'+ encoder_file+'/comb_without_norm' + str(i) + '--result.csv'
        file_AUCs = 'results/model_encoder'+encoder_file+'/comb_without_norm' + str(i) + '--AUCs.txt'
        AUCs = ('Epoch\tmse\tmae\trmse\tr2')
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')

        best_mse = 10000000000000
        for epoch in range(epochs):
            train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1,drug_dict)

            if (epoch + 0) % 10 == 0:
                S, T = predicting(model, device, drug1_loader_test, drug2_loader_test,drug_dict)
                # T is correct score
                # S is predict score
                # Y is predict label

                # compute preformence

                mae, mse, rmse = compute_mae_mse_rmse(S, T)
                r2 = compute_rsquared(S, T)
                # save data

                if best_mse > mse:
                    best_mse = mse
                    AUCs = [epoch, mse, mae, rmse, r2]
                    save_AUCs(AUCs, file_AUCs)
                    # torch.save(model.state_dict(), model_file_name)
                    independent_num = []
                    # independent_num.append(test_num)
                    independent_num.append(T)
                    # independent_num.append(Y)
                    independent_num.append(S)
                    txtDF = pd.DataFrame(data=independent_num)
                    txtDF.to_csv(result_file_name, index=False, header=False)

                print('best_mse', best_mse)




