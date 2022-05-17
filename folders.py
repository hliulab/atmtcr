import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
# import cv2
import scipy.io
import numpy as np
import csv
# from openpyxl import load_workbook
import torchvision
import torch

# 固定随机数种子
seed = 1
random.seed(seed)

class Folder(data.Dataset):

    def __init__(self):

        sample = []
        data_1 = np.load("MHC-encoder/HLA_antigen_encoded_result_test.npy")
        data_2 = torch.load("TCR-encoder/train_feature_graph.pt")


        data_1 = torch.from_numpy(data_1)

        print(data_1.shape)
        print(data_2.shape)

        p_data_1 = data_1[:int(data_1.shape[0]/2),:]
        n_data_1 = data_1[int(data_1.shape[0]/2):,:]
        p_data_2 = data_2[:int(data_2.shape[0]/2),:]
        n_data_2 = data_2[int(data_2.shape[0] / 2):, :]


        temp = torch.zeros_like(n_data_2)
        rand_lst = list(range(n_data_2.shape[0]))
        random.shuffle(rand_lst)
        for i, idex in enumerate(rand_lst):
            temp[i,:] = n_data_2[idex,:]
        n_data_2 = temp
        # print(p_data_1.shape,p_data_2.shape)
        # print(n_data_1.shape, n_data_2.shape)


        p_data = torch.cat((p_data_1, p_data_2), 1)
        n_data = torch.cat((n_data_1, n_data_2), 1)

        # print(p_data.shape)
        # print(n_data.shape)

        for i in range(p_data.shape[0]):
            sample.append((p_data[i,:], 1))
        for i in range(n_data.shape[0]):
            sample.append((n_data[i,:], 0))


        random.shuffle(sample)
        tr_sample = sample[0:int(round(0.8 * len(sample)))]
        te_sample = sample[int(round(0.8 * len(sample))):len(sample)]

        self.train_data = ()
        self.test_data = ()
        self.train_label = ()
        self.test_label = ()

        for item, (data, num) in enumerate(tr_sample):
            self.train_data = self.train_data + (data,)
            self.train_label = self.train_label + (num,)
        for item, (data, num) in enumerate(te_sample):
            self.test_data = self.test_data + (data,)
            self.test_label = self.test_label+ (num,)
        print("1")




class Dataset(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label


    def __len__(self):
        length = len(self.label)
        return length

    def __getitem__(self, index):
        return self.data[index], self.label[index]



if __name__ == '__main__':
     print("1")

