import torch
import torchvision
import folders
import numpy as np
import random

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

class DataLoader(object):

    def __init__(self, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain

    def get_data(self, data):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0)
        else:
            dataloader = torch.utils.data.DataLoader(
                data, batch_size=1, shuffle=True)
        return dataloader

if __name__ == '__main__':
    dataset = folders.Folder()
    tr_data = folders.Dataset(dataset.train_data, dataset.train_label)
    te_data = folders.Dataset(dataset.test_data, dataset.test_label)
    train_data = DataLoader(batch_size=64, istrain=True).get_data(tr_data)
    test_data = DataLoader(istrain=False).get_data(te_data)
    print("1")