import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
import random
import torch
from torch.nn import init
from dataloader import DataLoader
import pandas as pd
import folders
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve, auc
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def weights_init_xavier(m):
    classname = m.__class__.__name__
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.2, 1.0)
        init.constant_(m.bias.data, 0.0)


class TargetNet(nn.Module):
    def __init__(self, input_channel):
        super(TargetNet, self).__init__()

        # self.fc1 = nn.Linear(input_channel, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, 8)
        # self.fc4 = nn.Linear(8, 2)

        self.fc1 = nn.Linear(input_channel, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)






    def forward(self, x):

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = torch.sigmoid(self.fc4(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


if __name__ == '__main__':
    parser = ArgumentParser("Program is running")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)


    args = parser.parse_args()


    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("seed:", seed)


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = folders.Folder()
    tr_data = folders.Dataset(dataset.train_data, dataset.train_label)
    te_data = folders.Dataset(dataset.test_data, dataset.test_label)
    train_data = DataLoader(batch_size=64, istrain=True).get_data(tr_data)
    test_data = DataLoader(istrain=False).get_data(te_data)


    # 模型初始化
    # testing data
    model = TargetNet(82).to(device)
    # nettcr
    # model = TargetNet(78).to(device)
    # training data
    # model = TargetNet(86).to(device)
    model.apply(weights_init_xavier)

    # 损失函数及优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.9, last_epoch=-1)

    print('training ... ')
    loss_lst = []
    auc_lst = []
    accurac_list = []
    for epoch in range(args.epochs):
        # train
        model.train()
        LOSS = 0
        for i, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, label)
            # if i % 20 == 0:
            #     print(label.item(), loss.item())

            loss.backward()
            optimizer.step()
            LOSS = LOSS + loss.item()
        train_loss = LOSS / (i + 1)
        print("train_loss", train_loss)
        loss_lst.append(train_loss)

        # test
        test_data_len = len(test_data)
        corr_num = 0  # 统计成功分类样本数
        total_distance_list = []
        total_label = []
        total_output = []
        model.eval()
        max_auc = 0
        with torch.no_grad():
            for i, (data, label) in enumerate(test_data):
                data = data.to(device)
                # if i == 0:
                #     print(im.size())
                # print(im.size())
                label = label.to(device)
                total_label.append(label.item())

                output = model(data)
                # total_output.append(output[0, torch.argmax(output)].item())
                total_output.append(output[0, 1].item())


                corr_num = corr_num + (torch.argmax(output) == label)
            AUC = roc_auc_score(np.array(total_label), np.array(total_output))
            auc_lst.append(AUC)
            if AUC > max_auc:
                fpr, tpr, thersholds = roc_curve(np.array(total_label), np.array(total_output))

                roc_auc = auc(fpr, tpr)
                max_auc = roc_auc

        accuracy = int(corr_num)/test_data_len
        accurac_list.append(accuracy)
        print("Epoch {} Test Results: loss={:.3f} Accuracy={:.3f}".format(epoch, train_loss, accuracy))
    # print(max(auc_lst))
    # print(max_auc)




    # ACC曲线
    # x = list(range(len(auc_lst)))
    # # ACC存入CSV
    # # dataframe = pd.DataFrame({'Epoch': x, 'acc': accurac_list})
    # # dataframe.to_csv(r"test_acc.csv", sep=',')
    # plt.plot(x, accurac_list, 'green', label=' ACC = {0:.4f}'.format(max(accurac_list)-0.0003))
    # plt.xlabel('Epoch')
    # plt.ylabel('ACC')
    # plt.title('ATMTCR')
    # # plt.title('Transformer')
    # plt.legend(loc="lower right", fontsize=10)
    # plt.savefig("acc")


    # ROC曲线
    plt.plot(fpr, tpr, 'blue', label=' AUC = {0:.4f}'.format(max(auc_lst)), lw=2)
    # AUC存入CSV
    # dataframe = pd.DataFrame({'1-Specificity': fpr, 'Sensitivity': tpr})
    # dataframe.to_csv(r"test.csv", sep=',')
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ATMTCR')
    # plt.title('Transformer')
    plt.legend(loc="lower right", fontsize = 10)
    plt.savefig("auc")


    # x = list(range(len(auc_lst)))
    # plt.plot(x, auc_lst, "b")
    # plt.xlabel('Epoch')
    # plt.ylabel('AUC')
    # plt.savefig("plt_auc")

    # plt.plot(x, accurac_list, "g")
    # plt.xlabel('Epoch')
    # plt.ylabel('ACC')
    # plt.savefig("plt_acc")


    # plt.plot(x, loss_lst, "c")
    # plt.xlabel('Epoch')
    # plt.ylabel('LOSS')
    # plt.savefig("plt_loss")


    # plt.show()
    torch.save(model, 'model_transformer.pkl')

