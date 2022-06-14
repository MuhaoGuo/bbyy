import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def sliding_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length-1):
        _x = data[i: (i+seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

def data_loader() -> (Variable, Variable, Variable, Variable, object):
    seq_length = 4
    train_raw = pd.read_csv("./data/Train.csv")
    test_raw = pd.read_csv("./data/Test.csv")
    columns_names =['Date', 'Atmospheric Pressure', 'Minimum Temperature', 'Maximum Temperature', 'Relative Humidity', 'Wind Speed']

    # print(train_raw.columns)
    # print(test_raw.columns)
    # print("train_raw.shape", train_raw.shape)

    #todo 一个feature "Atmospheric Pressure"
    train = train_raw["Atmospheric Pressure"]
    train = np.array(train)
    test = test_raw["Atmospheric Pressure"]
    test = np.array(test)

    print("train", train.shape)
    print("test", test.shape)

    # 画图
    plot = True
    plot = False
    if plot == True:
        fig = plt.figure(figsize=(10, 4))
        # plt.subplots_adjust(left=0.04, bottom=0.05, right=0.85, top=0.85, wspace=0, hspace=0)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Train [Atmospheric Pressure]")
        ax.plot(train,)

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title("Test [Atmospheric Pressure]")
        ax1.plot(test,)
        plt.show()

    # 归一化: 压缩 数据到(0,1)
    normlization = False
    # normlization = True
    sc = None
    if normlization == True:
        sc = MinMaxScaler()               # 压缩至（0，1）

        train = np.expand_dims(train, axis=1)
        train = sc.fit_transform(train)     # fit_transform 只能输入的数据是纬度是每个数字要1纬度，如：[[1],[2],[4]...]
        train = np.squeeze(train)           # [[1],[2],[4]...] 转化为 [1,2,4...]
        # print("data", data.shape)
        test = np.expand_dims(test, axis=1)
        test = sc.transform(test)
        test = np.squeeze(test)

    # 产生时序的 feature: X 和 label: y
    X_train, y_train = sliding_windows(train, seq_length)
    X_test, y_test = sliding_windows(test, seq_length)

    # 训练集：
    X_train = Variable(torch.Tensor(np.array(X_train)))
    y_train = Variable(torch.Tensor(np.array(y_train)))
    # print("X_train:", X_train.shape)
    # print("y_train:", y_train.shape)
    # 测试集
    X_test = Variable(torch.Tensor(np.array(X_test)))
    y_test = Variable(torch.Tensor(np.array(y_test)))
    # print("X_test:", X_test.shape)
    # print("y_test:", y_test.shape)

    X_train = torch.unsqueeze(X_train, dim=2)
    y_train = torch.unsqueeze(y_train, dim=1)
    X_test = torch.unsqueeze(X_test, dim=2)
    y_test = torch.unsqueeze(y_test, dim=1)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    return(X_train, y_train, X_test, y_test, sc)

data_loader()
