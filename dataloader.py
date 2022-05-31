import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import Tensor
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
    train_raw = pd.read_csv("./data/Train.csv")
    test_raw = pd.read_csv("./data/Test.csv")

    columns_names =['Date', 'Atmospheric Pressure', 'Minimum Temperature', 'Maximum Temperature', 'Relative Humidity', 'Wind Speed']

    print(train_raw.columns)
    print(test_raw.columns)
    print(train_raw.shape)

    #todo 一个feature
    data = train_raw["Atmospheric Pressure"]
    data = np.array(data)
    data = torch.unsqueeze(Tensor(data), 1)
    print("data", data.shape)

    # 画图
    # plt.plot(data,)
    # plt.show()

    # 归一化: 压缩 数据到(0,1)
    sc = MinMaxScaler() # 压缩至（0，1）
    data = sc.fit_transform(data)
    # print(data)
    print("data", data.shape)

    # 产生时序的 feature: X 和 label: y
    seq_length = 100
    X, y = sliding_windows(data, seq_length)
    print("x:", X.shape)
    print("y:", y.shape)

    # 分开训练，测试集： 按0.67 比例，加载到 Tensor 中.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # train_size = int(len(y) * 0.67)
    # test_size = len(y) - train_size

    # 总数据：
    dataX = Variable(torch.Tensor(np.array(X)))
    dataY = Variable(torch.Tensor(np.array(y)))
    print("dataX:", dataX.shape)
    print("dataY:", dataY.shape)
    # 训练集：
    X_train = Variable(torch.Tensor(np.array(X_train)))
    y_train = Variable(torch.Tensor(np.array(y_train)))
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    # 测试集
    X_test = Variable(torch.Tensor(np.array(X_test)))
    y_test = Variable(torch.Tensor(np.array(y_test)))
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    return(X_train, y_train, X_test, y_test, sc)

data_loader()
