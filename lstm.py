import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()  # LSTM 继承 nn.Module

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size,  # The number of expected features in the input x
                            hidden_size=hidden_size,  # The number of features in the hidden state h
                            num_layers=num_layers,
                            # Number of recurrent layers. Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
                            batch_first=True,
                            # If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
                            )

        self.fc = nn.Linear(hidden_size, num_classes)  # 全联接层，应该在最后一层吧

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # print("x.size(0)", x.size(0)) # x.size(0) 93
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)  # 最后一层是线性全连接层
        return out



from dataloader import  data_loader
X_train, y_train, X_test, y_test, sc = data_loader()
# X_train: torch.Size([468, 100, 1])
# y_train: torch.Size([468, 1])

num_epochs = 2000
learning_rate = 0.01
input_size = 1  # The number of expected features in the input x
hidden_size = 2  # The number of features in the hidden state h
num_layers = 1  # Number of recurrent layers
num_classes = 1  #


seq_length = len(X_train[0])
# print(seq_length)
lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length=seq_length)
criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)


# Train the model
for epoch in range(num_epochs):  # 2000轮训练。
    outputs = lstm(X_train)  # 用训练集训练
    optimizer.zero_grad()  # 第一步 导数清零 clears old gradients from the last step

    # obtain the loss function
    loss = criterion(outputs, y_train)

    loss.backward()  # 第二步：computes the derivative of the loss w.r.t. the parameters

    optimizer.step()  # 第三步：causes the optimizer to take a step based on the gradients of the parameters.
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



# 评价模型
# 训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有BN层和Dropout所带来的的性质。
lstm.eval()  #不启用 Batch Normalization 和 Dropout

dataX = torch.cat((X_train, X_test), dim=0)
datay = torch.cat((y_train, y_test), dim=0)


# 预测：
train_predict = lstm(dataX)                 # 这里是 用的全部数据。包括训练集 和 测试集
print("train_predict", train_predict.shape) # 得到训练集结果 train_predict

data_predict = train_predict.data.numpy()   # 转化为 numpy
print("data_predict", data_predict.shape)

dataY_plot = datay.data.numpy()             # 训练集 真实值 dataY
print("dataY_plot", dataY_plot.shape)

data_predict = sc.inverse_transform(data_predict) # 从(0,1)又转化为原来的尺寸
dataY_plot = sc.inverse_transform(dataY_plot)     # 从(0,1)又转化为原来的尺寸

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
train_size = len(y_train)
plt.axvline(x=train_size, c='r', linestyle='--')

ax.plot(dataY_plot, c='blue', label='true')
ax.plot(data_predict, c='red', label='pred')
plt.title('Time-Series Prediction')
ax.legend()
plt.show()


