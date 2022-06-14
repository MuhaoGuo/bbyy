import torch
# from Factory import QShallowRegressionLSTM
from vqc2 import QShallowRegressionLSTM
from torch import nn
import matplotlib.pyplot as plt

from dataloader import data_loader

X_train, y_train, X_test, y_test, sc = data_loader()

print("X_train", X_train.shape)
print("y_train", y_train.shape)

num_epochs = 10
learning_rate = 0.01
input_size = 1   # The number of expected features in the input x
hidden_size = 2  # The number of features in the hidden state h
num_layers = 1   # Number of recurrent layers
num_classes = 1  # 最后 输出层的个数，这里是回归预测，应该一个神经元就够了。

seq_length = X_train.shape[1]

lstm = QShallowRegressionLSTM(num_sensors=1, hidden_units=2, n_qubits=4) #n_qubits=4
criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# print("lstm.parameters", lstm.parameters)

# Train the model
for epoch in range(num_epochs):  # 2000轮训练。
    outputs = lstm(X_train)  # 用训练集训练   X_train: torch.Size([468, 100, 1])
    optimizer.zero_grad()    # 第一步 导数清零 clears old gradients from the last step

    # obtain the loss function
    y_train_epoch = torch.squeeze(y_train)  # 这里先把 y变成一纬的了。为了适应loss 的格式
    loss = criterion(outputs, y_train_epoch)
    loss.backward()  # 第二步：computes the derivative of the loss w.r.t. the parameters
    optimizer.step()  # 第三步：causes the optimizer to take a step based on the gradients of the parameters.

    if epoch % 1 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



# 评价模型
print(".....评价模型......")
# 训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有BN层和Dropout所带来的的性质。
lstm.eval()  #不启用 Batch Normalization 和 Dropout

dataX = torch.cat((X_train, X_test), dim=0)
datay = torch.cat((y_train, y_test), dim=0)

# 预测：
train_predict = lstm(dataX)                 # 这里是 用的全部数据。包括训练集 和 测试集
print("train_predict", train_predict)       # 得到训练集结果 train_predict
train_predict = train_predict.data.numpy().reshape(-1, 1)  # 转化为 numpy
print("data_predict", train_predict)

dataY_plot = datay.data.numpy()             # 训练集 真实值 dataY
print("dataY_plot", dataY_plot)

train_predict = sc.inverse_transform(train_predict) # 从(0,1)又转化为原来的尺寸
dataY_plot = sc.inverse_transform(dataY_plot)     # 从(0,1)又转化为原来的尺寸


#### 画图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
train_size = len(y_train)
plt.axvline(x=train_size, c='r', linestyle='--')

ax.plot(dataY_plot, c='blue', label='true')
ax.plot(train_predict, c='red', label='pred')
plt.title('Time-Series Prediction')
ax.legend()


#### 画图
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# train_size = len(y_train)
# plt.axvline(x=train_size, c='r', linestyle='--')
#
# ax.plot(dataY_plot, c='blue', label='true')
# # ax.plot(data_predict, c='red', label='pred')
# plt.title('Time-Series Prediction')
# ax.legend()
plt.show()