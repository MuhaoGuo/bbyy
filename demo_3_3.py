import pyqpanda as pq
# from pyvqnet.qnn.qdrl.vqnet_model import qdrl_circuit
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import adam
from pyvqnet.nn.loss import CategoricalCrossEntropy, MeanSquaredError
from pyvqnet.tensor import QTensor, zeros, concatenate
import numpy as np
from pyvqnet.nn.module import Module
from pyvqnet.qnn.measure import expval
from pyvqnet.nn import activation as F
from pyvqnet.nn import Sigmoid, Tanh, Linear
from torch.utils.data import Dataset, TensorDataset, DataLoader
from dataloader import data_loader
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
matplotlib.use( 'tkagg' )
from tkinter import *

import math
from pyqpanda import init_quantum_machine, QMachineType

# 这个只是个样板，真正qdrl_circuit也可以直接倒入包，也可以自己定义。
# def qdrl_circuit(input, weights ,qlist ,clist ,machine):
#     x1 = input.squeeze()
#     param1 = weights.squeeze()
#     # 使用pyqpanda接口构建量子线路实例
#     circult = pq.QCircuit()
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[0]
#     circult.insert(pq.RZ(qlist[0], x1[0]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位x1[1]
#     circult.insert(pq.RY(qlist[0], x1[1]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[2]
#     circult.insert(pq.RZ(qlist[0], x1[2]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[0]
#     circult.insert(pq.RZ(qlist[0], param1[0]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位param1[1]
#     circult.insert(pq.RY(qlist[0], param1[1]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[2]
#     circult.insert(pq.RZ(qlist[0], param1[2]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[0]
#     circult.insert(pq.RZ(qlist[0], x1[0]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位x1[1]
#     circult.insert(pq.RY(qlist[0], x1[1]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[2]
#     circult.insert(pq.RZ(qlist[0], x1[2]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[3]
#     circult.insert(pq.RZ(qlist[0], param1[3]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位param1[4]
#     circult.insert(pq.RY(qlist[0], param1[4]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[5]
#     circult.insert(pq.RZ(qlist[0], param1[5]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[0]
#     circult.insert(pq.RZ(qlist[0], x1[0]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位x1[1]
#     circult.insert(pq.RY(qlist[0], x1[1]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[2]
#     circult.insert(pq.RZ(qlist[0], x1[2]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[6]
#     circult.insert(pq.RZ(qlist[0], param1[6]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位param1[7]
#     circult.insert(pq.RY(qlist[0], param1[7]))
#     # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[8]
#     circult.insert(pq.RZ(qlist[0], param1[8]))
#     # 构建量子程序
#     prog = pq.QProg()
#     prog.insert(circult)
#     print(circult)
#
#     # 获取概率测量值
#     prob = machine.prob_run_dict(prog, qlist, -1)
#     prob = list(prob.values())
#     return prob

#待训练参数个数
param_num = 4
#量子计算模块量子比特数
qbit_num = 4

# _circuit_input
# _circuit_update
# _circuit_output

def _circuit_forget(input, weights, qlist ,clist ,machine):
    machine = init_quantum_machine(QMachineType.CPU)
    qlist = machine.qAlloc_many(qbit_num)

    x1 = input.squeeze()
    param1 = weights.squeeze()
    # 使用pyqpanda接口构建量子线路实例
    circuit = pq.QCircuit()
    # circuit = VariationalQuantumCircuit()

    # Encoding 部分:
    for i in range(qbit_num):
        circuit.insert(pq.H(qlist[i]))
        circuit.insert(pq.RY(qlist[i], np.arctan(x1[i])))
        # circuit.insert(pq.RZ(qlist[i], np.arctan(x1[i] **2)))

    # 参数训练电路：
    # (1) Entangling 部分
    circuit.insert(pq.CNOT(qlist[0], qlist[1]))
    circuit.insert(pq.CNOT(qlist[1], qlist[2]))
    circuit.insert(pq.CNOT(qlist[2], qlist[3]))
    circuit.insert(pq.CNOT(qlist[3], qlist[0]))

    # circuit.insert(pq.CNOT(qlist[0], qlist[2]))
    # circuit.insert(pq.CNOT(qlist[1], qlist[3]))
    # circuit.insert(pq.CNOT(qlist[2], qlist[0]))
    # circuit.insert(pq.CNOT(qlist[3], qlist[1]))

    # (2) Variational 部分
    for i in range(0,4):
        circuit.insert(pq.RX(qlist[i], param1[i]))
    # for i in range(4,8):
    #     circuit.insert(pq.RY(qlist[i-4], param1[i]))
    # for i in range(8,12):
    #     circuit.insert(pq.RZ(qlist[i-8], param1[i]))

    # 构建量子程序
    prog = pq.QProg()
    prog.insert(circuit)
    # print(circuit)

    # # 获取概率测量值
    # prob = machine.prob_run_dict(prog, qlist, -1)
    # prob = list(prob.values())

    # 获取概率测量值 期望
    pauli_str_list = [{"Z{}".format(i): 1} for i in range(qbit_num)]
    # print("pauli_str_list", pauli_str_list)
    expectations = [expval(machine, prog, pauli_str_list[i], qlist) for i in range(qbit_num)]
    # print("expectation", expectations)
    return expectations

def _circuit_input(input, weights, qlist ,clist ,machine):
    machine = init_quantum_machine(QMachineType.CPU)
    qlist = machine.qAlloc_many(qbit_num)

    x1 = input.squeeze()
    param1 = weights.squeeze()
    # 使用pyqpanda接口构建量子线路实例
    circuit = pq.QCircuit()
    # circuit = VariationalQuantumCircuit()

    # Encoding 部分:
    for i in range(qbit_num):
        circuit.insert(pq.H(qlist[i]))
        circuit.insert(pq.RY(qlist[i], np.arctan(x1[i])))
        # circuit.insert(pq.RZ(qlist[i], np.arctan(x1[i] **2)))

    # 参数训练电路：
    # (1) Entangling 部分
    circuit.insert(pq.CNOT(qlist[0], qlist[1]))
    circuit.insert(pq.CNOT(qlist[1], qlist[2]))
    circuit.insert(pq.CNOT(qlist[2], qlist[3]))
    circuit.insert(pq.CNOT(qlist[3], qlist[0]))

    # circuit.insert(pq.CNOT(qlist[0], qlist[2]))
    # circuit.insert(pq.CNOT(qlist[1], qlist[3]))
    # circuit.insert(pq.CNOT(qlist[2], qlist[0]))
    # circuit.insert(pq.CNOT(qlist[3], qlist[1]))

    # (2) Variational 部分
    for i in range(0,4):
        circuit.insert(pq.RX(qlist[i], param1[i]))
    # for i in range(4,8):
    #     circuit.insert(pq.RY(qlist[i-4], param1[i]))
    # for i in range(8,12):
    #     circuit.insert(pq.RZ(qlist[i-8], param1[i]))

    # 构建量子程序
    prog = pq.QProg()
    prog.insert(circuit)
    # print(circuit)

    # # 获取概率测量值
    # prob = machine.prob_run_dict(prog, qlist, -1)
    # prob = list(prob.values())

    # 获取概率测量值 期望
    pauli_str_list = [{"Z{}".format(i): 1} for i in range(qbit_num)]
    # print("pauli_str_list", pauli_str_list)
    expectations = [expval(machine, prog, pauli_str_list[i], qlist) for i in range(qbit_num)]
    # print("expectation", expectations)
    return expectations

def _circuit_update(input, weights, qlist ,clist ,machine):
    machine = init_quantum_machine(QMachineType.CPU)
    qlist = machine.qAlloc_many(qbit_num)

    x1 = input.squeeze()
    param1 = weights.squeeze()
    # 使用pyqpanda接口构建量子线路实例
    circuit = pq.QCircuit()
    # circuit = VariationalQuantumCircuit()

    # Encoding 部分:
    for i in range(qbit_num):
        circuit.insert(pq.H(qlist[i]))
        circuit.insert(pq.RY(qlist[i], np.arctan(x1[i])))
        # circuit.insert(pq.RZ(qlist[i], np.arctan(x1[i] **2)))

    # 参数训练电路：
    # (1) Entangling 部分
    circuit.insert(pq.CNOT(qlist[0], qlist[1]))
    circuit.insert(pq.CNOT(qlist[1], qlist[2]))
    circuit.insert(pq.CNOT(qlist[2], qlist[3]))
    circuit.insert(pq.CNOT(qlist[3], qlist[0]))

    # circuit.insert(pq.CNOT(qlist[0], qlist[2]))
    # circuit.insert(pq.CNOT(qlist[1], qlist[3]))
    # circuit.insert(pq.CNOT(qlist[2], qlist[0]))
    # circuit.insert(pq.CNOT(qlist[3], qlist[1]))

    # (2) Variational 部分
    for i in range(0,4):
        circuit.insert(pq.RX(qlist[i], param1[i]))
    # for i in range(4,8):
    #     circuit.insert(pq.RY(qlist[i-4], param1[i]))
    # for i in range(8,12):
    #     circuit.insert(pq.RZ(qlist[i-8], param1[i]))

    # 构建量子程序
    prog = pq.QProg()
    prog.insert(circuit)
    # print(circuit)

    # # 获取概率测量值
    # prob = machine.prob_run_dict(prog, qlist, -1)
    # prob = list(prob.values())

    # 获取概率测量值 期望
    pauli_str_list = [{"Z{}".format(i): 1} for i in range(qbit_num)]
    # print("pauli_str_list", pauli_str_list)
    expectations = [expval(machine, prog, pauli_str_list[i], qlist) for i in range(qbit_num)]
    # print("expectation", expectations)
    return expectations

def _circuit_output(input, weights, qlist ,clist ,machine):
    machine = init_quantum_machine(QMachineType.CPU)
    qlist = machine.qAlloc_many(qbit_num)

    x1 = input.squeeze()
    param1 = weights.squeeze()
    # 使用pyqpanda接口构建量子线路实例
    circuit = pq.QCircuit()
    # circuit = VariationalQuantumCircuit()

    # Encoding 部分:
    for i in range(qbit_num):
        circuit.insert(pq.H(qlist[i]))
        circuit.insert(pq.RY(qlist[i], np.arctan(x1[i])))
        # circuit.insert(pq.RZ(qlist[i], np.arctan(x1[i] **2)))

    # 参数训练电路：
    # (1) Entangling 部分
    circuit.insert(pq.CNOT(qlist[0], qlist[1]))
    circuit.insert(pq.CNOT(qlist[1], qlist[2]))
    circuit.insert(pq.CNOT(qlist[2], qlist[3]))
    circuit.insert(pq.CNOT(qlist[3], qlist[0]))

    # circuit.insert(pq.CNOT(qlist[0], qlist[2]))
    # circuit.insert(pq.CNOT(qlist[1], qlist[3]))
    # circuit.insert(pq.CNOT(qlist[2], qlist[0]))
    # circuit.insert(pq.CNOT(qlist[3], qlist[1]))

    # (2) Variational 部分
    for i in range(0,4):
        circuit.insert(pq.RX(qlist[i], param1[i]))
    # for i in range(4,8):
    #     circuit.insert(pq.RY(qlist[i-4], param1[i]))
    # for i in range(8,12):
    #     circuit.insert(pq.RZ(qlist[i-8], param1[i]))

    # 构建量子程序
    prog = pq.QProg()
    prog.insert(circuit)
    # print(circuit)

    # # 获取概率测量值
    # prob = machine.prob_run_dict(prog, qlist, -1)
    # prob = list(prob.values())

    # 获取概率测量值 期望
    pauli_str_list = [{"Z{}".format(i): 1} for i in range(qbit_num)]
    # print("pauli_str_list", pauli_str_list)
    expectations = [expval(machine, prog, pauli_str_list[i], qlist) for i in range(qbit_num)]
    # print("expectation", expectations)
    return expectations


#定义一个继承于Module的机器学习模型类
class QModel(Module):
    def __init__(self):
        super(QModel, self).__init__()
        #使用QuantumLayer类，可以把带训练参数的量子线路纳入VQNet的自动微分的训练流程中
        self._circuit_forget = QuantumLayer(_circuit_forget, param_num, "cpu", qbit_num)
        self._circuit_input = QuantumLayer(_circuit_input, param_num, "cpu", qbit_num)
        self._circuit_update = QuantumLayer(_circuit_update, param_num, "cpu", qbit_num)
        self._circuit_output = QuantumLayer(_circuit_output, param_num, "cpu", qbit_num)

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.hidden_units = 16
        self.feature_size = 1
        self.linear1 = Linear(self.hidden_units + self.feature_size, qbit_num)
        self.linear2 = Linear(4, 16, use_bias=True)

    #定义模型前向函数
    def forward(self, x, initial_state):
        # print("x.shape", x.shape)  # [30, 4, 1]

        h_t, c_t = initial_state
        h_t = h_t[0]
        c_t = c_t[0]
        # print("h_t", h_t.shape)  # [30, 16]
        # print("c_t", c_t.shape)  # [30, 16]

        seq_length = 4
        for t in range(seq_length):
            x_t = x[:, t, :]
            # print("x_t", x_t.shape)  # [30, 1]
            v_t = concatenate([h_t, x_t], axis=1) # [30, 17]
            # print("v_t", v_t.shape)

            y_t = self.linear1(v_t)  # [30, 17] -> [30, 4]
            # print("y_t", y_t.shape) #  [30, 4]

            f_t = self.sigmoid(self.linear2(self._circuit_forget(y_t)))  # [30, 16]
            i_t = self.sigmoid(self.linear2(self._circuit_forget(y_t)))
            g_t = self.sigmoid(self.linear2(self._circuit_forget(y_t)))
            o_t = self.sigmoid(self.linear2(self._circuit_forget(y_t)))
            # print("f_t", f_t.shape)

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * self.tanh(c_t)
            # print("c_t", c_t.shape) # [30, 16]
            # print("h_t", h_t.shape) # [30, 16]
        # print("h_t", h_t.shape)
        return h_t

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden_units = 16
        self.num_layers = 1
        self.lstm = QModel()
        self.linear = Linear(self.hidden_units, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = zeros((self.num_layers, batch_size, self.hidden_units))
        c0 = zeros((self.num_layers, batch_size, self.hidden_units))
        # print(h0.shape) # [1, 30, 16]
        # print(c0.shape)  #[1, 30, 16]

        hn = self.lstm(x, (h0, c0))
        # print("hn", hn.shape)  # [30, 16]
        out = self.linear(hn)    # First dim of Hn is num_layers, which is set to 1 above.
        # print("out", out.shape)  # [30, 1]
        return out

#实例化定义的模型
model = Model()
#定义一个优化器，这里用的是Adam
optimizer = adam.Adam(model.parameters(), lr=0.1)
#定义一个损失函数，这里用的交叉熵损失函数
# Closs = CategoricalCrossEntropy()
Closs = MeanSquaredError()

X_train, y_train, X_test, y_test, sc = data_loader()
# print(y_test)

def GetScore(y_pred:np.array, y_test:np.array) -> float:
    '''
    Calculate test score based on prediction
    y_pred and y_test:  Should be np.array. Like: [[1],[2],[3],...]
    Note: we should make sure there is no "0" in y_test, otherwise, the acc will become infinite !!!!
    '''
    y_pred_list = y_pred.ravel().tolist()
    y_test_list = y_test.ravel().tolist()
    E2 = []
    for i, y in enumerate(y_test_list):
        if y == 0:
            continue
        else:
            y_true, y_pred = y_test_list[i], y_pred_list[i]
            Ei = abs(y_pred - y_true) / y_true
            E2.append(Ei**2)
    Acc = 1 - math.sqrt(np.array(E2).mean())
    # print("Acc")
    return Acc

def train():
    data = TensorDataset(X_train, y_train)
    # 最大训练迭代次数
    epoch = 5
    batch_size = 10
    for i in range(epoch):
        model.train()
        # accuracy = 0
        count = 0
        loss = 0
        myloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, drop_last=False)
        for batch in myloader:
            X_batch, y_batch = batch[0].numpy(), batch[1].numpy()
            # print("X_batch", X_batch.shape) #  (30, 4, 1)
            # print("y_batch", y_batch.shape) #  (30, 1)

            # 优化器中缓存梯度清零
            optimizer.zero_grad()
            # 模型前向计算
            output = model(X_batch)
            # print(output.shape)
            # 损失函数计算
            losss = Closs(y_batch, output)
            # 损失反向传播
            losss.backward()
            # 优化器参数更新
            optimizer._step()
            # 计算准确率等指标
            # accuracy += GetScore(output, y_batch)

            loss += losss.item()
            count += batch_size
        # print("Epoch:{}, train_accuracy:{}".format(i, accuracy/count))
        print("Epoch:{}, train_loss:{}".format(i, loss/count))


def EvalModel(X_test, y_test):
    batch_size = 1
    data = TensorDataset(X_test, y_test)
    myloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, drop_last=False)

    y_preds = []
    for batch in myloader:
        X_batch, y_batch = batch[0].numpy(), batch[1].numpy()
        X_batch, y_batch = QTensor(X_batch), QTensor(y_batch)
        # print(X_batch)
        # print(y_batch)
        output = model(X_batch)
        y_preds.append(output)

    y_preds = np.array([QTensor.to_numpy(y[0])for y in y_preds])  # QTensor to numpy
    y_test_array = y_test.numpy()      # Tensor to numpy

    if sc != None:
        y_preds = sc.inverse_transform(y_preds)   # 从(0,1)又转化为原来的尺寸
        y_test = sc.inverse_transform(y_test)     # 从(0,1)又转化为原来的尺寸

    test_accuracy = GetScore(y_pred = y_preds, y_test= y_test_array)
    print("Test Accuracy:{}".format(test_accuracy))
    print("y_preds", y_preds)

    plot = True
    if plot == True:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(y_preds, label='Pred')
        ax.plot(y_test, label='True')
        plt.legend()
        plt.show()


print("start training...........")
train()
model.eval()
print("start eval train...................")
EvalModel(X_train, y_train)
print("start eval test...................")
EvalModel(X_test, y_test)