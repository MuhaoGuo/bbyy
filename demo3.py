import pyqpanda as pq

#导入必须的库和函数
from pyvqnet.qnn.qdrl.vqnet_model import qdrl_circuit
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import adam
from pyvqnet.nn.loss import CategoricalCrossEntropy, MeanSquaredError
from pyvqnet.tensor import QTensor
import numpy as np
from pyvqnet.nn.module import Module
from pyvqnet.qnn.measure import expval

def qdrl_circuit(input ,weights ,qlist ,clist ,machine):
    x1 = input.squeeze()
    param1 = weights.squeeze()
    # 使用pyqpanda接口构建量子线路实例
    circult = pq.QCircuit()
    # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[0]
    circult.insert(pq.RZ(qlist[0], x1[0]))
    # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位x1[1]
    circult.insert(pq.RY(qlist[1], x1[1]))
    # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[2]
    circult.insert(pq.RZ(qlist[2], x1[2]))
    # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[0]
    circult.insert(pq.RZ(qlist[0], param1[0]))
    # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位param1[1]
    circult.insert(pq.RY(qlist[1], param1[1]))
    # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[2]
    circult.insert(pq.RZ(qlist[2], param1[2]))
    # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[0]
    # circult.insert(pq.RZ(qlist[0], x1[0]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位x1[1]
    # circult.insert(pq.RY(qlist[0], x1[1]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[2]
    # circult.insert(pq.RZ(qlist[0], x1[2]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[3]
    # circult.insert(pq.RZ(qlist[0], param1[3]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位param1[4]
    # circult.insert(pq.RY(qlist[0], param1[4]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[5]
    # circult.insert(pq.RZ(qlist[0], param1[5]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[0]
    # circult.insert(pq.RZ(qlist[0], x1[0]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位x1[1]
    # circult.insert(pq.RY(qlist[0], x1[1]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位x1[2]
    # circult.insert(pq.RZ(qlist[0], x1[2]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[6]
    # circult.insert(pq.RZ(qlist[0], param1[6]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RY门，参数位param1[7]
    # circult.insert(pq.RY(qlist[0], param1[7]))
    # # 使用pyqpanda接口在第一个量子比特上插入逻辑门RZ门，参数位param1[8]
    # circult.insert(pq.RZ(qlist[0], param1[8]))
    # 构建量子程序
    prog = pq.QProg()
    prog.insert(circult)
    # print("---")
    # print(circult)
    # print("---")

    # 获取概率测量值 期望
    pauli_str_list = {}
    for i in range(len(qlist)):
        pauli_str_list[f"Z{i}"] = 1

    expectation = expval(machine, prog, pauli_str_list, qlist)
    # prob = machine.prob_run_dict(prog, qlist, -1)
    # print(prob)
    # prob = list(prob.values())
    # print(prob)
    return expectation


#待训练参数个数
param_num = 3
#量子计算模块量子比特数
qbit_num  = 3
#定义一个继承于Module的机器学习模型类
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        #使用QuantumLayer类，可以把带训练参数的量子线路纳入VQNet的自动微分的训练流程中
        self.pqc = QuantumLayer(qdrl_circuit, param_num, "cpu", qbit_num)
    #定义模型前向函数
    def forward(self, x):
        x = self.pqc(x)
        return x


# 随机产生待训练数据的函数
def circle(samples:int,  rads = np.sqrt(2/np.pi)) :
    data_x, data_y = [], []
    for i in range(samples):
        x = 2*np.random.rand(2) - 1
        y = [0,1]
        if np.linalg.norm(x) < rads:
            y = [1,0]
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)

# 数据载入函数
def get_minibatch_data(x_data, label, batch_size):
    for i in range(0,x_data.shape[0]-batch_size+1,batch_size):
        idxs = slice(i, i + batch_size)
        yield x_data[idxs], label[idxs]

#计算准确率的函数
def get_score(pred, label):
    pred, label = np.array(pred.data), np.array(label.data)
    pred = np.argmax(pred, axis=1)
    score = np.argmax(label,1)
    score = np.sum(pred == score)
    return score


#实例化定义的模型
model = Model()
#定义一个优化器，这里用的是Adam
optimizer = adam.Adam(model.parameters(),lr =0.6)
#定义一个损失函数，这里用的交叉熵损失函数
# Closs = CategoricalCrossEntropy()
Closs = MeanSquaredError()

def train():
    # 随机产生待训练数据
    x_train, y_train = circle(500)
    x_train = np.hstack((x_train, np.zeros((x_train.shape[0], 1))))
    # 定义每个批次训练的数据个数
    batch_size = 32
    # 最大训练迭代次数
    epoch = 10
    print("start training...........")
    for i in range(epoch):
        model.train()
        accuracy = 0
        count = 0
        loss = 0
        for data, label in get_minibatch_data(x_train, y_train, batch_size):
            # 优化器中缓存梯度清零
            optimizer.zero_grad()
            # 模型前向计算
            output = model(data)
            label = label.T[0].reshape(-1,1)
            print(type(output))
            print(type(label))
            print("output",output.shape)
            print("label", label.shape)
            # 损失函数计算
            losss = Closs(label, output)
            # 损失反向传播
            losss.backward()
            # 优化器参数更新
            optimizer._step()
            # 计算准确率等指标
            accuracy += get_score(output,label)

            loss += losss.item()
            count += batch_size

        print(f"epoch:{i}, train_accuracy:{accuracy/count}")
        print(f"epoch:{i}, train_loss:{loss/count}\n")

def test():
    batch_size = 1
    model.eval()
    print("start eval...................")
    xtest, y_test = circle(500)
    test_accuracy = 0
    count = 0
    x_test = np.hstack((xtest, np.zeros((xtest.shape[0], 1))))
    predicted_test = []
    for test_data, test_label in get_minibatch_data(x_test,y_test, batch_size):

        test_data, test_label = QTensor(test_data),QTensor(test_label)
        output = model(test_data)
        test_accuracy += get_score(output, test_label)
        count += batch_size

    print(f"test_accuracy:{test_accuracy/count}")

train()