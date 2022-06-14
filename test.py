import math
import numpy as np
from pyvqnet.tensor import QTensor
from pyvqnet.nn import Sigmoid, Tanh

# a = np.array([[1],[0],[0]])
# b = np.array([[1],[1],[1]])
#
# print(a)
# print(b)
#
# def score(y_pred:np.array, y_test:np.array) -> float:
#     '''
#     Calculate test score based on prediction
#     y_pred and y_test should be np.array with 2-dim
#     like: [[1],[2],[3],...]
#     '''
#     E2 = (abs(y_pred - y_test)/y_test) ** 2
#     Acc = 1 - math.sqrt(E2.mean())
#     print("Acc:", Acc)
#     return Acc
#
# def GetScore(y_pred:np.array, y_test:np.array) -> float:
#     '''
#     Calculate test score based on prediction
#     y_pred and y_test should be np.array with 2-dim
#     like: [[1],[2],[3],...]
#     Note: we should make sure there is no "0" in y_test, otherwise, the acc will become infinite!!!!
#     '''
#     y_pred_list = y_pred.ravel().tolist()
#     y_test_list = y_test.ravel().tolist()
#
#     E2 = []
#     for i, y in enumerate(y_test_list):
#         if y == 0:
#             continue
#         else:
#             y_true = y_test_list[i]
#             y_pred = y_pred_list[i]
#             Ei = abs(y_pred - y_true) / y_true
#             E2.append(Ei**2)
#     Acc = 1 - math.sqrt(np.array(E2).mean())
#     print(Acc)
#
# score(a,b)
# GetScore(a,b)
#
# a = np.array([1,0,0])
# b = np.array([1,1,1])
# GetScore(a,b)


a = QTensor([[8],[3],[4]])
a = Sigmoid(a)
print(a)
