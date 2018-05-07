import openpyxl
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import random
import math
from numpy import linalg


# p1 = np.array([1, 1])
# p2 = np.array([3, 3])
# p3 = [-1, 1]
#
# #print(np.cross([1, 0], [2, 1]))
#
# print(linalg.norm(np.cross(p2-p1, p1-p3))/linalg.norm(p2-p1))
# print(linalg.norm(p2 - p1))

# a = np.array([[1, 2], [3, 4]])
# print(a.shape)
# b = np.array([2, 1])
# print(b.shape)
# c = a * b
# print(np.dot(a, b))
# a = np.array([[1, 2, 5], [3, 4, 6]])
# print(a[:, 2])
class Test():
    def __init__(self):
        self.x = 0
    def modify(self, x):
        x = 2

    def change(self):
        self.modify(self.x)
        print(self.x)

a = Test()
a.change()
