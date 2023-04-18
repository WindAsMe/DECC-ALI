import numpy as np
from pandas import read_csv
from os.path import abspath, dirname
import pandas as pd
import os
import scipy.io as sio
from itertools import *


def open(path):
    data = np.array(read_csv(path, header=None))
    return data


def DFS_main(Dim, Tuples):
    subcomponents = []
    for i in range(1, Dim+1):
        subcomponents.append([i])
    while len(Tuples) > 0:
        (e1, e2) = Tuples.pop()
        index_1 = -1
        index_2 = -1
        for i in range(len(subcomponents)):
            if e1 in subcomponents[i]:
                index_1 = i
            if e2 in subcomponents[i]:
                index_2 = i
        if index_1 == index_2:
            continue
        else:
            subcomponents[index_1].extend(subcomponents.pop(index_2))
    return subcomponents


def S_DSM(Dim, M1, M2):
    same = 0
    total = 0
    for i in range(Dim):
        for j in range(i+1, Dim):
            total += 1
            if M1[i][j] == M2[i][j]:
                same += 1
    return same / total


def DSM_to_tuple(Dim, DSM):
    Tuples = []
    for i in range(Dim):
        for j in range(i+1, Dim):
            if DSM[i][j] == 1:
                Tuples.append((i+1, j+1))
    return Tuples


Dims = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 905, 905, 1000]
for func_num in range(1, 15):
    Dim = Dims[func_num-1]
    this_path = dirname(abspath(__file__))

    P_01_path = this_path + "/DSMs/P_DSM_0.1/f" + str(func_num) + ".csv"
    P_001_path = this_path + "/DSMs/P_DSM_0.01/f" + str(func_num) + ".csv"

    DSM_01 = open(P_01_path)
    DSM_001 = open(P_001_path)

    Tuples_01 = DSM_to_tuple(Dim, DSM_01)
    Tuples_001 = DSM_to_tuple(Dim, DSM_001)
    subcomponents_01 = DFS_main(Dim, Tuples_01)
    subcomponents_001 = DFS_main(Dim, Tuples_001)
