import numpy as np
from cec2013lsgo.cec2013 import Benchmark
from pandas import read_csv
from os.path import abspath, dirname


def AIM(Dim, func, scale, noise):
    intensity = []
    aim_add = np.zeros((Dim, Dim))
    base = np.zeros(Dim)
    for i in range(Dim):
        base[i] = scale[0]
    intercept = func(base) * (1 + np.random.normal(loc=0, scale=noise, size=None))
    sP = singlePerturb(Dim, func, scale, noise)
    for i in range(Dim):
        for j in range(i+1, Dim):
            delta1 = DG(Dim, i, j, func, intercept, scale, sP, noise)
            aim_add[i][j], aim_add[j][i] = delta1, delta1
            intensity.append(delta1)
    return aim_add, intensity


def DG(Dim, e1, e2, func, intercept, scale, sP, noise):
    base = np.zeros(Dim)
    for i in range(Dim):
        base[i] = scale[0]
    base[e1] = scale[1]
    base[e2] = scale[1]
    f1, f2 = sP[e1], sP[e2]
    f3 = func(base) * (1 + np.random.normal(loc=0, scale=noise, size=None))
    delta = abs(f3 + intercept - f1 - f2)
    return delta


def singlePerturb(Dim, func, scale, noise):
    sP = []
    for i in range(Dim):
        base = np.zeros(Dim)
        for j in range(Dim):
            base[j] = scale[0]
        base[i] = scale[1]
        sP.append(func(base) * (1 + np.random.normal(loc=0, scale=noise, size=None)))
    return sP


def save(path, data):
    with open(path, 'ab') as f:
        np.savetxt(f, data, delimiter=",", fmt='%d')



def open_csv(path):
    data = np.array(read_csv(path, header=None))
    return data


def S_DSM(Dim, M1, M2):
    same = 0
    total = 0
    for i in range(Dim):
        for j in range(i+1, Dim):
            total += 1
            if M1[i][j] == M2[i][j]:
                same += 1
    return same / total


def AIM_to_DSM(Dim, AIM, SI):
    DSM = np.zeros((Dim, Dim))
    for i in range(Dim):
        for j in range(i+1, Dim):
            if AIM[i][j] > SI:
                DSM[i][j] = 1
                DSM[j][i] = 1
    return DSM


Dims = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 905, 905, 1000]

bench = Benchmark()
for func_num in range(4, 10):
    Dim = Dims[func_num-1]
    this_path = dirname(abspath(__file__))
    func = bench.get_function(func_num)
    info = bench.get_info(func_num)
    scale_range = [info["lower"], info["upper"]]

    Aim_01, intensity_01 = AIM(Dim, func, scale_range, 0.1)
    Aim_001, intensity_001 = AIM(Dim, func, scale_range, 0.01)

    P01_AIM_path = this_path + "/DSMs/P_AIM_0.1/f" + str(func_num) + ".csv"
    P001_AIM_path = this_path + "/DSMs/P_AIM_0.01/f" + str(func_num) + ".csv"
    P01_DSM_path = this_path + "/DSMs/P_DSM_0.1/f" + str(func_num) + ".csv"
    P001_DSM_path = this_path + "/DSMs/P_DSM_0.01/f" + str(func_num) + ".csv"

    index = int(0.995 * (len(intensity_01)-1))
    intensity_01 = sorted(intensity_01)
    intensity_001 = sorted(intensity_001)

    SI_01 = intensity_01[index]
    SI_001 = intensity_001[index]

    DSM_01 = AIM_to_DSM(Dim, Aim_01, SI_01)
    DSM_001 = AIM_to_DSM(Dim, Aim_001, SI_001)

    save(P01_DSM_path, DSM_01)
    save(P001_DSM_path, DSM_001)
