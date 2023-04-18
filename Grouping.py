import numpy as np
from copy import deepcopy


def AIM_dLI(Dim, func, scale):
    Tuples = AIM_DSM(Dim, func, scale)
    groups = DFS_main(Dim, Tuples)
    return groups


def DFS_main(Dim, Tuples):
    subcomponents = []
    for i in range(Dim):
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


def AIM_DSM(Dim, func, scale):
    Tuples = []
    aim_add, aim_multi, add_intensity, multi_intensity = AIM(Dim, func, scale)
    add_intensity = sorted(add_intensity)
    SI_add = add_intensity[int((len(add_intensity) - 1) * 0.99)]
    multi_intensity = sorted(multi_intensity)
    SI_multi = multi_intensity[int((len(multi_intensity) - 1) * 0.99)]

    for i in range(Dim):
        for j in range(i+1, Dim):
            if aim_add[i][j] > SI_add and aim_multi[i][j] > SI_multi:
                Tuples.append([i, j])
    return Tuples


def AIM(Dim, func, scale):
    add_intensity = []
    aim_add = np.zeros((Dim, Dim))
    base = np.zeros(Dim)
    for i in range(Dim):
        base[i] = scale[0]
    intercept = func(base)
    sP = singlePerturb(Dim, func, scale)
    for i in range(Dim):
        for j in range(i+1, Dim):
            delta1 = DG(Dim, i, j, func, intercept, scale, sP)
            aim_add[i][j], aim_add[j][i] = delta1, delta1
            add_intensity.append(delta1)
    return aim_add, add_intensity


def DG(Dim, e1, e2, func, intercept, scale, sP):
    base = np.zeros(Dim)
    for i in range(Dim):
        base[i] = scale[0]
    base[e1] = scale[1]
    base[e2] = scale[1]
    f1, f2 = sP[e1], sP[e2]
    # f3 = func(base) * (1 + np.random.normal(loc=0, scale=0.01, size=None))
    f3 = func(base)
    delta = abs(f3 + intercept - f1 - f2)
    return delta


def MDC(Dim, e1, e2, func, intercept, scale, sP):
    base = np.zeros(Dim)
    for i in range(Dim):
        base[i] = scale[0]
    base[e1] = scale[1]
    base[e2] = scale[1]
    f1, f2 = sP[e1], sP[e2]
    # f3 = func(base) * (1 + np.random.normal(loc=0, scale=0.01, size=None))
    f3 = func(base)
    if f1 > 0 and f2 > 0 and f3 > 0 and intercept > 0:
        delta = abs(np.log(f3) + np.log(intercept) - np.log(f1) - np.log(f2))
    else:
        delta = 10e5
    return delta


def singlePerturb(Dim, func, scale):
    sP = []
    for i in range(Dim):
        base = np.zeros(Dim)
        for j in range(Dim):
            base[j] = scale[0]
        base[i] = scale[1]
        # sP.append(func(base) * (1 + np.random.normal(loc=0, scale=0.01, size=None)))
        sP.append(func(base))
    return sP


def RG(Dim):
    order = np.random.permutation(list(range(Dim)))
    groups = []
    for i in range(10):
        back = min((i+1)*100, Dim)
        groups.append(list(order[i*100:back]))
    return groups


def CC(Dim):
    groups = []
    for i in range(Dim):
        groups.append([i])
    return groups


def DG_base(Dim):  # DDG, ERDG, DG2
    return [list(range(0, Dim))]



def DDG_noise(Dim, func, scale):
    DSM = np.zeros((Dim, Dim))
    temp = np.ones(Dim)
    for i in range(Dim):
        temp[i] = scale[0]
    intercept = func(temp) * (1 + np.random.normal(loc=0, scale=0.01, size=None))

    # GDDG
    for i in range(Dim):
        for j in range(i+1, Dim):
            p1 = np.zeros(Dim)
            for t in range(Dim):
                p1[t] = scale[0]

            p2 = deepcopy(p1)
            p3 = deepcopy(p1)

            p1[i] = scale[1]
            p2[j] = 0
            p3[i] = scale[1]
            p3[j] = 0

            f1 = func(p1) * (1 + np.random.normal(loc=0, scale=0.01, size=None))
            f2 = func(p2) * (1 + np.random.normal(loc=0, scale=0.01, size=None))
            f3 = func(p3) * (1 + np.random.normal(loc=0, scale=0.01, size=None))

            delta1 = abs((f3 - f2) - (f1 - intercept))
            delta2 = 10e5
            if f1 > 0 and f2 > 0 and f3 > 0 and intercept > 0:
                delta2 = abs(np.log(f3) + np.log(intercept) - np.log(f1) - np.log(f2))

            if delta1 > 10e-3 and delta2 > 10e-8:
                DSM[i][j] = 1
                DSM[j][i] = 1

    return DSM



def DG2_noise(Dim, func, scale):
    DSM = np.zeros((Dim, Dim))
    temp = np.ones(Dim)
    for i in range(Dim):
        temp[i] = scale[0]
    intercept = func(temp) * (1 + np.random.normal(loc=0, scale=0.01, size=None))

    for i in range(Dim):
        for j in range(i+1, Dim):
            p1 = np.zeros(Dim)
            for t in range(Dim):
                p1[t] = scale[0]

            p2 = deepcopy(p1)
            p3 = deepcopy(p1)

            p1[i] = scale[1]
            p2[j] = 0
            p3[i] = scale[1]
            p3[j] = 0

            f1 = func(p1) * (1 + np.random.normal(loc=0, scale=0.01, size=None))
            f2 = func(p2) * (1 + np.random.normal(loc=0, scale=0.01, size=None))
            f3 = func(p3) * (1 + np.random.normal(loc=0, scale=0.01, size=None))

            delta1 = abs((f3 - f2) - (f1 - intercept))

            if delta1 > 10e-3:
                DSM[i][j] = 1
                DSM[j][i] = 1

    return DSM


def DG2(Dim, func, scale):
    DSM = np.zeros((Dim, Dim))
    temp = np.ones(Dim)
    for i in range(Dim):
        temp[i] = scale[0]
    intercept = func(temp)

    for i in range(Dim):
        for j in range(i+1, Dim):
            p1 = deepcopy(temp)
            p2 = deepcopy(temp)
            p3 = deepcopy(temp)

            p1[i] = scale[1]
            p2[j] = 0
            p3[i] = scale[1]
            p3[j] = 0

            f1 = func(p1)
            f2 = func(p2)
            f3 = func(p3)
            delta1 = abs((f3 - f2) - (f1 - intercept))
            delta2 = 10e5
            if f1 > 0 and f2 > 0 and f3 > 0 and intercept > 0:
                delta2 = abs((np.log(f3) - np.log(f2)) - (np.log(f1) - np.log(intercept)))
            if delta1 > 0.001 and delta2 > 10e-5:
                DSM[i][j] = 1
                DSM[j][i] = 1
    return DSM
