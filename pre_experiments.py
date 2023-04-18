import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def DG(Dim, e1, e2, func, intercept, scale, sP):
    base = np.zeros(Dim)
    for i in range(Dim):
        base[i] = scale[0]
    base[e1] = scale[1]
    base[e2] = scale[1]
    f1, f2 = sP[e1], sP[e2]
    f3 = func(base) * (1 + np.random.normal(loc=0, scale=0.1, size=None))
    delta = abs(f3 + intercept - f1 - f2)
    return delta


def singlePerturb(Dim, func, scale):
    sP = []
    for i in range(Dim):
        base = np.zeros(Dim)
        for j in range(Dim):
            base[j] = scale[0]
        base[i] = scale[1]
        sP.append(func(base) * (1 + np.random.normal(loc=0, scale=0.1, size=None)))
    return sP


def AIM(Dim, func, scale):
    add_intensity = []
    aim_add = np.zeros((Dim, Dim))
    base = np.zeros(Dim)
    for i in range(Dim):
        base[i] = scale[0]
    intercept = func(base) * (1 + np.random.normal(loc=0, scale=0.1, size=None))
    sP = singlePerturb(Dim, func, scale)
    for i in range(Dim):
        for j in range(i+1, Dim):
            delta1 = DG(Dim, i, j, func, intercept, scale, sP)
            aim_add[i][j], aim_add[j][i] = delta1, delta1
            add_intensity.append(delta1)
    return aim_add, add_intensity


def PS(X):
    return (X[0] + X[1]) ** 2 + X[2] + (X[3] ** 2 + 1) * (X[4] ** 2 + 1)

Dim = 5
scale = [-1, 2]
aim_add, add_intensity = AIM(Dim, PS, scale)

intensity = add_intensity
print(aim_add)
y = []
for i in range(len(intensity)):
    y.append(np.random.uniform(-1, 1))

sep_x = []
sep_y = []

nonsep_x = []
nonsep_y = []

for i in range(len(intensity)):
    if intensity[i] > 8:
        nonsep_x.append(intensity[i])
        nonsep_y.append(y[i])
    else:
        sep_x.append(intensity[i])
        sep_y.append(y[i])


plt.scatter(sep_x, sep_y, c="b")
plt.scatter(nonsep_x, nonsep_y, c="r")
plt.savefig('add_scatter.png', dpi=750)
plt.show()
#
#
# intensity = []
# for i in range(Dim):
#     for j in range(i+1, Dim):
#         intensity.append(aim_multi[i][j])
# y = []
# for i in range(len(intensity)):
#     y.append(np.random.uniform(-1, 1))
#
# sep_x = []
# sep_y = []
#
# nonsep_x = []
# nonsep_y = []
#
# for i in range(len(intensity)):
#     if intensity[i] > 0.5 or 0.19 > intensity[i] > 0.15:
#         nonsep_x.append(intensity[i])
#         nonsep_y.append(y[i])
#     else:
#         sep_x.append(intensity[i])
#         sep_y.append(y[i])
#
# plt.scatter(sep_x, sep_y, c="b")
# plt.scatter(nonsep_x, nonsep_y, c="r")
# plt.savefig('multi_scatter.png', dpi=750)
# plt.show()
#
labels = ["$x_0$", "$x_1$", "$x_2$", "$x_3$", "$x_4$"]

sns.heatmap(aim_add, cmap="Reds", xticklabels=labels, yticklabels=labels)
plt.savefig('add.png', dpi=750)
plt.show()

