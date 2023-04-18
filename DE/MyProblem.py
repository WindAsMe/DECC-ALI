import geatpy as ea
import numpy as np


class CC_Problem(ea.Problem):

    def __init__(self, group, benchmark, scale_range, context):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
        self.Dim = len(group)
        varTypes = [0] * self.Dim
        lb = [scale_range[0]] * self.Dim
        ub = [scale_range[1]] * self.Dim
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        self.benchmark = benchmark
        self.group = group
        self.context = context
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        temp_Phen = []
        for i in range(len(pop.Chrom)):
            temp_Phen.append(self.context)
        temp_Phen = np.array(temp_Phen, dtype='float64')
        for var in self.group:
            temp_Phen[:, var] = pop.Phen[:, self.group.index(var)]
        result = []
        for p in temp_Phen:
            result.append([self.benchmark(p) * (1 + np.random.normal(loc=0, scale=0.01, size=None))])
        pop.ObjV = np.array(result)


class myProblem(ea.Problem):
    def __init__(self, Dim, benchmark, scale_range):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
        self.Dim = Dim
        varTypes = [0] * self.Dim
        lb = [scale_range[0]] * self.Dim
        ub = [scale_range[1]] * self.Dim
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        self.benchmark = benchmark
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        result = []
        for p in pop:
            chrom = p.Chrom[0]
            result.append([self.benchmark(np.array(chrom, dtype="double")) * (1 + np.random.normal(loc=0, scale=0.01, size=None))])
        pop.ObjV = np.array(result)
