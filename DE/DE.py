import geatpy as ea
import numpy as np
from DE import MyProblem, templet


def init_Pop(Dim, scale_range, NIND):

    Field = ea.crtfld('RI', np.array([0] * Dim), np.array([[scale_range[0]] * Dim, [scale_range[1]] * Dim]),
                      np.array([[1] * Dim, [1] * Dim]))
    population = ea.Population('RI', Field, NIND)
    population.initChrom(NIND)
    return population.Chrom


def subChrom(Chrom, group):
    sub_chrom = []
    for i in range(len(Chrom)):
        sub_chrom.append([])
    for var in group:
        for i in range(len(Chrom)):
            sub_chrom[i].append(Chrom[i][var])
    return sub_chrom


def MDECC(Dim, NIND, MAX_iteration, func, scale_range, groups):
    context = np.zeros(Dim)
    popChrom = init_Pop(Dim, scale_range, NIND)
    real_iteration = 0
    Objs = []
    while real_iteration < MAX_iteration:
        for i in range(len(groups)):
            sub_chrom = subChrom(popChrom, groups[i])
            iteration = len(groups[i])
            solution = MDECC_Opt(func, scale_range, groups[i], context, sub_chrom, iteration)
            for var in groups[i]:
                popChrom[:, var] = solution['lastPop'].Chrom[:, groups[i].index(var)]
                context[var] = solution['Vars'][0][groups[i].index(var)]
        real_iteration += 1
        obj = []
        for var in popChrom:
            obj.append(func(var) * (1 + np.random.normal(loc=0, scale=0.01, size=None)))
        Objs.append(min(obj))
    return [min(Objs)]


def MDECC_Opt(benchmark, scale_range, group, context, Chrom, iteration):

    Field = ea.crtfld('RI', np.array([0] * len(group)),
                      np.array([[scale_range[0]] * len(group), [scale_range[1]] * len(group)]),
                      np.array([[1] * len(group), [1] * len(group)]))
    population = ea.Population('RI', Field, len(Chrom))
    population.Chrom = np.array(Chrom)
    population.Phen = np.array(Chrom)

    problem = MyProblem.CC_Problem(group, benchmark, scale_range, context)  # 实例化问题对象

    """===========================算法参数设置=========================="""

    myAlgorithm = templet.soea_MDE_DS_templet(problem, population)
    myAlgorithm.MAXGEN = iteration + 1
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    solution = ea.optimize(myAlgorithm, verbose=False, outputMsg=False, drawLog=False, saveFlag=False)
    return solution



def MDEDS(Dim, NIND, MAX_iter, func, scale_range):
    max_iter = Dim * MAX_iter
    solution = MDE_Opt(Dim, NIND, func, scale_range, max_iter)
    return solution['ObjV'][0]


def MDE_Opt(Dim, NIND, benchmark, scale_range, iteration):

    Field = ea.crtfld('RI', np.array([0] * Dim),
                      np.array([[scale_range[0]] * Dim, [scale_range[1]] * Dim]),
                      np.array([[1] * Dim, [1] * Dim]))
    population = ea.Population('RI', Field, NIND)
    population.initChrom(NIND)

    problem = MyProblem.myProblem(Dim, benchmark, scale_range)

    """===========================算法参数设置=========================="""

    myAlgorithm = templet.soea_MDE_DS_templet(problem, population)
    myAlgorithm.MAXGEN = iteration + 1
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    solution = ea.optimize(myAlgorithm, verbose=False, outputMsg=False, drawLog=False, saveFlag=False)
    return solution