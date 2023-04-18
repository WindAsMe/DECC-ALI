# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
import random



class soea_DE_rand_1_L_templet(ea.SoeaAlgorithm):
    """
soea_DE_rand_1_L_templet : class - 差分进化DE/rand/1/L算法类
算法描述:
    本算法类实现的是经典的DE/rand/1/L单目标差分进化算法。算法流程如下：
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 选择差分变异的基向量，对当前种群进行差分变异，得到变异个体。
    5) 将当前种群和变异个体合并，采用指数交叉方法得到试验种群。
    6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    7) 回到第2步。
参考文献:
    [1] Karol R. Opara and Jarosław Arabas. 2019. Differential Evolution: A
    survey of theoretical analyses. Swarm and Evolutionary Computation 44, June
    2017 (2019), 546–558. https://doi.org/10.1016/j.swevo.2018.06.010
"""

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, trappedValue, maxTrappedCount, dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'DE/rand/1/L'
        self.selFunc = 'rcs'  # 基向量的选择方式，采用随机补偿选择
        if population.Encoding == 'RI':
            self.mutOper = ea.Mutde(F=0.5)  # 生成差分变异算子对象
            self.recOper = ea.Xovexp(XOVR=0.5, Half_N=True)  # 生成指数交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为''RI''.')

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        # population.initChrom(NIND)  # 初始化种群染色体矩阵
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        self.call_aimFunc(population)  # 计算种群的目标函数值
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        # ===========================开始进化============================
        while not self.terminated(population):
            # 进行差分进化操作
            r0 = ea.selecting(self.selFunc, population.FitnV, NIND)  # 得到基向量索引
            experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
            experimentPop.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field, [r0])  # 变异
            experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))  # 重组
            self.call_aimFunc(experimentPop)  # 计算目标函数值
            tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
            population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果



class soea_MDE_DS_templet(ea.SoeaAlgorithm):
    """
算法描述:
    为了实现矩阵化计算，本算法类采用打乱个体顺序来代替随机选择差分向量。算法流程如下：
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 采用MDE_DS的方法选择差分变异的各个向量，对当前种群进行差分变异，得到变异个体。
    5) 将当前种群和变异个体合并，采用指数交叉方法得到试验种群。
    6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    7) 回到第2步。
参考文献:
    [1] Das, Swagatam & Suganthan, Ponnuthurai. (2011). Differential Evolution:
        A Survey of the State-of-the-Art.. IEEE Trans. Evolutionary Computation. 15. 4-31.
"""

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, trappedValue, maxTrappedCount, dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'DE/MDE_DS/1/L'

        if population.Encoding == 'RI':
            self.mutOper = ea.Mutde(F=0.5)  # 生成差分变异算子对象
            self.recOper = ea.Xovexp(XOVR=0.5, Half_N=True)  # 生成指数交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为''RI''.')

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        # population.initChrom(NIND)  # 初始化种群染色体矩阵
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        self.call_aimFunc(population)  # 计算种群的目标函数值
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        # ===========================开始进化============================
        while not self.terminated(population):
            F = np.random.uniform(0.5, 2)

            index = np.argsort(-np.array(population.ObjV[:, 0]))
            elitePop = population[index[0: int(NIND / 2)]]
            X_b = population[index[0]].Chrom[0]
            X_c = centering(elitePop.Chrom)
            experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
            scale_range = [population.Field[0][0], population.Field[1][0]]
            experimentPop.Chrom = Mutation(F, X_c, X_b, population.Chrom, scale_range)  # 变异
            experimentPop.Chrom = Crossover(population.Chrom, experimentPop.Chrom, scale_range)  # 重组
            self.call_aimFunc(experimentPop)  # 计算目标函数值
            population = Selection(population, experimentPop)

            # self.mutOper = ea.Mutde(F=np.random.uniform(0.5, 2))  # 生成差分变异算子对象
            # self.recOper = ea.Xovexp(XOVR=np.random.uniform(0.3, 1), Half_N=True)  # 生成指数交叉算子对象，这里的XOVR即为DE中的Cr
            # # 进行差分进化操作
            # r0 = np.arange(NIND)
            # r_best = ea.selecting('ecs', population.FitnV, NIND)  # 执行'ecs'精英复制选择
            #
            # experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
            # experimentPop.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field,
            #                                       [r0, None, None, r_best, r0])  # 变异
            # experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))  # 重组
            # self.call_aimFunc(experimentPop)  # 计算目标函数值
            # tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
            # tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
            # population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果


def Mutation(F, X_c, X_b, Chrom, scale_range):
    newChrom = []
    NIND = len(Chrom)
    for i in range(NIND):
        if np.random.rand() < 0.5:
            chrom1 = Chrom[np.random.randint(0, NIND)]
            chrom2 = Chrom[np.random.randint(0, NIND)]
            newChrom.append(centroid(F, X_c, chrom1, chrom2, scale_range))
        else:
            newChrom.append(DMP(Chrom[i], X_b, scale_range))
    return np.array(newChrom)


def Crossover(originChrom, newChrom, scale_range):
    bs = [0.1, 0.5, 0.9]
    NIND = len(newChrom)
    D = len(newChrom[0])
    for i in range(NIND):
        for j in range(D):
            if np.random.rand() <= random.uniform(0.3, 1):
                b = random.choice(bs)
                newChrom[i][j] = min(max(scale_range[0], b * originChrom[i][j] + (1 - b) * newChrom[i][j]), scale_range[1])
            else:
                newChrom[i][j] = originChrom[i][j]
    return np.array(newChrom)


def Selection(population, experimentPop):
    NIND = len(population.Chrom)
    Chrom = []
    ObjV = []
    newPop = ea.Population(population.Encoding, population.Field, NIND)
    for i in range(NIND):

        delta_f = abs(experimentPop.ObjV[i][0] - population.ObjV[i][0])
        dis = Dis(population.Chrom[i], experimentPop.Chrom[i])
        if population.ObjV[i] > experimentPop.ObjV[i] or np.random.rand() <= np.exp(-delta_f / dis):
            Chrom.append(experimentPop.Chrom[i])
            ObjV.append(experimentPop.ObjV[i])
        else:
            Chrom.append(population.Chrom[i])
            ObjV.append(population.ObjV[i])
    newPop.Chrom = np.array(Chrom)
    newPop.Phen = np.array(Chrom)
    newPop.ObjV = np.array(ObjV)
    return newPop


def Dis(C1, C2):
    dis = 0
    for i in range(len(C1)):
        dis += abs(C1[i] - C2[i])
    if dis == 0:
        dis = 10e-9
    return dis


def centroid(F, X_c, chrom1, chrom2, scale_range):
    new_chrom = []
    D = len(X_c)
    for i in range(D):
        value = min(max(chrom1[i] + F * (X_c[i] - chrom2[i]), scale_range[0]), scale_range[1])
        new_chrom.append(value)
    return new_chrom


def DMP(X_i, X_b, scale_range):
    new_chrom = []
    D = len(X_b)
    unit_vector = np.random.rand(D)
    unit_vector /= np.linalg.norm(unit_vector)

    delta_m = 0
    for i in range(D):
        delta_m += (X_b[i] - X_i[i])
    delta_m /= D
    for i in range(D):
        new_chrom.append(min(max(X_i[i] + delta_m * unit_vector[i], scale_range[0]), scale_range[1]))
    return new_chrom


def centering(Chrom):
    Chrom = np.array(Chrom)
    center = []
    for i in range(len(Chrom[0])):
        center.append(np.mean(Chrom[:, i]))
    return center