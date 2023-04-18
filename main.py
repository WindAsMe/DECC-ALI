from cec2013lsgo.cec2013 import Benchmark
from os.path import abspath, dirname
from Grouping import AIM_dLI
from DE.DE import MDECC
import numpy as np


def save(path, data):
    with open(path, 'ab') as f:
        np.savetxt(f, data)


if __name__ == "__main__":
    bench = Benchmark()
    this_path = dirname(abspath(__file__))
    FEs = 3000000
    trial_run = 2
    Dims = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 905, 905, 1000]
    FEs_Proposal = [500501, 500501, 500501, 500501, 500501, 500501, 500501, 500501, 500501, 500501, 500501, 500501,
                    409966, 409966, 500501]
    NIND = 60
    """Optimization Comparison"""
    for func_num in range(1, 16):
        Dim = Dims[func_num-1]
        func = bench.get_function(func_num)
        info = bench.get_info(func_num)
        scale_range = [info["lower"], info["upper"]]
        Proposal_path = this_path + "/Data/Proposal/f" + str(func_num) + ".csv"
        Proposal_iter = int((FEs - FEs_Proposal[func_num]) / 60 / Dim)

        for i in range(trial_run):
            Proposal_groups = AIM_dLI(Dim, func, scale_range)
            ObjV_Proposal = MDECC(Dim, NIND, Proposal_iter, func, scale_range, Proposal_groups)
            save(Proposal_path, [ObjV_Proposal])





