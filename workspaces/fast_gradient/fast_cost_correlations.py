import examples
import random
from analysis import HolisticFPAnalysis
from mast_tools import MastOffsetAnalysis
import matplotlib.pyplot as plt
from gradient import InvslackCost, PriorityExtractor
from assignment import random_priority_jump, PDAssignment
from analysis import reset_wcrt
from gradient import CostFunction
from fast_analysis import FastHolisticFPAnalysis
from fast_costs import SumSlopeCost, AvgSlopeCost, WorstFlowSlopeCost


def correlate(extractor, ref_cost: CostFunction, cost: CostFunction, ax, system):
    rnd = random.Random(1)
    pd = PDAssignment()
    pd.apply(system)

    xy = []
    print(cost)
    for i in range(1000):
        print(i)
        if i % 20 == 0:
            pd.apply(system)
        random_priority_jump(system, rnd)
        input = extractor.extract(system)

        reset_wcrt(system)
        x = ref_cost.apply(system, input)

        reset_wcrt(system)
        y = cost.apply(system, input)

        xy.append((x, y))

    x, y = zip(*xy)
    ax.set_title(cost)
    ax.scatter(x, y, alpha=0.5)


def study_corr():
    s = examples.get_small_system(random=random.Random(42), utilization=0.7, balanced=True)
    extractor = PriorityExtractor()
    holistic = HolisticFPAnalysis(limit_factor=10, reset=False)
    ref = InvslackCost(extractor=extractor, analysis=holistic)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))

    costs = [InvslackCost(extractor, analysis=FastHolisticFPAnalysis(limit_i=-1, limit_p=-1, ceiling=True, fast=True)),
             InvslackCost(extractor, analysis=FastHolisticFPAnalysis(limit_i=-1, limit_p=-1, ceiling=False, fast=True)),
             InvslackCost(extractor, analysis=FastHolisticFPAnalysis(limit_i=1, limit_p=1, ceiling=False, fast=False)),
             WorstFlowSlopeCost(extractor),
             AvgSlopeCost(extractor),
             SumSlopeCost(extractor)]

    for ax, cost in zip(axs.flat, costs):
        correlate(extractor, ref, cost, ax, s)

    plt.show()


if __name__ == '__main__':
    study_corr()