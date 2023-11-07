from random import Random
import matplotlib.pyplot as plt
from functools import partial
from model import System
from analysis import HolisticFPAnalysis
from assignment import PDAssignment
from examples import get_small_system
from fast_analysis import FastHolisticFPAnalysis
from gradient import PriorityExtractor, InvslackCost, FixedIterationsStop, AvgSeparationDelta, SequentialBatchCostFunction, \
    StandardGradient, NoisyAdam, StandardGradientDescent, StandardStop
from examples import get_system
from generator import set_utilization


def gdpa_fp_fast_comparison():
    system = get_system((3, 4, 3), Random(42), balanced=True, name=str(1),
                        deadline_factor_min=0.5, deadline_factor_max=1)
    set_utilization(system, 0.8)

    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    fast_analysis = FastHolisticFPAnalysis(limit_factor=10, limit_p=-1, limit_i=-1, ceiling=True, fast=False)
    extractor = PriorityExtractor()
    fast_cost_function = InvslackCost(extractor=extractor, analysis=fast_analysis)
    real_cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=100)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = SequentialBatchCostFunction(cost_function=fast_cost_function)
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam()
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=real_cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        ref_cost_function=real_cost_function,
                                        verbose=True)
    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)



def gdpa_fp_test():
    fig, ax = plt.subplots()
    ax.set_xlabel("reference cost")
    ax.set_ylabel("opt cost")
    callback_func = partial(corr_chart, ax)

    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    fast_analysis = FastHolisticFPAnalysis(limit_factor=10, limit_p=-1, limit_i=-1, ceiling=True, fast=False)
    extractor = PriorityExtractor()
    fast_cost_function = InvslackCost(extractor=extractor, analysis=fast_analysis)
    real_cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = FixedIterationsStop(iterations=100)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = SequentialBatchCostFunction(cost_function=fast_cost_function)
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam()
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=fast_cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        ref_cost_function=real_cost_function,
                                        callback=callback_func,
                                        verbose=True)

    r = Random(42)
    pd = PDAssignment(normalize=True)
    system = get_small_system(random=r, utilization=0.7, balanced=True)
    pd.apply(system)

    x = optimizer.apply(system)
    extractor.insert(system, x)
    analysis.apply(system)
    print(system.is_schedulable())

    # ax.set_ylim((-0.3, -0.25))
    # ax.set_xlim((-0.1, 0.1))
    plt.show()



def corr_chart(ax, t, S, x, xb, cost, best, ref_cost):
    ax.scatter(ref_cost, cost, alpha=0.5)


if __name__ == '__main__':
    gdpa_fp_fast_comparison()