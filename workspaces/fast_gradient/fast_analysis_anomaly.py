from model import System
from gradient import *
from analysis import HolisticFPAnalysis, HolisticGlobalEDFAnalysis, reset_wcrt, repr_wcrts
from assignment import PDAssignment, HOPAssignment, repr_priorities
from vector import VectorHolisticFPBatchCosts
from evaluation import SchedRatioEval
from examples import get_system
from random import Random
from fast_analysis import FastHolisticFPAnalysis
from generator import set_utilization
from assignment import repr_priorities


def get_anomaly_system():
    # create population of examples
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    systems = [get_system(size, rnd, balanced=True, name=str(i),
                          deadline_factor_min=0.5,
                          deadline_factor_max=1) for i in range(n)]
    return systems[0]


def gdpa_fp_fast(system: System) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    fast_analysis = FastHolisticFPAnalysis(limit_factor=10, limit_p=-1, limit_i=-1, ceiling=True, fast=False)
    extractor = PriorityExtractor()
    fast_cost_function = InvslackCost(extractor=extractor, analysis=fast_analysis)
    real_cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=1)
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
    analysis.apply(system)


def gdpa_fp(system: System) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = PriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=1)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = SequentialBatchCostFunction(cost_function=cost_function)
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam()
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=True)
    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    analysis.apply(system)
    return system.is_schedulable()


def study():
    """
    It looks like when using the fast holistic (that should be the same as holistic), at some point the gradients
    are different.
    It happens at iteration 3:
    fast: iteration=3: cost=9.029 best=0.840 ref=9.029
    real: iteration=3: cost=2.728 best=0.840
    """
    system = get_anomaly_system()
    set_utilization(system, 0.8)
    print("REAL")
    reset_wcrt(system)
    gdpa_fp(system)
    print("FAST")
    reset_wcrt(system)
    gdpa_fp_fast(system)


def study_one():
    system = get_anomaly_system()
    set_utilization(system, 0.8)
    prio = [0.5, 0.05681818181818182, 0.25, 0.75, 0.75, 0.25, 0.25, 1.0, 1.0, 0.5, 1.0, 0.75]

    for p, t in zip(prio, system.tasks):
        t.priority = p

    holistic = HolisticFPAnalysis(limit_factor=10, reset=False)
    fast_holistic = FastHolisticFPAnalysis(limit_factor=10, limit_p=-1, limit_i=-1, ceiling=True, fast=False)

    reset_wcrt(system)
    holistic.apply(system)
    print(repr_wcrts(system))

    reset_wcrt(system)
    fast_holistic.apply(system)
    print(repr_wcrts(system))


if __name__ == '__main__':
    study_one()