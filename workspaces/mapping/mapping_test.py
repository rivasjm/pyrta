from model import System
from gradient import *
from analysis import HolisticFPAnalysis, HolisticGlobalEDFAnalysis
from assignment import PDAssignment, HOPAssignment, repr_priorities,EQFAssignment, EQSAssignment
from vector import VectorHolisticFPBatchCosts
from evaluation import SchedRatioEval
from examples import get_system
from generator import set_utilization
from random import Random
from fast_analysis import FastHolisticFPAnalysis
from functools import partial


def gdpa_pd_fp_vector(system: System) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = PriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=100)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = VectorHolisticFPBatchCosts()
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam(lr=3, gamma=0.9)
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


def pd_fp(system: System) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    pd = PDAssignment(normalize=True)
    pd.apply(system)
    analysis.apply(system)
    return system.is_schedulable()


def gdpa_pd_fp_mapping(system: System) -> bool:
    test = HolisticFPAnalysis(limit_factor=1, reset=True)
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = MappingPriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=100)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = SequentialBatchCostFunction(cost_function=cost_function)
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam(lr=3, gamma=0)
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=True)

    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)

    test.apply(system)
    return system.is_schedulable()


if __name__ == '__main__':
    # create population of examples
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    systems = [get_system(size, rnd, balanced=False, name=str(i),
                          deadline_factor_min=0.5,
                          deadline_factor_max=1) for i in range(n)]

    system = systems[0]
    set_utilization(system, 0.7)

    gdpa_pd_fp_mapping(system)
    # gdpa_pd_fp_vector(system)
