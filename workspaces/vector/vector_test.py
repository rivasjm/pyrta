from model import System
from gradient import *
from analysis import HolisticFPAnalysis, HolisticGlobalEDFAnalysis
import assignment
import vector
import vector
from examples import get_system
from random import Random
from fast_analysis import FastHolisticFPAnalysis
from functools import partial
import generator
import time


def pd_fp(system: System) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    pd = assignment.PDAssignment(normalize=True)
    pd.apply(system)
    analysis.apply(system)
    return system.is_schedulable()


def gdpa_pd_fp_vector(system: System) -> bool:
    test = HolisticFPAnalysis(limit_factor=1, reset=True)
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = MappingPriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = FixedIterationsStop(iterations=100)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = vector.VectorHolisticFPBatchCosts(vector.MappingPrioritiesMatrix())
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam()
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=True)

    pd = assignment.PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)

    test.apply(system)
    return system.is_schedulable()


def gdpa_pd_fp_vector_cached(system: System) -> bool:
    test = HolisticFPAnalysis(limit_factor=1, reset=True)
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = MappingPriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = FixedIterationsStop(iterations=100)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = vector2.VectorHolisticFPBatchCosts(vector2.MappingPrioritiesMatrix())
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam()
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=True)

    pd = assignment.PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)

    test.apply(system)
    return system.is_schedulable()


def gdpa_pd_fp_sequential(system: System) -> bool:
    test = HolisticFPAnalysis(limit_factor=1, reset=True)
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = MappingPriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = FixedIterationsStop(iterations=100)
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

    pd = assignment.PDAssignment(normalize=True)
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

    utilization = 0.8
    system = systems[0]
    generator.set_utilization(system, utilization)
    initial_state = assignment.extract_assignment(system)
    a = time.perf_counter()

    # PD
    print("pd ", end="")
    assignment.insert_assignment(system, initial_state)
    pd = pd_fp(system)
    b = time.perf_counter()
    print(f"{pd} {b-a}")

    # GDPA MAPPING (sequential)
    print("gdpa seq ", end="")
    assignment.insert_assignment(system, initial_state)
    gdpa_seq = gdpa_pd_fp_sequential(system)
    c = time.perf_counter()
    print(f"{gdpa_seq} {c-b}")

    # GDPA MAPPING (vector)
    print("gdpa vec ", end="")
    assignment.insert_assignment(system, initial_state)
    gdpa_vec = gdpa_pd_fp_vector(system)
    d = time.perf_counter()
    print(f"{gdpa_vec} {d-c}")

    # GDPA MAPPING (vector cached)
    print("gdpa vec cached ", end="")
    assignment.insert_assignment(system, initial_state)
    gdpa_vec = gdpa_pd_fp_vector_cached(system)
    e = time.perf_counter()
    print(f"{gdpa_vec} {e - d}")

