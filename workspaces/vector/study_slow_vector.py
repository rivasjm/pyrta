"""
Study why vectorized analysis is slower in gradient descent for priority+mapping optimization
Using the sequential batch gradient method is faster than the vectorized
"""

import pickle
import numpy as np
import vector, vector
import analysis
import gradient
import examples
import random
import assignment
from math import ceil


def run_sequential(system, inputs):
    holistic = analysis.HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = gradient.MappingPriorityExtractor()
    cost_function = gradient.InvslackCost(extractor=extractor, analysis=holistic)
    batch_cost_function = gradient.SequentialBatchCostFunction(cost_function=cost_function)

    res = batch_cost_function.apply(system, inputs)
    return res


def run_vectorized(system, inputs):
    batch_cost_function = vector.VectorHolisticFPBatchCosts(vector.MappingPrioritiesMatrix())
    res = batch_cost_function.apply(system, inputs)
    return res


def run_analyses():
    input = get_inputs()[0]
    system = get_example()
    analysis.init_wcrt(system)
    extractor = gradient.MappingPriorityExtractor()
    extractor.insert(system, input)
    initial_state = assignment.extract_assignment(system)

    # print(analysis.debug_repr(system))

    assignment.insert_assignment(system, initial_state)
    holistic = analysis.HolisticFPAnalysis2(reset=False, verbose=True)
    holistic.apply(system)
    r1 = [t.wcrt for t in system.tasks]

    assignment.insert_assignment(system, initial_state)
    vectorized = vector2.VectorHolisticFPAnalysis()
    vectorized.apply(system)
    r2 = [t.wcrt for t in system.tasks]

    print(r1)
    print(r2)


def run_gradients():
    inputs = get_inputs()
    system = get_example()
    initial_state = assignment.extract_assignment(system)

    assignment.insert_assignment(system, initial_state)
    run_sequential(system, inputs)

    assignment.insert_assignment(system, initial_state)
    run_vectorized(system, inputs)


def get_inputs():
    f = open("input.pkl", "rb")
    inputs = pickle.load(f)
    f.close()
    return inputs


def get_example():
    rnd = random.Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    systems = [examples.get_system(size, rnd, balanced=False, name=str(i),
                                   deadline_factor_min=0.5,
                                   deadline_factor_max=1) for i in range(n)]
    system = systems[0]

    input = get_inputs()[0]
    analysis.init_wcrt(system)
    extractor = gradient.MappingPriorityExtractor()
    extractor.insert(system, input)

    return system


def study():
    system = get_example()
    analysis.init_wcrt(system)
    t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = system.tasks
    print(analysis.debug_repr(system))

    p = 1
    w = p*t0.wcet

    w = (p*t0.wcet
         + ceil((t1.jitter+w)/t1.period)*t1.wcet
         + ceil((t2.jitter + w)/t2.period)*t2.wcet
         + ceil((t3.jitter+w)/t3.period)*t3.wcet
         + ceil((t5.jitter+w)/t5.period)*t5.wcet
         + ceil((t8.jitter+w)/t8.period)*t8.wcet
         + ceil((t11.jitter+w)/t11.period)*t11.wcet)
    print(f"w={w}")


if __name__ == '__main__':
    # study()
    run_analyses()