from model import System
from gradient import *
from analysis import HolisticFPAnalysis, HolisticGlobalEDFAnalysis
from assignment import PDAssignment, HOPAssignment, repr_priorities,EQFAssignment, EQSAssignment
from vector import VectorHolisticFPBatchCosts, MappingPrioritiesMatrix, PrioritiesMatrix
from evaluation import SchedRatioEval
from examples import get_system
from random import Random
from fast_analysis import FastHolisticFPAnalysis
from functools import partial


def gdpa(system: System) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = PriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=100)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = VectorHolisticFPBatchCosts(PrioritiesMatrix())
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam()
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=False)

    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    analysis.apply(system)
    return system.is_schedulable()


def gdpa_mapping(system: System, **kwargs) -> bool:
    i = kwargs["i"]
    d = kwargs["d"]
    lr = kwargs["lr"]
    b1 = kwargs["b1"]
    b2 = kwargs["b2"]
    e = kwargs["e"]
    g = kwargs["g"]

    test = HolisticFPAnalysis(limit_factor=1, reset=True)
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = MappingPriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=i)
    delta_function = AvgSeparationDelta(factor=d)
    batch_cost_function = VectorHolisticFPBatchCosts(MappingPrioritiesMatrix())
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam(lr=lr, beta1=b1, beta2=b2, epsilon=e, gamma=g)
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=False)

    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    test.apply(system)
    return system.is_schedulable()


def build_func(i, d, lr, b1, b2, e, g):
    label = f"i={i} d={d} lr={lr} b1={b1} b2={b2}, e={e}, g={g}"
    func = partial(gdpa_mapping, i=i, d=d, lr=lr, b1=b1, b2=b2, e=e, g=g)
    return label, func


if __name__ == '__main__':
    # create population of examples
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    systems = [get_system(size, rnd, balanced=False, name=str(i),
                          deadline_factor_min=0.5,
                          deadline_factor_max=1) for i in range(n)]

    utilizations = [0.75]

    tools = [
        ("gdpa", gdpa),
        build_func(i=200, d=1.5, lr=1.5, b1=0.9, b2=0.999, e=0.1, g=0.9),
        build_func(i=200, d=1.5, lr=1, b1=0.9, b2=0.999, e=0.1, g=0.9),
        build_func(i=200, d=1.5, lr=2, b1=0.9, b2=0.999, e=0.1, g=0.9),
        build_func(i=200, d=1.5, lr=1.5, b1=0.9, b2=0.999, e=0.1, g=0.5),
        build_func(i=200, d=1.5, lr=1.5, b1=0.9, b2=0.999, e=0.1, g=1.5),
        build_func(i=200, d=1, lr=1.5, b1=0.9, b2=0.999, e=0.1, g=0.9),
        build_func(i=200, d=2, lr=1.5, b1=0.9, b2=0.999, e=0.1, g=0.9),
        build_func(i=200, d=2, lr=1, b1=0.9, b2=0.999, e=0.1, g=0.5)
    ]

    # best: d=1.5 lr=1.5 b1=0.9 b2=0.999 e=0.1 g=0.5

    labels, funcs = zip(*tools)
    runner = SchedRatioEval("mapping-params", labels=labels, funcs=funcs,
                            systems=systems, utilizations=utilizations, threads=6)
    runner.run()
