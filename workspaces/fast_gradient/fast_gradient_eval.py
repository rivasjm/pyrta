from model import System
from gradient import *
from analysis import HolisticFPAnalysis, HolisticGlobalEDFAnalysis
from assignment import PDAssignment, HOPAssignment, repr_priorities
from vector import VectorHolisticFPBatchCosts
from evaluation import SchedRatioEval
from examples import get_system
from random import Random
from fast_analysis import FastHolisticFPAnalysis
from functools import partial


def gdpa_fp_vector(system: System) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = PriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=100)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = VectorHolisticFPBatchCosts()
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


def gdpa_fp(system: System) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = PriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=100)
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
                                        verbose=False)
    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    analysis.apply(system)
    return system.is_schedulable()


def gdpa_fp_fast(system: System, limit_p=1, limit_i=1, ceiling=False, fast=True,
                 delta=1.5, lr=3, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=0.9) -> bool:

    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    fast_analysis = FastHolisticFPAnalysis(limit_factor=10, limit_p=limit_p, limit_i=limit_i, ceiling=ceiling, fast=fast)
    extractor = PriorityExtractor()
    fast_cost_function = InvslackCost(extractor=extractor, analysis=fast_analysis)
    real_cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = FixedIterationsStop(iterations=100)
    delta_function = AvgSeparationDelta(factor=delta)

    # gradient with fast cost function
    batch_cost_function = SequentialBatchCostFunction(cost_function=fast_cost_function)
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, gamma=gamma)
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=real_cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=False)
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


def hopa_fp(system: System) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    hopa = HOPAssignment(analysis=analysis)
    hopa.apply(system)
    analysis.apply(system)
    return system.is_schedulable()


if __name__ == '__main__':
    # create population of examples
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    systems = [get_system(size, rnd, balanced=True, name=str(i),
                          deadline_factor_min=0.5,
                          deadline_factor_max=1) for i in range(n)]

    # utilizations between 50 % and 90 %
    utilizations = np.linspace(0.5, 0.9, 20)

    # tools = [("gdpa", gdpa_fp_vector),
    #          ("gdpa_fast", gdpa_fp_fast),
    #          ("hopa", hopa_fp),
    #          ("pd", pd_fp)]

    fast1 = partial(gdpa_fp_fast, limit_p=1, limit_i=1, ceiling=False, fast=True,
            delta=1.5, lr=3, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=0.9)

    fast2 = partial(gdpa_fp_fast, limit_p=-1, limit_i=-1, ceiling=False, fast=True,
                    delta=1.5, lr=3, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=0.9)

    fast3 = partial(gdpa_fp_fast, limit_p=-1, limit_i=-1, ceiling=True, fast=False,
                    delta=1.5, lr=3, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=0.9)

    tools = [("gdpa_fast1", fast1),
             ("gdpa_fast2", fast2),
             ("gdpa_fast3", fast3)
             ]

    labels, funcs = zip(*tools)
    runner = SchedRatioEval("test", labels=labels, funcs=funcs,
                            systems=systems, utilizations=utilizations, threads=6)
    runner.run()
