from model import System
from gradient import *
from analysis import HolisticFPAnalysis, HolisticGlobalEDFAnalysis
from assignment import PDAssignment, HOPAssignment, repr_priorities
from vector import VectorHolisticFPBatchCosts
from evaluation import SchedRatioEval
from examples import get_system
from random import Random


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
                          deadline_factor_max=1) for i in range(50)]

    # utilizations between 50 % and 90 %
    utilizations = np.linspace(0.5, 0.9, 20)

    tools = [("gdpa", gdpa_fp_vector),
             ("hopa", hopa_fp),
             ("pd", pd_fp)]
    labels, funcs = zip(*tools)
    runner = SchedRatioEval("test", labels=labels, funcs=funcs,
                          systems=systems, utilizations=utilizations, threads=8)
    runner.run()