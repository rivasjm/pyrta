import assignment
import examples
import generator
import gradient
from model import System, SchedulerType
from gradient import *
from analysis import *
from assignment import PDAssignment, HOPAssignment, repr_priorities, EQFAssignment, EQSAssignment, repr_deadlines_mini
from vector import VectorHolisticFPBatchCosts
from evaluation import SchedRatioEval
from examples import get_system
from random import Random
from fast_analysis import FastHolisticFPAnalysis
from functools import partial
import simulator


def item(system, analysis, assignment):
    assignment.apply(system)
    analysis.apply(system)
    return system.is_schedulable()


def edf_local_pd(system: System) -> bool:
    return item(system, HolisticLocalEDFAnalysis(limit_factor=10, reset=False), PDAssignment())


def edf_local_eqs(system: System) -> bool:
    return item(system, HolisticLocalEDFAnalysis(limit_factor=10, reset=False), EQSAssignment())


def edf_local_eqf(system: System) -> bool:
    return item(system, HolisticLocalEDFAnalysis(limit_factor=10, reset=False), EQFAssignment())


def edf_local_hopa(system: System) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    return item(system, analysis, HOPAssignment(analysis=analysis))


def edf_local_gdpa(system: System) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    extractor = DeadlineExtractor()
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

    PDAssignment().apply(system)
    schedulable = item(system, HolisticLocalEDFAnalysis(limit_factor=1, reset=True), optimizer)
    if schedulable and any(t.deadline < 0 for t in system.tasks):
        print(f"Negative deadlines found in {system.name}, u={system.utilization}, {assignment.repr_deadlines_mini(system)}")
    return schedulable


if __name__ == '__main__':
    # create population of examples
    rnd = Random(42)
    size = (8, 8, 5)  # flows, tasks, procs
    n = 50

    systems = examples.get_fast_systems(n, 1000, size, rnd, balanced=True, name="medium",
                                        deadline_factor_min=0.5, sched=SchedulerType.EDF,
                                        deadline_factor_max=1)

    # utilizations between 50 % and 90 %
    utilizations = np.linspace(0.5, 0.9, 20)

    tools = [("EDF-L PD", edf_local_pd),
             ("EDF-L EQS", edf_local_eqs),
             ("EDF-L EQF", edf_local_eqf),
             ("EDF-L HOPA", edf_local_hopa),
             ("EDF-L GDPA", edf_local_gdpa)]

    labels, funcs = zip(*tools)
    runner = SchedRatioEval("edf_local_big", labels=labels, funcs=funcs,
                            systems=systems, utilizations=utilizations, threads=6)
    runner.run()
