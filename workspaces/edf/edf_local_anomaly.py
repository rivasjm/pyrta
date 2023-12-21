import assignment
import examples
import generator
import gradient
from model import System, SchedulerType
from gradient import *
from analysis import *
from assignment import PDAssignment, repr_deadlines_mini
import pickle
from examples import get_system
from random import Random

from mast_tools import MastHolisticAnalysis


def item(system, analysis, assignment):
    assignment.apply(system)
    analysis.apply(system)
    return system.is_schedulable()

# there seems to be EDF-L systems deemed schedulable by the Python but not by MAST

def get_systems():
    # create population of examples
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    systems = [get_system(size, rnd, utilization=0.7, balanced=True, name=str(i), sched=SchedulerType.EDF,
                          deadline_factor_min=0.5, deadline_factor_max=1) for i in range(n)]
    return systems


def edf_local_gdpa(system: System):
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    extractor = DeadlineExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = gradient.FixedIterationsStop(iterations=100)
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

    PDAssignment().apply(system)
    optimizer.apply(system)


def dump_deadlines(system):
    d = [t.deadline for t in system.tasks]
    with open('anomaly.pkl', 'wb') as f:
        pickle.dump(d, f)


def find_anomaly_system():
    systems = get_systems()
    test_python = HolisticLocalEDFAnalysis(limit_factor=1)
    test_mast = MastHolisticAnalysis(limit_factor=1, local=True)

    for system in systems:
        print(".")
        edf_local_gdpa(system)

        reset_wcrt(system)
        test_python.apply(system)
        sched_python = system.is_schedulable()

        reset_wcrt(system)
        test_mast.apply(system)
        sched_mast = system.is_schedulable()

        if sched_mast != sched_python:
            print("found!")
            print(assignment.repr_deadlines_mini(system))
            dump_deadlines(system)
            break


def get_anomaly_system():
    system = get_systems()[0]
    with open('anomaly.pkl', 'rb') as f:
        sds = pickle.load(f)
        for d, t in zip(sds, system.tasks):
            t.deadline = d
    return system


def study_anomaly():
    system = get_anomaly_system()
    test_python = HolisticLocalEDFAnalysis(limit_factor=1, verbose=True)
    test_mast = MastHolisticAnalysis(limit_factor=100, local=True)

    print(assignment.repr_deadlines(system))

    reset_wcrt(system)
    test_python.apply(system)
    sched_python = system.is_schedulable()
    print(repr_wcrts(system))

    reset_wcrt(system)
    test_mast.apply(system)
    sched_mast = system.is_schedulable()
    print(repr_wcrts(system))


def study_palencia():
    system = examples.get_palencia_system()
    generator.set_utilization(system, utilization=0.2)
    generator.to_edf(system)
    edf_local_gdpa(system)

    test_python = HolisticLocalEDFAnalysis(limit_factor=1, verbose=True)
    test_mast = MastHolisticAnalysis(limit_factor=100, local=True)

    print(assignment.repr_deadlines(system))

    reset_wcrt(system)
    test_python.apply(system)
    sched_python = system.is_schedulable()
    print(repr_wcrts(system))

    reset_wcrt(system)
    test_mast.apply(system)
    sched_mast = system.is_schedulable()
    print(repr_wcrts(system))

    print(sched_python, sched_mast)


if __name__ == '__main__':
    study_palencia()