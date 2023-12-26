import generator
import simulator
from model import System, SchedulerType
from gradient import *
from analysis import *
from assignment import PDAssignment, HOPAssignment, repr_priorities, EQFAssignment, EQSAssignment, repr_deadlines_mini
from examples import get_system
from random import Random
from mast_tools import MastHolisticAnalysis
import matplotlib.pyplot as plt


def item(system, analysis, assignment):
    assignment.apply(system)
    analysis.apply(system)


def edf_local_gdpa(system: System):
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
    return item(system, HolisticLocalEDFAnalysis(limit_factor=1), optimizer)


def run_tool(system, func):
    print("GDPA")
    func(system)    # apply optimization
    wcrts = [t.wcrt for t in system.tasks]
    print([t.deadline for t in system.tasks])

    print("Simulator")
    sim = simulator.Simulation(system, verbose=False)
    sim.run(until=50000000)
    worts = [sim.results.task_wort(t) for t in system.tasks]

    print(sim.results.pessimism())
    for analyzed, measured, task in zip(wcrts, worts, system.tasks):
        print(f"{task.name}: {measured} {analyzed}")
        if analyzed < measured:
            print(f"anomaly detected in task {task.name}")


if __name__ == '__main__':
    # create population of examples
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    system = get_system(size, rnd, balanced=True, name="example",
                         deadline_factor_min=0.8, sched=SchedulerType.EDF,
                         deadline_factor_max=1, utilization=0.7)

    generator.to_int(system)
    for proc in system.processors:
        proc.local = True

    run_tool(system, edf_local_gdpa)
