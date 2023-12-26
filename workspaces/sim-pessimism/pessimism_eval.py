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


def edf_local_pd(system: System):
    item(system, HolisticLocalEDFAnalysis(limit_factor=10, reset=False), PDAssignment())


def edf_local_eqs(system: System):
    item(system, HolisticLocalEDFAnalysis(limit_factor=10, reset=False), EQSAssignment())


def edf_local_eqf(system: System):
    item(system, HolisticLocalEDFAnalysis(limit_factor=10, reset=False), EQFAssignment())


def edf_local_hopa(system: System):
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    item(system, analysis, HOPAssignment(analysis=analysis, verbose=True))


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
                                        verbose=True)

    PDAssignment().apply(system)
    return item(system, HolisticLocalEDFAnalysis(limit_factor=1), optimizer)


def run_tool(system, name, func, ax: plt.axes):
    print(name)
    func(system)    # apply optimization

    task = system.tasks[-2]
    wcrt = task.wcrt

    sim = simulator.Simulation(system, verbose=False)
    sim.run(until=1000000)
    rts = sim.results.task_rts(task)
    ax.hist(rts, 10)
    ax.axvline(x=wcrt, color='red', label='WCRT')
    ax.axvline(x=max(rts), color='blue', label='WORT')
    ax.set_title(name)

if __name__ == '__main__':
    # create population of examples
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    system = get_system(size, rnd, balanced=True, name="example",
                         deadline_factor_min=0.5, sched=SchedulerType.EDF,
                         deadline_factor_max=1, utilization=0.7)

    generator.to_int(system)
    for proc in system.processors:
        proc.local = True

    tools = [("EDF-L PD", edf_local_pd),
             ("EDF-L EQF", edf_local_eqf),
             ("EDF-L HOPA", edf_local_hopa),
             ("EDF-L GDPA", edf_local_gdpa)]

    fig, axs = plt.subplots(2, 2)

    for ax, (name, func) in zip(axs.flat, tools):
        run_tool(system, name, func, ax)

    plt.show()
