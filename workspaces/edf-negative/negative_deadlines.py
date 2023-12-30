import generator
import simulator
from model import System, SchedulerType
from gradient import *
from analysis import *
from assignment import PDAssignment, repr_deadlines_mini
from examples import get_system
from random import Random
import matplotlib.pyplot as plt


extractor = DeadlineExtractor()
xy = []

def callback(t, S, x, xb, cost, best, ref_cost):
    print(repr_deadlines_mini(S))
    sim = simulator.Simulation(system=S, verbose=False)
    sim.run(until=1000)
    sim_cost = max([(sim.results.flow_wort(flow)-flow.deadline)/flow.deadline for flow in system.flows])
    print(cost, sim_cost)
    xy.append((cost, sim_cost))
    fig, ax = plt.subplots(figsize=(10, 8))
    x, y = zip(*xy)
    ax.scatter(x, y, alpha=0.5)
    plt.show()

def item(system, analysis, assignment):
    assignment.apply(system)
    analysis.apply(system)
    return system.is_schedulable()


def edf_local_gdpa(system: System) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
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
                                        callback=callback,
                                        verbose=True)

    PDAssignment().apply(system)
    return item(system, HolisticLocalEDFAnalysis(limit_factor=1), optimizer)


if __name__ == '__main__':
    # create population of examples
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    systems = [get_system(size, rnd, balanced=True, name=str(i),
                          deadline_factor_min=0.5, sched=SchedulerType.EDF,
                          deadline_factor_max=1) for i in range(n)]

    system = systems[4]

    generator.set_utilization(system, 0.8)
    generator.to_int(system)

    edf_local_gdpa(system)
