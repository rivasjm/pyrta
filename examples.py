from random import Random

import generator
import simulator
from generator import generate_system, set_utilization
from model import *
import numpy as np


def get_palencia_system() -> System:
    system = System()

    # 2 cpus + 1 network
    cpu1 = Processor(name="cpu1", sched=SchedulerType.FP)
    cpu2 = Processor(name="cpu2", sched=SchedulerType.FP)
    network = Processor(name="network", sched=SchedulerType.FP)
    system.add_procs(cpu1, cpu2, network)

    # priority levels
    HIGH = 10
    LOW = 1

    # 2 flows
    flow1 = Flow(name="flow1", period=30, deadline=60)
    flow2 = Flow(name="flow2", period=40, deadline=80)

    # tasks
    flow1.add_tasks(
        Task(name="a1", wcet=5, priority=HIGH, processor=cpu1),
        Task(name="a2", wcet=2, priority=LOW, processor=network),
        Task(name="a3", wcet=20, priority=LOW, processor=cpu2)
    )
    flow2.add_tasks(
        Task(name="a4", wcet=5, priority=HIGH, processor=cpu2),
        Task(name="a5", wcet=10, priority=HIGH, processor=network),
        Task(name="a6", wcet=10, priority=LOW, processor=cpu1)
    )
    system.add_flows(flow1, flow2)
    system.name = "palencia"
    return system


def get_three_tasks() -> System:
    system = System()

    # 1 cpu
    cpu = Processor(name="cpu", sched=SchedulerType.FP)
    system.add_procs(cpu)

    # priority levels
    HIGH = 10
    MEDIUM = 5
    LOW = 1

    # 1 flows
    flow = Flow(name="flow", period=30, deadline=90)

    # tasks
    flow.add_tasks(
        Task(name="a1", wcet=2, priority=HIGH, processor=cpu),
        Task(name="a2", wcet=5, priority=MEDIUM, processor=cpu),
        Task(name="a3", wcet=20, priority=LOW, processor=cpu)
    )

    system.add_flows(flow)
    system.name = "three_tasks"
    return system


def get_system(size, random=Random(), utilization=0.5, balanced=False, name=None,
               deadline_factor_min=0.5, deadline_factor_max=1, sched: SchedulerType = SchedulerType.FP) -> System:
    n_flows, t_tasks, n_procs = size
    system = generate_system(random,
                             n_flows=n_flows,
                             n_tasks=t_tasks,
                             n_procs=n_procs,
                             utilization=utilization,
                             sched=sched,
                             period_min=100,
                             period_max=100*3,
                             deadline_factor_min=deadline_factor_min,
                             deadline_factor_max=deadline_factor_max,
                             balanced=balanced)
    system.name = name
    return system


def get_fast_systems(number, population, size, random=Random(), utilization=0.5, balanced=False, name="system",
                     deadline_factor_min=0.5, deadline_factor_max=1, sched: SchedulerType = SchedulerType.FP):
    assert number <= population
    # generate population
    systems = [get_system(size, random, utilization, balanced, f"{name}{i}",
                          deadline_factor_min, deadline_factor_max, sched) for i in range(population)]

    # to integer values
    systems = list(map(generator.to_int, systems))

    # sort by hyperperiod
    systems.sort(key=simulator.hyperperiod)

    # get "number" systems with shortest hyperperiod
    return systems[:number]


def get_barely_schedulable() -> System:
    random = Random(123)
    n_flows, t_tasks, n_procs = (4, 5, 3)
    return get_system((n_flows, t_tasks, n_procs), random, 0.84, name="barely")


def get_small_system(random=Random(), utilization=0.5, balanced=False, sched=SchedulerType.FP) -> System:
    n_flows, t_tasks, n_procs = (3, 4, 3)
    return get_system((n_flows, t_tasks, n_procs), random, utilization, balanced, name="small", sched=sched)


def get_medium_system(random=Random(), utilization=0.84, balanced=False, sched=SchedulerType.FP) -> System:
    n_flows, t_tasks, n_procs = (4, 5, 3)
    return get_system((n_flows, t_tasks, n_procs), random, utilization, balanced, name="medium", sched=sched)


def get_big_system(random=Random(), utilization=0.84, balanced=False, sched=SchedulerType.FP) -> System:
    n_flows, t_tasks, n_procs = (8, 8, 5)
    return get_system((n_flows, t_tasks, n_procs), random, utilization, balanced, name="big", sched=sched)


def generate_anomaly_system() -> System:
    """
    Anomaly trace: 8-th utilization, system 9 and 23
    2105-bf-hol-2 0.67(8) 9 	-> hopa gdpa-r gdpa-pd
    2105-bf-hol-2 0.67(8) 23 	-> hopa gdpa-pd
    :return:
    """
    size = (2, 10, 5)  # flows, tasks/flow, processors
    population = 50
    utilization_min = 0.5
    utilization_max = 0.9
    utilization_steps = 20

    random = Random(42)
    utilizations = np.linspace(utilization_min, utilization_max, utilization_steps)
    systems = [get_system(size, random, balanced=True, name=str(i)) for i in range(population)]

    utilization = utilizations[8]  # 0.67 utilization
    system = systems[9]  #
    set_utilization(system, utilization)
    return system


def get_simple_gpu() -> System:
    system = System()

    # 1 cpu
    cpu1 = Processor(name="cpu1", sched=SchedulerType.FP)
    system.add_procs(cpu1)

    # 2 flows
    una = Flow(name="unavailable", period=60, deadline=60)
    gen = Flow(name="genetico", period=500, deadline=500)

    # tasks
    una.add_tasks(
        Task(name="a1_1", type=TaskType.Offset, wcet=10, bcet=10),
        Task(name="a1_2", wcet=50, bcet=50, priority=254, processor=cpu1))

    gen.add_tasks(
        Task(name="a2_1", wcet=5, bcet=5, priority=100, processor=cpu1),
        Task(name="a2_2", type=TaskType.Delay, wcet=50, bcet=30),
        Task(name="a2_3", wcet=8, bcet=8, priority=99, processor=cpu1),
        Task(name="a2_4", wcet=3, bcet=3, priority=96, processor=cpu1)
    )
    system.add_flows(una, gen)
    system.name = "genetico-gpu"
    return system


def get_validation_example():
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    system = [get_system(size, rnd, balanced=True, name=str(i),
                          deadline_factor_min=0.5,
                          deadline_factor_max=1) for i in range(n)][38]

    utilization = np.linspace(0.5, 0.9, 20)[13]
    set_utilization(system, utilization)
    return system