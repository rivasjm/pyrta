from random import Random
from generator import generate_system, set_utilization
from model import *
import numpy as np


def get_palencia_system() -> System:
    system = System()

    # 2 cpus + 1 network
    cpu1 = Processor(name="cpu1")
    cpu2 = Processor(name="cpu2")
    network = Processor(name="network")
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


def get_system(size, random=Random(), utilization=0.5, balanced=False, name=None,
               deadline_factor_min=0.5, deadline_factor_max=1) -> System:
    n_flows, t_tasks, n_procs = size
    system = generate_system(random,
                             n_flows=n_flows,
                             n_tasks=t_tasks,
                             n_procs=n_procs,
                             utilization=utilization,
                             period_min=100,
                             period_max=100*3,
                             deadline_factor_min=deadline_factor_min,
                             deadline_factor_max=deadline_factor_max,
                             balanced=balanced)
    system.name = name
    return system


def get_barely_schedulable() -> System:
    random = Random(123)
    n_flows, t_tasks, n_procs = (4, 5, 3)
    return get_system((n_flows, t_tasks, n_procs), random, 0.84, name="barely")


def get_small_system(random=Random(), utilization=0.5, balanced=False) -> System:
    n_flows, t_tasks, n_procs = (3, 4, 3)
    return get_system((n_flows, t_tasks, n_procs), random, utilization, balanced, name="small")


def get_medium_system(random=Random(), utilization=0.84, balanced=False) -> System:
    n_flows, t_tasks, n_procs = (4, 5, 3)
    return get_system((n_flows, t_tasks, n_procs), random, utilization, balanced, name="medium")


def get_big_system(random=Random(), utilization=0.84, balanced=False) -> System:
    n_flows, t_tasks, n_procs = (8, 8, 5)
    return get_system((n_flows, t_tasks, n_procs), random, utilization, balanced, name="big")


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
