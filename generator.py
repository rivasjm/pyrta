import math
from random import Random
from math import pow, log, exp
from model import *


def uunifast(random: Random, n_tasks: int, utilization: float) -> [float]:
    sum_u = utilization
    us = []

    for i in range(1, n_tasks):
        next_sum_u = sum_u * pow(random.random(), 1 / (n_tasks - i))
        us.append(sum_u - next_sum_u)
        sum_u = next_sum_u

    us.append(sum_u)
    return us


def log_uniform(random: Random, lowest: float, highest: float) -> float:
    r = random.uniform(log(lowest), log(highest))
    return exp(r)


def set_processor_utilization(processor: Processor, utilization: float):
    factor = utilization/processor.utilization
    for task in processor.tasks:
        task.wcet *= factor


def set_utilization(system: System, utilization: float):
    for proc in system.processors:
        set_processor_utilization(proc, utilization)


def generate_system(random: Random, n_flows, n_tasks, n_procs, utilization,
                    period_min, period_max, deadline_factor_min, deadline_factor_max,
                    balanced=False) -> System:

    system = System()
    procs = [Processor(name=f"proc{i}") for i in range(n_procs)]
    system.add_procs(*procs)

    # set the general structure
    for f in range(n_flows):
        period = log_uniform(random, period_min, period_max)
        deadline = random.uniform(
            period * n_tasks * deadline_factor_min,
            period * n_tasks * deadline_factor_max)
        flow = Flow(name=f"flow{f}", period=period, deadline=deadline)

        # for now leave the WCET empty
        tasks = [Task(name=f"task{f}_{t}", wcet=0, processor=random.choice(procs)) for t in range(n_tasks)]
        flow.add_tasks(*tasks)
        system.add_flows(flow)

    # if balanced=True, balance the number of tasks per processor (ignore current mapping)
    if balanced:
        # r = Random(len(system.tasks))
        tasks = system.tasks
        random.shuffle(tasks)
        for i, task in enumerate(tasks):
            task.processor = procs[i % len(procs)]

    # set the WCET's
    for proc in procs:
        tasks = proc.tasks
        if tasks:
            us = uunifast(random, len(tasks), utilization)
            for task, u in zip(tasks, us):
                task.wcet = u * task.period

    return system


def copy(system: System):
    new_procs = {proc.name: Processor(name=proc.name) for proc in system.processors}
    new_system = System()

    for flow in system:
        new_flow = Flow(name=flow.name, period=flow.period, deadline=flow.deadline)

        for task in flow:
            new_task = task.copy()
            new_task.processor = new_procs[task.processor.name]
            new_flow.add_tasks(new_task)
        new_system.add_flows(new_flow)

    return new_system


def create_series(template: System, utilizations) -> [System]:
    systems = []
    for utilization in utilizations:
        system = copy(template)
        for proc in system.processors:
            set_processor_utilization(proc, utilization)
        systems.append(system)
    return systems


def walk_series(system: System, utilizations, callback) -> None:
    save_tasks_params(system)
    for utilization in utilizations:
        for proc in system.processors:
            set_processor_utilization(proc, utilization)
        if callback:
            callback(system)
    restore_tasks_params(system)
