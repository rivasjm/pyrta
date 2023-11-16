from model import Task, Processor, Flow, System, TaskType, SchedulerType
import numpy as np
import pandas as pd
import itertools
import mast_tools, mast_meta
import matplotlib.pyplot as plt


def run():


    periods = np.linspace(0.6, 60, 20)
    iterations = np.arange(1, 11, 1)
    df = pd.DataFrame()
    df["Period"] = periods
    for i in iterations:
        name = f"{i} iterations"
        res = [item(p,i) for p in periods]
        df[name] = res
    #
    # for p, i in itertools.product(periods, iterations):
    #     s = System()
    #     cpu = Processor(name="cpu", sched=SchedulerType.FP)
    #     s.add_procs(cpu)
    #     flow_un = mast_meta.unavailable_flow(period=p, window=p/6, cpu=cpu)
    #     flow_gen = genetic_flow(i, cpu)
    #     s.add_flows(flow_un, flow_gen)
    #
    #     analysis.apply(s)
    #     results.append((p, i, flow_gen.wcrt))
    #
    # df = pd.DataFrame(results, columns=['Period', 'Iterations', 'WCRT'])
    print(df)
    df.set_index("Period").to_csv("results.csv")


def item(p, i) -> float:
    analysis = mast_tools.MastOffsetPrecedenceAnalysis()
    s = System()
    cpu = Processor(name="cpu", sched=SchedulerType.FP)
    s.add_procs(cpu)
    flow_un = mast_meta.unavailable_flow(period=p, window=p / 6, cpu=cpu)
    flow_gen = genetic_flow(i, cpu)
    s.add_flows(flow_un, flow_gen)
    analysis.apply(s)
    return flow_gen.wcrt


def chart(file):
    df = pd.read_csv(file)
    df.plot()
    plt.show()


def genetic_flow(iterations, cpu):
    """
    Builds a flow that models the genetic algorithm. Each iteration uses a GPU once
    """
    prio = 100
    flow = Flow(name="genetic", period=10000, deadline=10000)
    flow.add_tasks(Task(name="previo", wcet=5, bcet=5, priority=prio, processor=cpu))

    for i in range(iterations):
        prio -= 1
        flow.add_tasks(Task(name="gpu_"+str(i), type=TaskType.Delay, wcet=50, bcet=30))
        flow.add_tasks(Task(name="iter_"+str(i), wcet=8, bcet=8, priority=prio, processor=cpu))

    prio -= 1
    flow.add_tasks(Task(name="resultado", wcet=3, bcet=3, priority=prio, processor=cpu))
    return flow


if __name__ == '__main__':
    run()
    # chart("results.csv")