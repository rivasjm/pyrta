from model import Task, Processor, Flow, System, TaskType, SchedulerType
import numpy as np
import pandas as pd
import itertools
import mast_tools, mast_meta
import matplotlib.pyplot as plt


def run():

    periods = np.arange(1.25, 31.25, 1.25)
    iterations = np.arange(1, 6, 1)
    df = pd.DataFrame()
    df["Period"] = periods

    analysis_techniques = ["holistic", "precedence"]

    for technique in analysis_techniques:
        for i in iterations:
            name = f"{i} iterations"
            res = [item(p,i,technique) for p in periods]
            df[name] = res
        print(df)
        df.set_index("Period").to_csv("results_"+technique+".csv")


def item(p, i, t) -> float:

    if t == "holistic":
        analysis = mast_tools.MastHolisticAnalysis()
    elif t == "precedence":
        analysis = mast_tools.MastOffsetPrecedenceAnalysis()

    s = System()
    cpu = Processor(name="cpu", sched=SchedulerType.FP)
    s.add_procs(cpu)
    window_size = p / 2.5
    flow_un = mast_meta.unavailable_flow(period=p, window=window_size, cpu=cpu)
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

    flow = Flow(name="genetic", period=100, deadline=100)
    flow.add_tasks(Task(name="prev", wcet=3, bcet=3, priority=prio, processor=cpu))
    flow.add_tasks(Task(name="gpu_" + str(0), type=TaskType.Delay, wcet=10, bcet=10))

    for i in range(iterations):
        prio -= 1
        flow.add_tasks(Task(name="iter_" + str(i), wcet=4, bcet=4, priority=prio, processor=cpu))
        flow.add_tasks(Task(name="gpu_"+str(i+1), type=TaskType.Delay, wcet=10, bcet=10))

    prio -= 1
    flow.add_tasks(Task(name="post", wcet=3, bcet=3, priority=prio, processor=cpu))
    return flow


if __name__ == '__main__':
    run()
    # chart("results.csv")