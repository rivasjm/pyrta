from functools import partial
from examples import get_palencia_system
from model import Task, System
from analysis import higher_priority, HolisticFPAnalysis, init_wcrt, repr_wcrts
from mast_tools import MastHolisticAnalysis
import math
import numpy as np
import matplotlib.pyplot as plt
import examples
import assignment
import random
import time


def converge(f, x):
    v = f(x)
    return v if math.isclose(v, x) else converge(f, v)


def fast_converge(f, x2):
    x1 = 0
    y1 = f(x1)
    y2 = f(x2)
    sol = y1/(1-(y2-y1)/(x2-x1))
    return sol


def chart(task: Task, p, ax):
    n = len(task.flow.tasks)
    f = partial(func_r, task, p)
    x = np.linspace(0, math.ceil(task.period)*n, math.ceil(task.flow.deadline))
    sol = converge(f, task.wcrt)

    # DBF
    ax.plot(x, f(x), color='red')

    # w=w
    ax.plot(x, x, color='blue')

    # intersection
    ax.axhline(y=sol, color='g', linestyle='-')

    # p limit
    ax.axhline(y=task.period + task.jitter, color='orange')

    # approx intersection
    ax.axhline(y=fast_converge(f, task.period), color='black')

    ax.text(sol, sol - 20, f"{sol:.2f}")
    ax.text(0.05, 0.9, f"p={p}", transform=ax.transAxes)


def func_w(task: Task, p, w):
    hp = higher_priority(task)
    result = (p * task.wcet
              + sum(map(lambda t: np.ceil((t.jitter + w) / t.period) * t.wcet, hp)))
    return result


def holistic_fp_w(s: System):
    init_wcrt(s)
    rprev = np.array([task.wcrt for task in s.tasks])
    r = np.zeros_like(rprev)

    while not np.allclose(r, rprev):
        rprev = r
        for task in s.tasks:
            p = 1
            while True:
                # iterate p=1,2,..., until wp is less than this bound
                bound = p*task.period
                f = partial(func_w, task, p)
                wp = converge(f, p*task.wcet)
                rp = wp - (p-1)*task.period + task.jitter
                if rp > task.wcrt:
                    task.wcrt = rp
                if wp <= bound:
                    break
                p += 1
        r = np.array([task.wcrt for task in s.tasks])


def func_r(task: Task, p, r):
    hp = higher_priority(task)
    result = (p * task.wcet
              + sum(map(lambda t: np.ceil((t.jitter + r + (p-1)*task.period - task.jitter) / t.period) * t.wcet, hp))
              - (p-1)*task.period + task.jitter)
    return result


def holistic_fp_r(s: System):
    init_wcrt(s)
    rprev = np.array([task.wcrt for task in s.tasks])
    r = np.zeros_like(rprev)

    while not np.allclose(r, rprev):
        rprev = r
        for task in s.tasks:
            p = 1
            while True:
                bound = task.period + task.jitter
                f = partial(func_r, task, p)
                rp = converge(f, p*task.wcet)
                if rp > task.wcrt:
                    task.wcrt = rp
                if rp <= bound:
                    break
                p += 1
        r = np.array([task.wcrt for task in s.tasks])


if __name__ == '__main__':
    # s = get_palencia_system()
    s = examples.get_medium_system(random=random.Random(1), utilization=0.7)
    pd = assignment.PDAssignment()
    pd.apply(s)

    # Holistic FP MAST
    init_wcrt(s)
    before = time.process_time()
    MastHolisticAnalysis().apply(s)
    after = time.process_time()
    print(repr_wcrts(s), end="")
    print(f"MAST time={after-before}\n")

    # Holistic FP
    init_wcrt(s)
    before = time.process_time()
    HolisticFPAnalysis().apply(s)
    after = time.process_time()
    print(repr_wcrts(s), end="")
    print(f"FP time={after-before}\n")

    # Holistic FP R
    init_wcrt(s)
    before = time.process_time()
    holistic_fp_r(s)
    after = time.process_time()
    print(repr_wcrts(s), end="")
    print(f"FP R time={after-before}\n")

    # Holistic FP W
    init_wcrt(s)
    before = time.process_time()
    holistic_fp_w(s)
    after = time.process_time()
    print(repr_wcrts(s), end="")
    print(f"FP W time={after-before}\n")

    fig, axs = plt.subplots(2, 2)
    for p, ax in enumerate(axs.flat):
        chart(s.tasks[7], p + 1, ax)
        p = p + 1

    plt.show()