from functools import partial
from examples import get_palencia_system
from model import Task, System
from analysis import higher_priority, HolisticFPAnalysis, init_wcrt, repr_wcrts
from mast_tools import MastHolisticAnalysis, MastOffsetPrecedenceAnalysis, MastOffsetAnalysis
import math
import numpy as np
import matplotlib.pyplot as plt
import examples
import assignment
import random
import time
from assignment import random_priority_jump
from metrics import invslack


def converge(f, x, stop_func=None):
    v = f(x)
    if stop_func and stop_func(v):
        return v
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
    ax.axhline(y=fast_converge(f, task.period*n), color='black')

    ax.text(sol, sol - 20, f"{sol:.2f}")
    ax.text(0.05, 0.9, f"p={p}", transform=ax.transAxes)


def func_w(task: Task, p, w, ceiling=True):
    hp = higher_priority(task)
    ceil = np.ceil if ceiling else lambda x: x
    result = (p * task.wcet
              + sum(map(lambda t: ceil((t.jitter + w) / t.period) * t.wcet, hp)))
    return result


def func_r(task: Task, p, r):
    hp = higher_priority(task)
    result = (p * task.wcet
              + sum(map(lambda t: np.ceil((t.jitter + r + (p-1)*task.period - task.jitter) / t.period) * t.wcet, hp))
              - (p-1)*task.period + task.jitter)
    return result


def w_to_r(w, task, p):
    return w - (p - 1) * task.period + task.jitter


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
                wp = converge(f, p*task.wcet, stop_func=lambda w: w_to_r(w, task, p) > task.flow.deadline*10)
                rp = w_to_r(wp, task, p)
                if rp > task.wcrt:
                    task.wcrt = rp
                if wp <= bound or rp > 10*task.flow.deadline:
                    break
                p += 1
        r = np.array([task.wcrt for task in s.tasks])


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


def fast_holistic_fp(s: System, ceiling=True, limit_i=-1, limit_p=-1):
    init_wcrt(s)
    rprev = np.array([task.wcrt for task in s.tasks])
    r = np.zeros_like(rprev)

    i = limit_i
    while not np.allclose(r, rprev):
        if i == 0:
            break
        i -= 1

        rprev = r
        for task in s.tasks:
            n = len(task.flow.tasks)
            p = 1
            while True:
                # iterate p=1,2,..., until wp is less than this bound
                bound = p*task.period
                f = partial(func_w, task, p, ceiling=ceiling)
                wp = fast_converge(f, task.period*n)
                rp = wp - (p-1)*task.period + task.jitter
                if rp > task.wcrt:
                    task.wcrt = rp
                if rp > 10*task.flow.deadline:
                    return
                if wp <= bound or p == limit_p:
                    break
                p += 1

        r = np.array([task.wcrt for task in s.tasks])


def holistic_comparison():
    # s = get_palencia_system()
    s = examples.get_big_system(random=random.Random(1), utilization=0.7)
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

    # Holistic FP W
    init_wcrt(s)
    before = time.process_time()
    holistic_fp_w(s)
    after = time.process_time()
    print(repr_wcrts(s), end="")
    print(f"FP W time={after-before}\n")

    # Holistic FP R
    init_wcrt(s)
    before = time.process_time()
    holistic_fp_r(s)
    after = time.process_time()
    print(repr_wcrts(s), end="")
    print(f"FP R time={after-before}\n")

    fig, axs = plt.subplots(2, 2)
    for p, ax in enumerate(axs.flat):
        chart(s.tasks[7], p + 1, ax)
        p = p + 1

    plt.show()


def correlation_chart(ax: plt.Axes, funca, funcb, s):
    rnd = random.Random(1)
    pd = assignment.PDAssignment()
    pd.apply(s)

    xy = []
    for i in range(1000):
        print(i)
        init_wcrt(s)
        funca(s)
        x = invslack(s)

        init_wcrt(s)
        funcb(s)
        y = invslack(s)

        xy.append((x, y))
        random_priority_jump(s, rnd)

    x, y = zip(*xy)
    ax.scatter(x, y, alpha=0.5)
    corr = np.corrcoef(x, y)
    return corr


def correlation():
    s = examples.get_small_system(random=random.Random(42), utilization=0.7, balanced=True)
    holistic = HolisticFPAnalysis(reset=False)
    offsets = MastOffsetAnalysis()

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    funca = lambda s: holistic.apply(s)

    # 1 holistic vs offsets
    ax = axs.flat[0]
    c = correlation_chart(ax, funca, lambda s: offsets.apply(s), s)
    ax.text(0.05, 0.9, f"c={c[0, 1]:.2f}", transform=ax.transAxes)
    ax.set_title("holistic vs offsets")

    setups = []
    # 2 holistic vs fast (i=inf, p=inf, ceil=true)
    setups.append(("holistic vs fast", -1, -1, True))
    # 3 holistic vs fast (i=1, p=1, ceil=true)
    setups.append(("holistic vs fast (1)", 1, 1, True))
    # 4 holistic vs fast (i=1, p=1, ceil=false)
    setups.append(("holistic vs fast (1, false)", 1, 1, False))

    for ax, (name, i, p, ceil) in zip(axs.flat[1:], setups):
        funcb = partial(fast_holistic_fp, ceiling=ceil, limit_i=i, limit_p=p)
        c = correlation_chart(ax, funca, funcb, s)
        ax.text(0.05, 0.9, f"c={c[0, 1]:.2f}", transform=ax.transAxes)
        ax.set_title(name)

    plt.show()


def anomaly():
    s = examples.get_medium_system(random=random.Random(2), utilization=0.4)
    pd = assignment.PDAssignment()
    pd.apply(s)

    rnd = random.Random(1)
    for i in range(1000):
        if i == 384:
            print(assignment.repr_priorities(s))
            funcb = partial(fast_holistic_fp, ceiling=True, limit_i=-1, limit_p=-1)
            funcb(s)
            x = invslack(s)
            print(x)

        random_priority_jump(s, rnd)


if __name__ == '__main__':
    correlation()
    # anomaly()