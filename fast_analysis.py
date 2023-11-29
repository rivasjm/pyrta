from model import System, Task
import numpy as np
from analysis import higher_priority, init_wcrt
from functools import partial
import math


def func_w(task: Task, p, w, ceiling=True):
    """w function, eq (1) from the paper 'On the Schedulability ....'"""
    hp = higher_priority(task)
    ceil = np.ceil if ceiling else lambda x: x
    result = (p * task.wcet
              + sum(map(lambda t: ceil((t.jitter + w) / t.period) * t.wcet, hp)))
    return result


def func_r(w, p, task):
    """eq to compute response time from W, taken from the paper 'On the schedulability ...'"""
    return w - (p - 1) * task.period + task.jitter


def fast_converge(f, x2):
    """Fast converge function f (not exact but faster than recursive solution).
    This is an intersection netween 2 lines: one that mimics the w function, and line y=x"""
    x1 = 0
    y1 = f(x1)
    y2 = f(x2)
    sol = y1/(1-(y2-y1)/(x2-x1))
    return sol


def converge(f, x, stop_func=None):
    """Find convergence of function f recursively, starting form input value x"""
    v = f(x)
    if stop_func and stop_func(v):
        return v
    return v if math.isclose(v, x) else converge(f, v)


class FastHolisticFPAnalysis:
    def __init__(self, limit_factor=10, limit_i=-1, limit_p=-1, ceiling=True, fast=True):
        self.limit_factor = limit_factor    # analysis stops when any task reached r > 10*deadline
        self.limit_i = limit_i              # number of external iterations (-1 means unlimited)
        self.limit_p = limit_p              # maximum p checked (-1 means unlimited)
        self.ceiling = ceiling              # true if ceiling function is used to compute w
        self.fast = fast                    # true for fast but approximate convergence of w function

    def apply(self, system: System) -> None:
        init_wcrt(system)
        rprev = np.array([task.wcrt for task in system.tasks])
        r = np.zeros_like(rprev)

        i = self.limit_i
        while not np.allclose(r, rprev):
            if i == 0:
                break
            i -= 1

            rprev = r
            for task in system.tasks:
                n = len(task.flow.tasks)
                p = 1
                while True:
                    # iterate p=1,2,..., until wp is less than this bound
                    bound = p * task.period
                    r_limit = self.limit_factor * task.flow.deadline

                    # converge W function
                    f = partial(func_w, task, p, ceiling=self.ceiling)
                    if self.fast:
                        wp = fast_converge(f, task.flow.deadline - task.jitter)
                    else:
                        wp = converge(f, task.wcet*p, lambda v: func_r(v, p, task) > r_limit)

                    # response time of that W value
                    rp = func_r(wp, p, task)

                    # determine if iterations must stop
                    if rp > task.wcrt:
                        task.wcrt = rp
                    if rp > r_limit:
                        for succ in task.all_successors:
                            succ.wcrt = rp
                        return
                    if wp <= bound or p == self.limit_p:
                        break
                    p += 1

            r = np.array([task.wcrt for task in system.tasks])