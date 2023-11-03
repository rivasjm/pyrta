from model import System, Task
import numpy as np
from analysis import higher_priority, init_wcrt
from functools import partial


def func_w(task: Task, p, w, ceiling=True):
    """eq (1) from the paper "On the Schedulability ...."""
    hp = higher_priority(task)
    ceil = np.ceil if ceiling else lambda x: x
    result = (p * task.wcet
              + sum(map(lambda t: ceil((t.jitter + w) / t.period) * t.wcet, hp)))
    return result


def fast_converge(f, x2):
    """Fast converge function f (not exact but faster than recursive solution)"""
    x1 = 0
    y1 = f(x1)
    y2 = f(x2)
    sol = y1/(1-(y2-y1)/(x2-x1))
    return sol


class FastHolisticFPAnalysis:
    def __init__(self, limit_factor=10, limit_i=-1, limit_p=-1, ceiling=True):
        self.limit_factor = limit_factor
        self.limit_i = limit_i
        self.limit_p = limit_p
        self.ceiling = ceiling

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
                    f = partial(func_w, task, p, ceiling=self.ceiling)
                    wp = fast_converge(f, task.period * n)
                    rp = wp - (p - 1) * task.period + task.jitter
                    if rp > task.wcrt:
                        task.wcrt = rp
                    if rp > 10 * task.flow.deadline:
                        return
                    if wp <= bound or p == self.limit_p:
                        break
                    p += 1

            r = np.array([task.wcrt for task in system.tasks])