from functools import partial
from examples import get_palencia_system
from model import Task
from analysis import higher_priority, HolisticFPAnalysis, init_wcrt
import math


def converge(f, x):
    v = f(x)
    return v if math.isclose(v, x) else converge(f, v)


def func_r(task: Task, p, r):
    hp = higher_priority(task)
    result = (p * task.wcet + sum(map(lambda t: math.ceil((t.jitter + r + (p-1)*task.period - task.jitter) / t.period) * t.wcet, hp))
              - (p-1)*task.period + task.jitter)
    return result


if __name__ == '__main__':
    s = get_palencia_system()
    init_wcrt(s)
    task = s.tasks[2]
    f = partial(func_r, task, 1)
    sol = converge(f, 0)
    print(sol)