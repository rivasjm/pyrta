import numpy as np
from examples import get_palencia_system
from generator import to_edf
from assignment import PDAssignment
from model import System
from analysis import init_wcrt


def analysis():
    system = to_edf(get_palencia_system())
    init_wcrt(system)
    PDAssignment(globalize=True).apply(system)
    c, t, d, j, fd, s, procs, _ = get_vectors(system)
    l = np.zeros_like(c)
    m = procs == procs.T

    l = compute_l(l, j, t, c, m)
    p = compute_p(l, j, t)
    psi = compute_psi(l, j, t, d, p)
    print(psi)


def compute_l(l_prev, j, t, c, m):
    while True:
        l = ((np.ceil((l_prev + j) / t) * c).T @ m).T
        if np.allclose(l, l_prev):
            return l
        l_prev = l


def compute_p(l, j, t):
    return np.ceil((l+j)/t).astype(int)


def compute_psi(l, j, t, d, p):
    # TODO optimizar esto para usar operaciones matriciales en vez de un for
    size = (t.shape[0], np.max(p))
    psi = np.zeros(size)
    for i, x in enumerate(p):
        row = np.pad(np.arange(1, x+1), (0, size[1]-x.item()), 'constant')
        psi[i, :] = row
    mask = psi > 0  # to remove those cases where (p-1) becomes negative
    psi = (psi-1)*t-j+d
    return psi * mask


def get_vectors(S: System):
    tasks = S.tasks
    t = len(tasks)
    wcets = np.zeros((t, 1), dtype=np.float64)
    periods = np.zeros((t, 1), dtype=np.float64)
    deadlines = np.zeros((t, 1), dtype=np.float64)
    jitters = np.zeros((t, 1), dtype=np.float64)
    flow_deadlines = np.zeros((t, 1), dtype=np.float64)
    successors = np.zeros((t, 1), dtype=np.int64)
    mappings = np.zeros((t, 1), dtype=object)
    priorities = np.zeros((t, 1), dtype=np.float64)

    taskmap = {task: i for i, task in enumerate(tasks)}

    for task, i in taskmap.items():
        wcets[i] = task.wcet
        periods[i] = task.period
        deadlines[i] = task.deadline
        jitters[i] = task.jitter
        flow_deadlines[i] = task.flow.deadline
        mappings[i] = task.processor.name
        priorities[i] = task.priority
        successors[i] = taskmap[task.successors[0]] + 1 if task.successors else -1

    return wcets, periods, deadlines, jitters, flow_deadlines, successors, mappings, priorities


if __name__ == '__main__':
    analysis()