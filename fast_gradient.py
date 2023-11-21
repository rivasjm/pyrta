from examples import get_three_tasks
from vector import get_vectors, priority_matrix, successor_matrix, jitter_matrix
from model import System
import numpy as np
from analysis import init_wcrt


def vectors(S: System):
    tasks = S.tasks
    procs_map = {proc: i for i, proc in enumerate(S.processors)}
    tasks_map = {task: i for i, task in enumerate(tasks)}

    shape = (1, len(tasks))  # horizontal vectors
    # wcets, periods, priorities, deadlines, mappings, successors
    wcets = np.array([t.wcet for t in tasks], dtype=np.float32).reshape(shape)
    periods = np.array([t.period for t in tasks], dtype=np.float32).reshape(shape)
    priorities = np.array([t.priority for t in tasks], dtype=np.float32).reshape(shape)
    deadlines = np.array([t.flow.deadline for t in tasks], dtype=np.float32).reshape(shape)
    mappings = np.array([procs_map[t.processor] for t in tasks], dtype=np.int32).reshape(shape)
    successors = np.array([tasks_map[t.successors[0]] if t.successors else -1 for t in tasks], dtype=np.int32).reshape(shape)
    jitters = np.array([t.jitter for t in tasks], dtype=np.float32).reshape(shape)
    return wcets, periods, priorities, deadlines, mappings, successors, jitters


def priority_matrix(priorities, mappings):
    return priorities.T < priorities * (mappings.T == mappings)


class FastGDPA:
    def apply(self, S: System):
        init_wcrt(S)

        wcets, periods, priorities, deadlines, mappings, successors, jitters = vectors(S)
        pm = priority_matrix(priorities, mappings)
        shape = wcets.T.shape
        p = 1

        x1 = np.zeros_like(wcets).T
        y1 = p*wcets.T + np.sum(pm*(jitters + x1) * wcets / periods, axis=1).reshape(shape)

        x2 = (deadlines - jitters).reshape(shape)
        y2 = p*wcets.T + np.sum(pm*(jitters + x2) * wcets / periods, axis=1).reshape(shape)

        sol = y1/(1-(y2-y1)/(x2-x1))
        # print(y1)
        # print(y2)
        return sol


if __name__ == '__main__':
    fgdpa = FastGDPA()
    s = get_three_tasks()
    fgdpa.apply(s)

