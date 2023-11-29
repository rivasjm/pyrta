from examples import get_three_tasks
from gradient_funcs import BatchCostFunction
from vector import successor_matrix, jitter_matrix
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
    t, s = priorities.shape
    planes = priorities.ravel(order='F').reshape(s, t, 1)
    return planes < planes.transpose(0, 2, 1) * (mappings == mappings.T)


class FastHolisticBatchCosts(BatchCostFunction):
    def apply(self, S: System, inputs: [[float]]) -> [float]:
        init_wcrt(S)

        wcets, periods, _, deadlines, mappings, successors, jitters = vectors(S)
        priorities = np.array(inputs).transpose()
        pm = priority_matrix(priorities, mappings)
        shape = (pm.shape[0], pm.shape[1], 1)
        p = 1

        x1 = np.zeros_like(wcets).T
        y1 = p*wcets.T + np.sum(pm*(jitters + x1) * wcets / periods, axis=2).reshape(shape)
        x2 = (deadlines - jitters).reshape(x1.shape)
        y2 = p*wcets.T + np.sum(pm*(jitters + x2) * wcets / periods, axis=2).reshape(shape)
        r = y1/(1-(y2-y1)/(x2-x1)) + jitters.T

        invslack = np.max((r-deadlines.T)/deadlines.T, axis=1).reshape((-1, ))
        return invslack.tolist()


if __name__ == '__main__':
    batch = FastHolisticBatchCosts()
    input = [[10, 5, 1], [1, 5, 10]]
    s = get_three_tasks()
    invslack = batch.apply(s, input)
    print(invslack)


