from model import System
import numpy as np
from gradient_funcs import BatchCostFunction


class VectorHolisticFPBatchCosts(BatchCostFunction):
    def apply(self, S: System, inputs: [[float]]) -> [float]:
        tasks = S.tasks
        n = len(tasks)
        priorities = np.array(inputs).transpose()
        deadlines = np.array([task.flow.deadline for task in tasks]).reshape(n, 1)
        vholistic = VectorHolisticFPAnalysis(limit_factor=10, verbose=False)
        vholistic.set_priority_scenarios(priorities)
        vholistic.apply(S)
        r = vholistic.response_times
        costs = np.max((r - deadlines) / deadlines, axis=0)
        return costs


class VectorHolisticFPAnalysis:
    def __init__(self, verbose=False, limit_factor=10):
        self.verbose = verbose
        self.limit_factor = limit_factor
        self.priority_scenarios = None
        self._response_times = None
        self._full_priorities = None
        self._full_response_times = None

    def clear(self):
        self.priority_scenarios = None
        self._response_times = None
        self._full_priorities = None
        self._full_response_times = None

    @property
    def response_times(self):
        """Response vector_times matrix of the priority scenarios"""
        return self._response_times

    @property
    def full_response_times(self):
        """Response vector_times matrix of the system + the priority scenarios"""
        return self._full_response_times

    @property
    def full_priorities(self):
        """Matrix with the system priorities + the priority scenarios"""
        return self._full_priorities

    def set_priority_scenarios(self, priorities):
        """Set the priority scenarios to analysis, in addition to the priorities currently set in the system"""
        self.priority_scenarios = priorities

    @staticmethod
    def _analysis(wcets, periods, deadlines, successors, mappings, priorities, verbose=False, limit=10):
        assert wcets.shape == periods.shape == deadlines.shape == successors.shape == mappings.shape
        assert wcets.shape[1] == 1

        # there are t tasks, and s scenarios
        t, s = priorities.shape

        # 'priorities' has several columns, each column is a priority scenario for the system
        # create a 3D priority matrix, where each plane is a priority matrix for each scenario
        # the objective is to be able to analyze several priority assignments at the same vector_times
        PM = priority_matrix(priorities) * (mappings == mappings.T)

        # the successors' matrix maps, for each task (row), which task is its successor (column)
        # this is a 2D matrix (all scenarios have the same successors mapping)
        S = successor_matrix(successors)

        # initialize response times
        # 3D column vector, each plane for each scenario
        Rmax = np.zeros((s, t, 1), dtype=np.float64)
        R = np.full_like(Rmax, 0.)

        # update jitter matrix with current response times
        # 3D column vector, with jitter for each scenario (plane)
        J = jitter_matrix(S, Rmax)

        # a limit on the response times for each task
        # when a task provisional response vector_times reaches its r-limit:
        # - the analysis of its scenario should be stopped
        # - the response vector_times of the affected task and its successors should be set to the limit
        # - the system is therefore deemed non schedulable
        Rlimit = limit * deadlines

        # r mask. 3D column vector
        # when a task response vector_times converges, its value here is set to 0
        # when a task reaches its r-limit, the values for the whole scenario are set to 0
        # TODO this should be a bit per scenario: if a scenario converges or any task reaches its limit, mask=0
        rmask = np.ones_like(Rmax)

        # stop convergence of response vector_times if all tasks converged, or reached their r-limit
        while rmask.any():
            Rprev = R

            # initial activation index
            # TODO: idea to batch together several p iterations
            # define a batch size: how many p values to test at the same vector_times
            # add the batches as additional planes
            # for S scenarios, B batch size: we will have B*S planes
            # I guess I cannot use broadcasting with the STOP vector: expand STOP myself
            # The final +p*wcets in eq (1) cannot be broadcasted either (I guess). Expand this myself too.
            # I need to create a (B*S, tasks, 1) matrix to store the p values for each plane
            p = 1

            # p-limit mask. when a task reaches its p-limit, its bit is set to False here.
            pmask = np.ones_like(J)

            while True:
                # activation index is increased when w is smaller than STOP
                STOP = p * periods

                # initialize W
                W = np.zeros_like(J)
                Wprev = np.full_like(J, -1.)

                # W iteration. this stops when W converges
                while not np.allclose(W, Wprev):
                    Wprev = W
                    # Eq. (1) of "On the schedulability Analysis for Distributed Hard Real-Time Systems"
                    W = PM * np.ceil((W + PM * J.transpose(0, 2, 1)) / periods.T) @ wcets + p * wcets

                    # Ignore those tasks that have reached their p-limit already
                    W = W * pmask

                    # find the provisional response vector_times here
                    Rprov = rmask * (W - (p - 1) * periods + J)

                    # update worst-case response times and jitters
                    # there may be a task that have reached its r-limit, no problem,
                    # take the wcrt into account, and afterwards mask the scenario, so it's convergence stops
                    Rmax = np.maximum(Rprov, Rmax)
                    J = jitter_matrix(S, Rmax)

                    # identify the tasks that have reached their r-limit
                    # if a task reached its r-limit, set all the r-masks of its scenario to 0
                    # that scenario analysis is now stopped
                    rmask = rmask * np.all(Rprov < Rlimit, axis=1).reshape((s, 1, 1))

                    # also stop the p-iterations if already reached r-limit
                    pmask = rmask * pmask

                # once W converges, calculate the response times for this p
                # I can use the last Rprov for this. equation: R = W-(p-1)*periods+J
                R = rmask * Rprov

                # stop the p iterations if all W meet the stopping criteria
                # update the pmask here, and stop when pmask is all zeroes
                pmask = pmask * np.logical_not(W < STOP)
                if not pmask.any():
                    break

                # if no stopping criteria, try with next p
                p += 1

            # if a task response vector_times has not changed, sets its bit in the mask to zero
            rmask = rmask * np.logical_not(np.allclose(R, Rprev))

        return Rmax

    def apply(self, system: System):
        wcets, periods, deadlines, successors, mappings, priorities = get_vectors(system)

        # pack all priority scenarios in the same matrix
        # the first scenario is for the priorities in the input system
        if self.priority_scenarios is not None:
            priorities = np.hstack((priorities, self.priority_scenarios))
        self._full_priorities = priorities

        # get response times for all scenarios
        r = self._analysis(wcets, periods, deadlines, successors, mappings, priorities,
                          verbose=self.verbose, limit=self.limit_factor)

        # set the response times of the first scenario as the wcrt of the input system
        for task, wcrt in zip(system.tasks, r[0].ravel()):
            task.wcrt = wcrt

        # save scenarios response times
        scenarios = priorities.shape[1]
        self._full_response_times = r.ravel(order="F").reshape((len(system.tasks), scenarios))
        self._response_times = r[1:, :, :].ravel(order="F").reshape((len(system.tasks), scenarios-1)) \
            if scenarios > 1 else None


def successor_matrix(succesors):
    """Builds the successor matrix from a flat successor list. Task id's start from 1"""
    s = np.zeros((succesors.size, succesors.size), dtype=np.int32)
    for i, v in enumerate(succesors):
        if v > -1:
            s[i, v-1] = 1
    return s


def jitter_matrix(smatrix, r):
    """Builds the jitter matrix. Assumes at most 1 successor per task"""
    return smatrix.T @ r


def priority_matrix(priorities):
    """Builds a 3D priority matrix for the given priority scenarios"""
    t, s = priorities.shape
    planes = priorities.ravel(order='F').reshape(s, t, 1)
    P = planes < planes.transpose((0, 2, 1))
    return P


def get_vectors(system: System):
    """Transform a system into vectors. The vectorized analysis is based on these vectors"""
    tasks = system.tasks
    t = len(tasks)
    wcets = np.zeros((t, 1), dtype=np.float64)
    periods = np.zeros((t, 1), dtype=np.float64)
    deadlines = np.zeros((t, 1), dtype=np.float64)
    successors = np.zeros((t, 1), dtype=np.int64)
    mappings = np.zeros((t, 1), dtype=object)
    priorities = np.zeros((t, 1), dtype=np.float64)

    taskmap = {task: i for i, task in enumerate(tasks)}

    for task, i in taskmap.items():
        wcets[i] = task.wcet
        periods[i] = task.period
        deadlines[i] = task.flow.deadline
        mappings[i] = task.processor.name
        priorities[i] = task.priority
        successors[i] = taskmap[task.successors[0]]+1 if task.successors else -1

    return wcets, periods, deadlines, successors, mappings, priorities