from model import System
import numpy as np
from gradient_funcs import BatchCostFunction


def system_priority_matrix(system: System) -> np.array:
    """Creates the 2D priority matrix of the given input system"""
    tasks = system.tasks
    procs = system.processors
    priorities = np.array([task.priority for task in tasks]).reshape((-1, 1))
    mapping = np.array([procs.index(task.processor) for task in tasks]).reshape((-1, 1))
    pm = (priorities < priorities.T) * (mapping == mapping.T)
    return pm


class PriorityScenarios:
    """
    Class to generate the 3D matrix of priority scenarios given a list of input parameters
    """
    def apply(self, S: System, inputs: [[float]]) -> np.array:
        """
        Returns a 3D priority matrix for the given list of input parameters
        Each input list generates one priority scenario, which is stored in a plane of the output 3D matrix
        """
        pass


class VectorHolisticFPBatchCosts(BatchCostFunction):
    def __init__(self, vector_matrix: PriorityScenarios):
        self.vector_matrix = vector_matrix

    def apply(self, S: System, inputs: [[float]]) -> [float]:
        tasks = S.tasks
        n = len(tasks)
        PM = self.vector_matrix.apply(S, inputs)
        deadlines = np.array([task.flow.deadline for task in tasks]).reshape(n, 1)
        vholistic = VectorHolisticFPAnalysis(limit_factor=10, verbose=False)
        vholistic.apply(S, scenarios=PM)
        r = vholistic.scenarios_response_times
        costs = np.max((r - deadlines) / deadlines, axis=0)
        return costs


# class VectorHolisticFPBatchCostsOrig(BatchCostFunction):
#     def apply(self, S: System, inputs: [[float]]) -> [float]:
#         tasks = S.tasks
#         n = len(tasks)
#         priorities = np.array(inputs).transpose()
#         deadlines = np.array([task.flow.deadline for task in tasks]).reshape(n, 1)
#         vholistic = VectorHolisticFPAnalysis(limit_factor=10, verbose=False)
#         vholistic.set_priority_scenarios(priorities)
#         vholistic.apply(S)
#         r = vholistic.response_times
#         costs = np.max((r - deadlines) / deadlines, axis=0)
#         return costs


def proc(task_mapping):
    proc_index = np.array(task_mapping).argmax() + 1
    return proc_index


class MappingPrioritiesMatrix(PriorityScenarios):
    def apply(self, S: System, inputs: [[float]]) -> np.array:
        p = len(S.processors)
        n = len(S.tasks)
        pm = np.zeros((len(inputs), n, n))
        # each input x of length p*n + n
        #  first p*t is for mapping
        #  rest n for priorities
        for i, x in enumerate(inputs):
            mapping = np.array([proc(x[t*p:t*p+p]) for t in range(n)]).reshape(-1, 1)
            priorities = np.array(x[-n:]).reshape(-1, 1)
            temp = (priorities < priorities.T) * (mapping == mapping.T)
            pm[i::] = temp
        return pm


class PrioritiesMatrix(PriorityScenarios):
    def apply(self, S: System, inputs: [[float]]) -> np.array:
        n = len(S.tasks)
        pm = np.zeros((len(inputs), n, n))
        procs = S.processors
        mapping = np.array([procs.index(task.processor)+1 for task in S.tasks]).reshape(-1, 1)
        for i, x in enumerate(inputs):
            priorities = np.array(x[-n:]).reshape(-1, 1)
            temp = (priorities < priorities.T) * (mapping == mapping.T)
            pm[i::] = temp
        return pm


class VectorHolisticFPAnalysis:
    def __init__(self, verbose=False, limit_factor=10):
        self.verbose = verbose
        self.limit_factor = limit_factor
        self._scenarios_response_times = None
        self._full_response_times = None

    def clear_results(self):
        self._scenarios_response_times = None
        self._full_response_times = None

    @property
    def scenarios_response_times(self):
        """2D Matrix of response times for the additional priority scenarios. One column per scenario"""
        return self._scenarios_response_times

    @property
    def full_response_times(self):
        """2D Matrix of response times for the input system + additional priority scenarios. One column per scenario"""
        return self._full_response_times

    # @property
    # def full_priorities(self):
    #     """Matrix with the system priorities + the priority scenarios"""
    #     return self._full_priorities

    # def set_priority_scenarios(self, priorities):
    #     """Set the priority scenarios to analye, in addition to the priorities currently set in the system"""
    #     self.priority_scenarios = priorities

    # def set_escenarios_priority_matrix(self, scenarios_pm: np.array):
    #     """
    #     Set the additional priority scenarios to analyze.
    #     These are added over the priority matrix of the input system
    #     """
    #     self.scenarios_pm = scenarios_pm

    @staticmethod
    def _analysis(priority_matrix, wcets, periods, deadlines, successors, verbose=False, limit=10):
        assert wcets.shape == periods.shape == deadlines.shape == successors.shape
        assert wcets.shape[1] == 1
        assert priority_matrix.shape[1] == wcets.shape[0]

        # create a 3D priority matrix, where each plane is a priority matrix for each scenario
        # the objective is to be able to analyze several priority assignments at the same vector_times
        PM = priority_matrix  # 3D matrix with the priority scenarios (one plane per scenario)

        # there are t tasks, and s scenarios
        s, t, _ = PM.shape

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

    def apply(self, system: System, scenarios: np.array = None):
        """
        Execute the vectorized Holistic analysis for FP systems in the given input system
        Optionally, additional priority scenarios can be provided. Array "scenarios" is a
        3D matrix with the additional priority scenarios, in which each plane (first dimension
        is the priority matrix of a scenario)
        """
        wcets, periods, deadlines, successors, _, _ = get_vectors(system)

        n = len(wcets)  # number of tasks
        input_pm = system_priority_matrix(system).reshape(1, n, n)  # priority matrix of input system
        s = 0 if scenarios is None else scenarios.shape[0]  # number of additionl scenarios

        # pack all priority scenarios in the same matrix
        # the first scenario is for the priorities in the input system
        pm = np.concatenate((input_pm, scenarios), axis=0) if s > 0 else input_pm

        # get response times for all scenarios. r is a 3D matrix (s+1, n, 1)
        r = self._analysis(pm, wcets, periods, deadlines, successors, verbose=self.verbose, limit=self.limit_factor)

        # set the response times of the first scenario as the wcrt of the input system
        for task, wcrt in zip(system.tasks, r[0].ravel()):
            task.wcrt = wcrt

        # save scenarios response times
        self._full_response_times = r.ravel(order="F").reshape((n, s+1))
        self._scenarios_response_times = r[1:, :, :].ravel(order="F").reshape((n, s)) if s > 0 else None


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


def get_vectors(system: System, single_precision=False):
    """Transform a system into vectors. The vectorized analysis is based on these vectors"""
    float_type = np.float32 if single_precision else np.float64
    int_type = np.int32 if single_precision else np.int64

    tasks = system.tasks
    t = len(tasks)
    wcets = np.zeros((t, 1), dtype=float_type)
    periods = np.zeros((t, 1), dtype=float_type)
    deadlines = np.zeros((t, 1), dtype=float_type)
    successors = np.zeros((t, 1), dtype=int_type)
    mappings = np.zeros((t, 1), dtype=object)
    priorities = np.zeros((t, 1), dtype=float_type)

    taskmap = {task: i for i, task in enumerate(tasks)}

    for task, i in taskmap.items():
        wcets[i] = task.wcet
        periods[i] = task.period
        deadlines[i] = task.flow.deadline
        mappings[i] = task.processor.name
        priorities[i] = task.priority
        successors[i] = taskmap[task.successors[0]]+1 if task.successors else -1

    return wcets, periods, deadlines, successors, mappings, priorities