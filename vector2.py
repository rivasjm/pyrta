from model import System
import numpy as np
from gradient_funcs import BatchCostFunction


class ResultsCache:
    def __init__(self):
        self.data = dict()

    @staticmethod
    def _key(priority_matrix: np.ndarray):
        return priority_matrix.tobytes()

    def insert(self, priority_matrix: np.ndarray, results: np.ndarray):
        key = self._key(priority_matrix)
        if key not in self.data:
            self.data[key] = results

    def get(self, priority_matrix: np.ndarray):
        key = self._key(priority_matrix)
        return self.data[key] if key in self.data else None

    def has_results(self, priority_matrix: np.ndarray):
        key = self._key(priority_matrix)
        return key in self.data

    def clear(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)


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

    @staticmethod
    def _analysis(priority_matrix, wcets, periods, deadlines, successors, verbose=False, limit=10):
        assert wcets.shape == periods.shape == deadlines.shape == successors.shape
        assert wcets.shape[1] == 1
        assert priority_matrix.shape[1] == wcets.shape[0]

        # cache results
        cache = ResultsCache()

        # working priority matrix. initially with every scenario. scenarios will pop-out when they converge
        pm = priority_matrix.copy()

        # there are t tasks, and s scenarios
        s, t, _ = pm.shape

        # the successors' matrix maps, for each task (row), which task is its successor (column)
        # this is a 2D matrix (all scenarios have the same successors mapping)
        sm = successor_matrix(successors)

        # initialize response times. 3D matrix (s, t, 1)
        r_max = np.zeros((s, t, 1), dtype=np.float64)  # stores max WCRT found for each task and scenario
        r = np.full_like(r_max, 0.)                  # temporarily stores iteration response times

        # jitter matrix with current wcrt. 3D matrix (s, t, 1)
        j = jitter_matrix(sm, r_max)

        # wcrt limit for each task. if the provisional wcrt of any task reaches this, that scenario should stop
        r_limit = limit * deadlines  # 2D matrix (t, 1)

        # response time convergence loop
        while pm.size > 0:
            r_prev = r  # remember response times of previous iteration

            # iterate p=1,2,... until w<=p*T
            p = 1

            # p_mask: true when a task w <= p*t
            p_mask = np.full_like(pm, False)

            # p iterations loop
            while not np.all(p_mask):
                # no more p when w <= p*T
                stop_p = p * periods

                # initialize w
                w = np.zeros_like(r)
                w_prev = np.full_like(r, -1.)

                # w convergence loop
                while not np.allclose(w, w_prev):
                    w_prev = w

                    # Eq. (1) of "On the schedulability Analysis for Distributed Hard Real-Time Systems"
                    w = pm * np.ceil((w + pm * j.transpose(0, 2, 1)) / periods.T) @ wcets + p * wcets

                    # response time for this w. used to stop as early as possible if a task surpassed its WCRT limit
                    r_prov = w - (p - 1) * periods + j

                    # save the highest response time seen until now
                    r_max = np.maximum(r_prov, r_max)

                    # identify the scenarios that have reached their r-limit
                    over = scenarios_over_limit(r_max, r_limit)

                    if np.any(over):
                        cache_scenario_results(r_max, pm, over, cache)


                    # update jitters
                    j = jitter_matrix(sm, r_max)

                    # cache results of those scenarios that are over their limit
                    # remove those scenarios from necessary matrices
                    # TODO

                p_mask = w <= stop_p  # tasks that have this true don't need to try more p's
                p += 1

            # identify tasks whose WCRT has not changed from the previous iteration
            # identify finished scenarios: all its tasks have converged
            # cache results of finished scenarios, remove finished scenarios
            # TODO
            # converged = np.all(r == r_prev, axis=1).reshape((s, 1, 1))
            # isclose instead?

            # remove scenarios from boolean array
            # np.delete(a, np.array([False, True, False]), axis=0)

        # reconstruct the r_max matrix from the cache
        # TODO return reconstructed r_max



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


def finished_scenarios(rmask: np.ndarray) -> [int]:
    """
    rmask is a (s, n, 1) matrix, where s is the number of scenarios and n the number of tasks
    A scenario is finished when all the values in its plane are 0
    Returns the indices of the finished scenarios
    """
    return np.where(~rmask.any(axis=1))[0].tolist()


def scenarios_over_limit(r, r_limit):
    """
    returns (s) boolean vector indicating which scenarios have surpassed their limit
    A scenario surpasses his limit when any of its tasks surpasses its limit
    """
    s, _, _ = r.shape
    over = np.any(r > r_limit, axis=1).reshape(s)
    return over


def extract_scenario_data(data: np.ndarray, scenario: int) -> np.ndarray:
    """
    data is a (s, x, y) matrix, where s is the number of scenarios
    Returns the data of a scenario as a 2D array, with shape (x, y)
    """
    _, x, y = data.shape
    return data[scenario, ::].reshape(x, y)


def cache_scenario_results(r, pm, scenarios, cache: ResultsCache):
    """
    r is a (s, t, 1) matrix with the results
    pm is a (s, t, t) full 3D priority matrix
    scenarios is a (s) boolean matrix indicating which scenarios to cache
    cache is the storage of results. key a 2D (t,t) priority matrix, value the (t,1) results
    """
    for s in np.where(scenarios)[0]:
        key = extract_scenario_data(pm, s)
        value = extract_scenario_data(r, s)
        cache.insert(key, value)


def remove_scenarios(scenarios, *matrices):
    """
    scenarios is a (s) boolean matrix indicating which scenarios to remove
    matrices is a list of (s, x, y) matrices
    Removes the indicated scenarios from the matrices, and returns them
    """
    res = [matrix[~scenarios] for matrix in matrices]
    return res


def build_results_from_cache(pm, cache: ResultsCache):
    """
    pm is the full 3D (s, t, t) priority matrix with every scenario
    cache has the results of different 2D priority matrices (t, t)
    each result in the cache has a shape (t, 1)
    Returns a 2D matrix of results (t, s)
    """
    s, _, _ = pm.shape
    data = [cache.get(extract_scenario_data(pm, i)) for i in range(s)]
    res = np.concatenate(data, axis=1)
    return res