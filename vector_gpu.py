from model import System
import torch
from gradient_funcs import BatchCostFunction
import analysis


class ResultsCache:
    def __init__(self):
        self.data = dict()

    @staticmethod
    def _key(priority_matrix: torch.Tensor):
        return hash(tuple(priority_matrix.reshape(-1).tolist()))

    def insert(self, priority_matrix: torch.Tensor, results: torch.Tensor):
        key = self._key(priority_matrix)
        if key not in self.data:
            self.data[key] = results

    def get(self, priority_matrix: torch.Tensor):
        key = self._key(priority_matrix)
        return self.data[key] if key in self.data else None

    def has_results(self, priority_matrix: torch.Tensor):
        key = self._key(priority_matrix)
        return key in self.data

    def reset(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)


def device():
    return torch.device("mps")


def system_priority_matrix(system: System) -> torch.Tensor:
    """Creates the 2D priority matrix of the given input system"""
    tasks = system.tasks
    procs = system.processors
    priorities = torch.FloatTensor([task.priority for task in tasks]).reshape((-1, 1)).to(device())
    mapping = torch.FloatTensor([procs.index(task.processor) for task in tasks]).reshape((-1, 1)).to(device())
    pm = (priorities <= priorities.T) * (mapping == mapping.T) * (~torch.eye(len(tasks), dtype=torch.bool, device=device()))
    return pm


class PriorityScenarios:
    """
    Class to generate the 3D matrix of priority scenarios given a list of input parameters
    """
    def apply(self, S: System, inputs: [[float]]) -> torch.Tensor:
        """
        Returns a 3D priority matrix for the given list of input parameters
        Each input list generates one priority scenario, which is stored in a plane of the output 3D matrix
        """
        pass


class VectorHolisticFPBatchCosts(BatchCostFunction):
    def __init__(self, vector_matrix: PriorityScenarios):
        self.vector_matrix = vector_matrix
        self.cache = ResultsCache()

    def reset(self):
        self.cache.reset()

    def apply(self, S: System, inputs: [[float]]) -> [float]:
        tasks = S.tasks
        n = len(tasks)
        PM = self.vector_matrix.apply(S, inputs)
        deadlines = torch.FloatTensor([task.flow.deadline for task in tasks]).reshape(n, 1).to(device())
        vholistic = VectorHolisticFPAnalysis(limit_factor=10, verbose=False, cache=self.cache)
        vholistic.apply(S, scenarios=PM)
        r = vholistic.scenarios_response_times
        costs = torch.max((r - deadlines) / deadlines, axis=0)
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
    proc_index = torch.FloatTensor(task_mapping).argmax() + 1
    return proc_index


class MappingPrioritiesMatrix(PriorityScenarios):
    def apply(self, S: System, inputs: [[float]]) -> torch.Tensor:
        p = len(S.processors)
        n = len(S.tasks)
        pm = torch.zeros((len(inputs), n, n)).to(device())
        # each input x of length p*n + n
        #  first p*t is for mapping
        #  rest n for priorities
        for i, x in enumerate(inputs):
            mapping = torch.FloatTensor([proc(x[t*p:t*p+p]) for t in range(n)]).reshape(-1, 1).to(device())
            priorities = torch.FloatTensor(x[-n:]).reshape(-1, 1).to(device())
            temp = (priorities < priorities.T) * (mapping == mapping.T)
            pm[i::] = temp
        return pm


class PrioritiesMatrix(PriorityScenarios):
    def apply(self, S: System, inputs: [[float]]) -> torch.Tensor:
        n = len(S.tasks)
        pm = torch.zeros((len(inputs), n, n)).to(device())
        procs = S.processors
        mapping = torch.FloatTensor([procs.index(task.processor)+1 for task in S.tasks]).reshape(-1, 1).to(device())
        for i, x in enumerate(inputs):
            priorities = torch.FloatTensor(x[-n:]).reshape(-1, 1).to(device())
            temp = (priorities < priorities.T) * (mapping == mapping.T)
            pm[i::] = temp
        return pm


class VectorHolisticFPAnalysis:
    def __init__(self, verbose=False, limit_factor=10, cache: ResultsCache = ResultsCache()):
        self.verbose = verbose
        self.limit_factor = limit_factor
        self.cache = cache
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
    def _analysis(priority_matrix, wcets, periods, deadlines, successors, wcrts, jitters, cache, verbose=False, limit=10):
        assert wcets.shape == periods.shape == deadlines.shape == successors.shape == wcrts.shape == jitters.shape
        assert wcets.shape[1] == 1
        assert priority_matrix.shape[1] == wcets.shape[0]

        # working priority matrix. initially with every scenario. scenarios will pop-out when they finish
        pm = priority_matrix.clone().detach()  # size (s, t, t)

        # remove scenearios for which we already have cached results
        pm = prune_known_scenarios(pm, cache)

        # there are t tasks, and s scenarios
        s, t, _ = pm.shape

        # the successors' matrix maps, for each task (row), which task is its successor (column)
        # this is a 2D matrix (t, t) (all scenarios have the same successors mapping)
        sm = successor_matrix(successors)

        # initialize response times. 3D matrix (s, t, 1)
        r_max = wcrts.clone().detach().reshape(1, t, 1).to(device()).repeat(s, 1, 1)
        # r_max = np.zeros((s, t, 1), dtype=np.float64)  # stores max WCRT found for each task and scenario
        # r = np.full_like(r_max, 0.)                    # temporarily stores iteration response times
        r = r_max.clone().detach()
        r_max_prev = r_max                                 # remember the wcrt currently found before this iteration

        # jitter matrix with current wcrt. 3D matrix (s, t, 1)
        j = jitter_matrix(sm, r_max)

        # wcrt limit for each task. if the provisional wcrt of any task reaches this, that scenario should stop
        r_limit = limit * deadlines  # 2D matrix (t, 1)

        if verbose:
            print("Starting Holistic FP analysis (vectorized, cached)")

        # response time convergence loop
        while pm.numel() > 0:
            r_max_prev = r_max

            # iterate p=1,2,... until w<=p*T
            p = 1

            # p_mask: true when a task w <= p*t (if true, task reached p-limit)
            p_mask = torch.full(r.shape, False).to(device())  # (s, t, 1)

            # p iterations loop
            while not torch.all(p_mask):
                # no more p when w <= p*T
                stop_p = p * periods

                # initialize w
                w = p*wcets.reshape(1, t, 1).repeat(pm.shape[0], 1, 1)
                # w = np.zeros_like(r)            # (s, t, 1)
                w_prev = torch.full_like(w, 0)   # (s, t, 1)

                # w convergence loop
                while not torch.allclose(w, w_prev):
                    w_prev = w

                    # Eq. (1) of "On the schedulability Analysis for Distributed Hard Real-Time Systems"
                    w = pm * torch.ceil((w + pm * torch.permute(j, (0, 2, 1))) / periods.T) @ wcets + p * wcets
                    w = w*~p_mask  # do not consider W's of tasks that have reached their p-limit already

                    # response time for this w. used to stop as early as possible if a task surpassed its WCRT limit
                    r = w - (p - 1) * periods + j

                    # save the highest response time seen until now
                    r_max = torch.maximum(r, r_max)

                    # identify the scenarios that have reached their r-limit
                    over = scenarios_over_limit(r_max, r_limit)  # (s) vector

                    if verbose:
                        print(f"p={p} w iteration, over={torch.sum(over)}")

                    # cache results of those scenarios that are over their limit
                    # remove those scenarios from necessary matrices
                    if torch.any(over):
                        cache_scenario_results(r_max, pm, over, cache)
                        pm, r_max, r, r_max_prev, j, p_mask, w, w_prev = (
                            remove_scenarios(over, pm, r_max, r, r_max_prev, j, p_mask, w, w_prev))

                    # update jitters
                    j = jitter_matrix(sm, r_max)

                p_mask = w <= stop_p  # tasks that have this true don't need to try more p's
                if verbose:
                    print(f"p={p} w converged, p_mask={torch.sum(p_mask)}")
                p += 1

            # at this point all the tasks have gone through all their p-values,
            # and r_max is updated with the maximum wcrt's found until now
            # now identify scenenarios with WCRT's that haven't changed from the previous iterations
            # those scenarios are considered finished -> cache their results

            converged = converged_scenarios(r_max, r_max_prev)  # (s) boolean vector
            cache_scenario_results(r_max, pm, converged, cache)
            pm, r_max, r, r_max_prev, j, p_mask = remove_scenarios(converged, pm, r_max, r, r_max_prev, j, p_mask)

            if verbose:
                print(f"p finished for all tasks, converged scenarios={torch.sum(converged)}")

        res = build_results_from_cache(priority_matrix, cache)  # 2D matrix (t, s)

        if verbose:
            print(f"All scenarios converged, analysis finished")
        return res

    def apply(self, system: System, scenarios: torch.Tensor = None):
        """
        Execute the vectorized Holistic analysis for FP systems in the given input system
        Optionally, additional priority scenarios can be provided. Array "scenarios" is a
        3D matrix with the additional priority scenarios, in which each plane is a 2D priority matrix
        """

        analysis.init_wcrt(system)
        wcets, periods, deadlines, successors, wcrts, jitters = get_vectors(system)

        n = len(wcets)  # number of tasks
        input_pm = system_priority_matrix(system).reshape(1, n, n)  # priority matrix of input system
        s = 0 if scenarios is None else scenarios.shape[0]          # number of additionl scenarios

        # pack all priority scenarios in the same matrix
        # the first scenario is for the priorities in the input system
        pm = torch.concatenate((input_pm, scenarios), axis=0) if s > 0 else input_pm

        # get response times for all scenarios. r is a 2D matrix (t, s+1)
        r = self._analysis(pm, wcets, periods, deadlines, successors, wcrts, jitters,
                           self.cache, verbose=self.verbose, limit=self.limit_factor)

        # set the response times of the first scenario as the wcrt of the input system
        # first scenario is the first column of s
        for task, wcrt in zip(system.tasks, r[:, 0]):
            task.wcrt = wcrt

        # save scenarios response times
        self._full_response_times = r
        self._scenarios_response_times = r[:, 1:] if s > 0 else None


def successor_matrix(succesors):
    """Builds the successor matrix from a flat successor list. Task id's start from 1"""
    l = succesors.numel()
    s = torch.zeros((l, l), dtype=torch.int, device=device())
    for i, v in enumerate(succesors):
        if v > -1:
            s[i, v-1] = 1
    return s


def jitter_matrix(smatrix, r):
    """Builds the jitter matrix. Assumes at most 1 successor per task"""
    j = smatrix.T.float() @ r
    return j


def priority_matrix(priorities):
    """Builds a 3D priority matrix for the given priority scenarios"""
    t, s = priorities.shape
    planes = priorities.ravel(order='F').reshape(s, t, 1)
    P = planes < planes.transpose((0, 2, 1))
    return P


def get_vectors(system: System):
    """Transform a system into vectors. The vectorized analysis is based on these vectors"""
    float_type = torch.float32
    int_type = torch.int32

    tasks = system.tasks
    t = len(tasks)
    wcets = torch.zeros((t, 1), dtype=float_type).to(device())
    periods = torch.zeros((t, 1), dtype=float_type).to(device())
    deadlines = torch.zeros((t, 1), dtype=float_type).to(device())
    successors = torch.zeros((t, 1), dtype=int_type).to(device())
    wcrts = torch.zeros((t, 1), dtype=float_type).to(device())
    jitters = torch.zeros((t, 1), dtype=float_type).to(device())

    taskmap = {task: i for i, task in enumerate(tasks)}

    for task, i in taskmap.items():
        wcets[i] = task.wcet
        periods[i] = task.period
        deadlines[i] = task.flow.deadline
        wcrts[i] = task.wcrt
        jitters[i] = task.jitter
        successors[i] = taskmap[task.successors[0]]+1 if task.successors else -1

    return wcets, periods, deadlines, successors, wcrts, jitters


def converged_scenarios(r, r_prev):
    """
    r and r_prev are (s, t, 1) matrices with the response times
    Returns a (s) boolean vector indicating which scenarios have converged (no change in response time)
    """
    s, _, _ = r.shape
    res = torch.all(r == r_prev, axis=1).reshape(s)
    return res


def scenarios_over_limit(r, r_limit):
    """
    returns (s) boolean vector indicating which scenarios have surpassed their limit
    A scenario surpasses his limit when any of its tasks surpasses its limit
    """
    s, _, _ = r.shape
    over = torch.any(r > r_limit, axis=1).reshape(s)
    return over


def extract_scenario_data(data: torch.Tensor, scenario: int) -> torch.Tensor:
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
    for s in torch.where(scenarios)[0]:
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
    res = torch.concatenate(data, axis=1)
    return res


def prune_known_scenarios(pm, cache: ResultsCache):
    """
    pm is the full 3D (s, t, t) priority matrix with every scenario
    cache is a results cache that maps a 2D priority matrix to a result
    Returns the pm matrix without the scenenarios for which we already know the results
    """
    s = pm.shape[0]  # number of scenarios
    removal = torch.full((s,), False)
    for i in range(s):
        key = extract_scenario_data(pm, i)
        removal[i] = cache.has_results(key)
    return remove_scenarios(removal, pm)[0]

