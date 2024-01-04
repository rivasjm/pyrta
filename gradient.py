from gradient_funcs import *
from model import System
import math
from assignment import extract_assignment, insert_assignment
import numpy as np
from vector import VectorHolisticFPBatchCosts


def apply_mask(mask: [bool], x: [float]):
    assert len(mask) == len(x)
    for i in range(len(x)):
        x[i] = x[i] * mask[i]


class StandardGradientDescent(GradientDescentFunction):
    def __init__(self,
                 extractor: Extractor,
                 cost_function: CostFunction,               # to compute cost at each iteration, not to compute gradient
                 stop_function: StopFunction,
                 gradient_function: GradientFunction,
                 update_function: UpdateFunction,
                 ref_cost_function: CostFunction = None,    # secondary cost function for logging, not to optimize
                 callback=None,
                 verbose=False):
        self.extractor = extractor
        self.cost_function = cost_function
        self.stop_function = stop_function
        self.gradient_function = gradient_function
        self.update_function = update_function
        self.ref_cost_function = ref_cost_function
        self.callback = callback
        self.verbose = verbose

    def reset(self):
        self.extractor.reset()
        self.cost_function.reset()
        self.stop_function.reset()
        self.gradient_function.reset()
        self.update_function.reset()
        self.ref_cost_function.reset()

    def apply(self, S: System) -> [float]:
        t = 1
        x = self.extractor.extract(S)  # initial input
        best = float('inf')     # best cost value, for logging purposes, not necessarily the cost of the solution
        ref_cost = None         # optional alternative cost value, just for logging purposes
        xb = x                  # best input, for logging purposes, not necessarily returned as solution

        while True:
            cost = self.cost_function.apply(S, x)
            if cost < best:
                best = cost
                xb = x

            if self.ref_cost_function:
                ref_cost = self.ref_cost_function.apply(S, x)

            if self.verbose:
                msg = f"iteration={t}: cost={cost:.3f} best={best:.3f}"
                if self.ref_cost_function:
                    msg += f" ref={ref_cost:.3f}"
                print(msg)

            if self.callback:
                self.callback(t, S, x, xb, cost, best, ref_cost)

            stop = self.stop_function.should_stop(S, x, cost, t)
            if stop:
                break

            nabla = self.gradient_function.apply(S, x)
            update = self.update_function.apply(S, x, nabla, t)
            x = [a + b for a, b in zip(x, update)]
            t = t + 1

            # insert into system, extract again to get x properly compressed
            self.extractor.insert(S, x)
            x = self.extractor.extract(S)

        solution = self.stop_function.solution(S)
        self.extractor.insert(S, solution)
        if self.verbose:
            print(f"Returning solution with cost={self.stop_function.solution_cost():.3f}")

        return solution


class DeadlineExtractor(Extractor):
    def extract(self, system: System) -> [float]:
        max_d = max([task.deadline for task in system.tasks])
        x = [t.deadline/max_d for t in system.tasks]
        return x

    def insert(self, system: System, x: [float]):
        max_d = max([task.deadline for task in system.tasks])
        tasks = system.tasks
        assert len(tasks) == len(x)
        for v, t in zip(x, tasks):
            t.deadline = v*max_d


class PriorityExtractor(Extractor):
    def extract(self, system: System) -> [float]:
        max_priority = max(map(lambda t: t.priority, system.tasks))
        r = [t.priority/max_priority for t in system.tasks]
        return r

    def insert(self, system: System, x: [float]):
        tasks = system.tasks
        assert len(tasks) == len(x)
        for v, t in zip(x, tasks):
            t.priority = v


class MappingPriorityExtractor(Extractor):
    def __init__(self):
        self.prio_extractor = PriorityExtractor()

    def reset(self):
        self.prio_extractor.reset()

    def extract(self, S: System) -> [float]:
        m_vector = [0.55 if task.processor == proc else 0.45 for task in S.tasks for proc in S.processors]
        p_vector = self.prio_extractor.extract(S)
        return m_vector + p_vector

    def insert(self, S: System, x: [float]) -> None:
        tasks = S.tasks
        procs = S.processors
        p = len(procs)
        t = len(tasks)
        assert len(x) == p*t + t

        # parse mapping values (fist p*t values)
        for i in range(t):
            sub = x[i*p: i*p+3]
            proc_index = sub.index(max(sub))
            tasks[i].processor = procs[proc_index]

        # parse priority values (last t values)
        self.prio_extractor.insert(S, x[-t:])


class InvslackCost(CostFunction):
    def __init__(self, extractor: Extractor, analysis):
        self.extractor = extractor
        self.analysis = analysis

    def reset(self):
        self.extractor.reset()

    def apply(self, S: System, x: [float]) -> float:
        a = extract_assignment(S)
        self.extractor.insert(S, x)
        self.analysis.apply(S)
        cost = max([(flow.wcrt - flow.deadline) / flow.deadline for flow in S.flows])
        insert_assignment(S, a)
        return cost


class StandardGradient(GradientFunction):
    def __init__(self, delta_function: DeltaFunction, batch_cost_function: BatchCostFunction):
        self.delta_function = delta_function
        self.batch_cost_function = batch_cost_function

    def reset(self):
        self.delta_function.reset()
        self.batch_cost_function.reset()

    @staticmethod
    def _gradient_inputs(x, deltas) -> [[float]]:
        ret = []
        for i in range(len(x)):
            vector = x[:]
            vector[i] += deltas[i]
            ret.append(vector)
            vector = x[:]
            vector[i] -= deltas[i]
            ret.append(vector)
        return ret

    @staticmethod
    def _gradient_from_costs(costs, deltas) -> [float]:
        gradient = [0] * int(len(costs) / 2)
        for i in range(len(gradient)):
            gradient[i] = (costs[2*i] - costs[2*i + 1]) / \
                          (2 * deltas[i % len(deltas)])
        return gradient

    def apply(self, S: System, x: [float]) -> [float]:
        deltas = self.delta_function.apply(S, x)
        inputs = self._gradient_inputs(x, deltas)
        costs = self.batch_cost_function.apply(S, inputs)  # one cost value per input
        gradient = self._gradient_from_costs(costs, deltas)
        return gradient


class AvgSeparationDelta(DeltaFunction):
    def __init__(self, factor=1.5):
        self.factor = factor

    def apply(self, S: System, x: [float]) -> [float]:
        seps = [abs(x[i + 1] - x[i]) for i in range(len(x) - 1)]
        return [self.factor * sum(seps) / len(seps)]*len(x)


class StandardStop(StopFunction):
    def __init__(self, limit=100):
        self.limit = limit
        self.best = float("inf")  # best cost value
        self.xb = None            # best solution

    def reset(self):
        self.best = float("inf")
        self.xb = None

    def should_stop(self, S: System, x: [float], cost: float, t: int) -> bool:
        if cost < self.best:
            self.best = cost
            self.xb = x
        return cost < 0 or t > self.limit

    def solution(self, S: System):
        return self.xb

    def solution_cost(self):
        return self.best


class FixedIterationsStop(StopFunction):
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.best = float("inf")  # best cost value
        self.xb = None            # best solution

    def reset(self):
        self.best = float("inf")
        self.xb = None

    def should_stop(self, S: System, x: [float], cost: float, t: int) -> bool:
        if cost < self.best:
            self.best = cost
            self.xb = x
        return t > self.iterations

    def solution(self, S: System):
        return self.xb

    def solution_cost(self):
        return self.best


class FixedAccumIterationsStop(StopFunction):
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.xs = []
        self.best = float('inf')

    def reset(self):
        self.xs = []
        self.best = float('inf')

    def should_stop(self, S: System, x: [float], cost: float, t: int) -> bool:
        self.xs.append(x)
        return t > self.iterations

    def solution(self, S: System):
        analysis = VectorHolisticFPBatchCosts()
        costs = analysis.apply(S, self.xs)
        index = np.argmin(costs)
        self.best = costs[index]
        return self.xs[index]

    def solution_cost(self):
        return self.best


class SequentialBatchCostFunction(BatchCostFunction):
    def __init__(self, cost_function: CostFunction):
        self.cost_function = cost_function

    def reset(self):
        self.cost_function.reset()

    def apply(self, S: System, inputs: [[float]]):
        costs = [self.cost_function.apply(S, x) for x in inputs]
        return costs


class GradientNoise(UpdateFunction):
    def __init__(self, lr, gamma=1.2, seed=1):
        self.lr = lr
        self.gamma = gamma
        self.seed = seed
        self.rng = None
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def apply(self, S: System, x: [float], nabla: [float], t: int) -> [float]:
        # noise added to the gradients helps with the optimization
        # the noise decays with the iterations
        # for big systems (e.g. 10x10x5), it is beneficial to reduce the noise added, so
        # I added a reducing factor to the noise (len(nabla)): bigger systems need less noise
        # for smaller systems, this reduction seems to not affect negatively
        std = self.lr / (1 + t + len(nabla)) ** self.gamma
        noise = self.rng.normal(0, std, len(nabla))
        for j in range(len(nabla)):
            nabla[j] += noise[j]
        return nabla


class Adam(UpdateFunction):
    def __init__(self, lr=3, beta1=0.9, beta2=0.999, epsilon=0.1):
        self.size = None
        self.m = None
        self.v = None
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.size = None
        self.m = None
        self.v = None

    def apply(self, S: System, x: [float], nabla: [float], t: int) -> [float]:
        if not self.size:
            self.size = len(nabla)
            self.m = [0]*self.size
            self.v = [0]*self.size

        updates = [0]*self.size
        for i in range(self.size):
            self.m[i] = self.beta1 * self.m[i] + (1 + self.beta1) * nabla[i]
            self.v[i] = self.beta2 * self.v[i] + (1 + self.beta2) * nabla[i] ** 2

            me = self.m[i] / (1 - self.beta1 ** t)
            ve = self.v[i] / (1 - self.beta2 ** t)

            updates[i] = -self.lr*me/(math.sqrt(ve)+self.epsilon)

        return updates


class NoisyAdam(UpdateFunction):
    def __init__(self, lr=3, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=0.9, seed=1):
        self.noise = GradientNoise(lr=lr, gamma=gamma, seed=seed)
        self.adam = Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)

    def reset(self):
        self.noise.reset()
        self.adam.reset()

    def apply(self, S: System, x: [float], nabla: [float], t: int) -> [float]:
        noisy_gradient = self.noise.apply(S, x, nabla, t)
        update = self.adam.apply(S, x, noisy_gradient, t)
        return update


def sigmoid(x):
  return 1 / (1 + math.exp(-x/100))


def map_range(value, old_min, old_max, new_min, new_max):
    return ((value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
