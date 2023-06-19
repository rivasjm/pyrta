from gradient_funcs import *
from model import System
import math
from assignment import save_assignment, restore_assignment
import numpy as np


class StandardGradientDescent(GradientDescentFunction):
    def __init__(self,
                 extractor: Extractor,
                 cost_function: CostFunction,
                 stop_function: StopFunction,
                 gradient_function: GradientFunction,
                 update_function: UpdateFunction,
                 verbose = False):
        self.extractor = extractor
        self.cost_function = cost_function
        self.stop_function = stop_function
        self.gradient_function = gradient_function
        self.update_function = update_function
        self.verbose = verbose

    def apply(self, S: System) -> [float]:
        t = 1
        x = self.extractor.extract(S)
        best = float('inf')     # best cost value
        xb = x                  # best input
        while True:
            cost = self.cost_function.apply(S, x)
            if cost < best:
                best = cost
                xb = x

            if self.verbose:
                print(f"iteration={t}: cost={cost} best={best}")

            stop = self.stop_function.apply(S, x, cost, t)
            if stop:
                break

            nabla = self.gradient_function.apply(S, x)
            update = self.update_function.apply(S, x, nabla, t)
            x = [a + b for a, b in zip(x, update)]
            t = t + 1

        return xb


class DeadlineExtractor(Extractor):
    def extract(self, system: System) -> [float]:
        r = [t.deadline for t in system.tasks]
        return r

    def insert(self, system: System, x: [float]):
        tasks = system.tasks
        assert len(tasks) == len(x)
        for v, t in zip(x, tasks):
            t.deadline = v


class PriorityExtractor(Extractor):
    def extract(self, system: System) -> [float]:
        r = [t.priority for t in system.tasks]
        return r

    def insert(self, system: System, x: [float]):
        tasks = system.tasks
        assert len(tasks) == len(x)
        for v, t in zip(x, tasks):
            t.priority = v


class InvslackCost(CostFunction):
    def __init__(self, extractor: Extractor, analysis):
        self.extractor = extractor
        self.analysis = analysis

    def apply(self, S: System, x: [float]) -> float:
        save_assignment(S)
        self.extractor.insert(S, x)
        self.analysis.apply(S)
        cost = max([(flow.wcrt - flow.deadline) / flow.deadline for flow in S.flows])
        restore_assignment(S)
        return cost


class StandardGradient(GradientFunction):
    def __init__(self, delta_function: DeltaFunction, batch_cost_function: BatchCostFunction):
        self.delta_function = delta_function
        self.batch_cost_function = batch_cost_function

    def _gradient_inputs(self, x, deltas) -> [[float]]:
        ret = []
        for i in range(len(x)):
            vector = x[:]
            vector[i] += deltas[i % len(deltas)]
            ret.append(vector)
            vector = x[:]
            vector[i] -= deltas[i % len(deltas)]
            ret.append(vector)
        return ret

    def _gradient_from_costs(self, costs, deltas) -> [float]:
        gradient = [0] * int(len(costs) / 2)
        for i in range(len(gradient)):
            gradient[i] = (costs[i] - costs[i + 1]) / \
                          (2 * deltas[i % len(deltas)])
        return gradient

    def apply(self, S: System, x: [float]) -> [float]:
        deltas = self.delta_function.apply(S, x)
        inputs = self._gradient_inputs(x, deltas)
        costs = self.batch_cost_function.apply(S, inputs)  # one cost value per input
        gradient = self._gradient_from_costs(costs, deltas)
        return gradient


class AvgSeparationDelta(DeltaFunction):
    def __init__(self, factor=1):
        self.factor = factor

    def apply(self, S: System, x: [float]) -> [float]:
        ordered = sorted(x)
        dist = 0
        count = 0
        for i in range(len(ordered) - 1):
            a = ordered[i]
            b = ordered[i+1]
            if math.isclose(a, b):
                continue
            dist += abs(a - b)
            count += 1
        return [self.factor * dist/count] if count > 0 else 1


class StandardStop(StopFunction):
    def __init__(self, limit=100):
        self.limit = limit

    def apply(self, S: System, x: [float], cost: float, t: int) -> bool:
        return cost < 0 or t > self.limit


class SequentialBatchCostFunction(BatchCostFunction):
    def __init__(self, cost_function: CostFunction):
        self.cost_function = cost_function

    def apply(self, S: System, inputs: [[float]]):
        costs = [self.cost_function.apply(S, x) for x in inputs]
        return costs


class GradientNoise(UpdateFunction):
    def __init__(self, lr, gamma=0.9, seed=42):
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
        # I added a reducing factor to the noise (len(coeffs)): bigger systems -> less noise
        # for smaller systems, this reduction seems to not affect negatively
        std = self.lr / (1 + t + len(nabla)) ** self.gamma
        noise = self.rng.normal(0, std, len(nabla))
        for j in range(len(nabla)):
            nabla[j] += noise[j]
        return nabla


class Adam(UpdateFunction):
    def __init__(self, lr=0.2, beta1=0.9, beta2=0.999, epsilon=10**-8):
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

            updates[i] = self.lr*me/(math.sqrt(ve)+self.epsilon)

        return updates


class NoisyAdam(UpdateFunction):
    def __init__(self, lr=0.2, beta1=0.9, beta2=0.999, epsilon=10**-8, gamma=0.9, seed=42):
        self.noise = GradientNoise(lr=lr, gamma=gamma, seed=seed)
        self.adam = Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)

    def reset(self):
        self.noise.reset()
        self.adam.reset()

    def apply(self, S: System, x: [float], nabla: [float], t: int) -> [float]:
        noisy_gradient = self.noise.apply(S, x, nabla, t)
        update = self.adam.apply(S, x, noisy_gradient, t)
        return update