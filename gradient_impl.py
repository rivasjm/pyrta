from gradient import *
from model import System
import math
from assignment import save_assignment, restore_assignment
import numpy as np


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
        self.extractor.apply(S, x)
        self.analysis.apply(S)
        cost = max([(flow.wcrt - flow.deadline) / flow.deadline for flow in S.flows])
        restore_assignment(S)
        return cost


class StandardGradient(GradientFunction):
    def __init__(self, deltas, batch_analyzer):
        self.deltas = deltas
        self.batch_analyzer = batch_analyzer

    def _gradient_inputs(self, x) -> [[float]]:
        ret = []
        for i in range(len(x)):
            vector = x[:]
            vector[i] += self.deltas[i % len(self.deltas)]
            ret.append(vector)
            vector = x[:]
            vector[i] -= self.deltas[i % len(self.deltas)]
            ret.append(vector)
        return ret

    def apply(self, S: System, x: [float]) -> [float]:
        inputs = self._gradient_inputs(x)
        wcrts = self.batch_analyzer(S, inputs)
        





def gradient_from_costs(costs: [float], deltas) -> [float]:
    gradient = [0]*(len(costs)/2)
    for i in range(len(gradient)):
        gradient[i] = (costs[i] - costs[i+1])/(2*deltas[i % len(deltas)])


class SequentialGradientCompute:
    def __init__(self, extractor, cost_function, deltas, analysis):
        self.extractor = extractor
        self.cost_function = cost_function
        self.deltas = deltas
        self.analysis = analysis

    def compute(self, system: System, x: [float]):
        vectors = gradient_vectors(x, self.deltas)
        costs = []
        for vector in vectors:
            cost = self.cost_function(system, vector, self.extractor, self.analysis)
            costs.append(cost)
        nabla = gradient_from_costs(costs)
        return nabla


class GradientNoise:
    def __init__(self, lr, gamma, seed=42):
        self.lr = lr
        self.gamma = gamma
        self.seed = seed
        self.rng = None
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def apply(self, gradient: [float], iteration: int):
        # noise added to the gradients helps with the optimization
        # the noise decays with the iterations
        # for big systems (e.g. 10x10x5), it is beneficial to reduce the noise added, so
        # I added a reducing factor to the noise (len(coeffs)): bigger systems -> less noise
        # for smaller systems, this reduction seems to not affect negatively
        std = self.lr / (1 + iteration + len(gradient)) ** self.gamma
        noise = self.rng.normal(0, std, len(gradient))
        for j in range(len(gradient)):
            gradient[j] += noise[j]


class Adam:
    def __init__(self, lr=0.2, beta1=0.9, beta2=0.999, epsilon=10**-8, gamma=0.9, seed=1):
        self.size = None
        self.m = None
        self.v = None
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gamma = gamma
        self.seed = seed
        self.rng = None
        self.reset()

    def reset(self):
        self.size = None
        self.m = None
        self.v = None

    def step(self, gradient, iteration) -> [float]:
        if not self.size:
            self.size = len(gradient)
            self.m = [0]*self.size
            self.v = [0]*self.size

        updates = [0]*self.size
        for i in range(self.size):
            self.m[i] = self.beta1 * self.m[i] + (1 + self.beta1) * gradient[i]
            self.v[i] = self.beta2 * self.v[i] + (1 + self.beta2) * gradient[i] ** 2

            me = self.m[i] / (1 - self.beta1 ** iteration)
            ve = self.v[i] / (1 - self.beta2 ** iteration)

            updates[i] = -self.lr*me/(math.sqrt(ve)+self.epsilon)

        return updates