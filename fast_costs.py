from fast_analysis import func_w
from gradient_funcs import CostFunction, Extractor
from model import System, Task
from assignment import save_assignment, restore_assignment
from analysis import init_wcrt


class AvgSlopeCost(CostFunction):
    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def _slope(self, task: Task):
        x0 = 0
        x1 = task.flow.deadline
        y0 = func_w(task, 0, x0)
        y1 = func_w(task, 0, x1)
        return (y1-y0)/x1

    def apply(self, S: System, x: [float]) -> float:
        init_wcrt(S)
        slopes = [self._slope(t) for t in S.tasks]
        return sum(slopes)/len(slopes)


class SumSlopeCost(CostFunction):
    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def _slope(self, task: Task):
        x0 = 0
        x1 = task.flow.deadline
        y0 = func_w(task, 0, x0)
        y1 = func_w(task, 0, x1)
        return (y1-y0)/x1

    def apply(self, S: System, x: [float]) -> float:
        init_wcrt(S)
        slopes = [self._slope(t) for t in S.tasks]
        return sum(slopes)


class WorstFlowSlopeCost(CostFunction):
    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def _slope(self, task: Task):
        x0 = 0
        x1 = task.flow.deadline
        y0 = func_w(task, 0, x0)
        y1 = func_w(task, 0, x1)
        return (y1-y0)/x1

    def apply(self, S: System, x: [float]) -> float:
        init_wcrt(S)
        ret = 0
        for flow in S.flows:
            s = sum([self._slope(t) for t in flow.tasks])
            if s > ret:
                ret = s
        return ret
