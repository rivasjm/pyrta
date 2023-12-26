import sys
import warnings
from enum import Enum


class System:
    def __init__(self):
        self.name = None
        self.flows = list()
        self.processors = list()

    def add_flows(self, *flows):
        self.flows += flows
        for flow in flows:
            flow.system = self

    def add_procs(self, *procs):
        for proc in procs:
            self.processors.append(proc)
            proc.system = self

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.flows[item]
        elif isinstance(item, str):
            match = [flow for flow in self.flows if flow.name == item]
            return match[0] if len(match) > 0 else None
        return None

    def apply(self, function):
        function.apply(self)
        return self

    @property
    def tasks(self):
        return [task for flow in self.flows for task in flow]

    def processor(self, name):
        return next((p for p in self.processors if p.name == name), None)

    def is_schedulable(self):
        return all(map(lambda f: f.is_schedulable(), self))

    @property
    def utilization(self):
        us = [proc.utilization for proc in self.processors]
        return sum(us)/len(us)

    @property
    def max_utilization(self):
        us = [proc.utilization for proc in self.processors]
        return max(us)

    @property
    def slack(self):
        slacks = [flow.slack for flow in self.flows]
        return min(slacks) if len(slacks) > 0 else sys.float_info.min

    @property
    def avg_flow_wcrt(self):
        return sum(map(lambda f: f.wcrt, self.flows))/len(self.flows)

    def __repr__(self):
        return "\n".join(map(lambda f: str(f), self.flows))


class SchedulerType(Enum):
    FP = "Fixed_Priority"
    EDF = "EDF"


class Processor:
    def __init__(self, name, sched=SchedulerType.FP, local=True):
        self.system = None
        self.name = name
        self.sched = sched
        self.local = local      # local=True means no clock sync (EDF-L). No meaning in FP.

    def __repr__(self):
        return f"{self.name}"

    @property
    def tasks(self):
        return [task for flow in self.system for task in flow
                if task.processor == self] if self.system else []

    @property
    def utilization(self):
        u = [task.wcet / task.period for task in self.tasks]
        return sum(u)


class Flow:
    def __init__(self, name, period, deadline):
        self.system = None
        self.name = name
        self.period = period
        self.deadline = deadline
        self.tasks = list()
        self.phase = 0  # for simulation

    def add_tasks(self, *tasks):
        self.tasks += tasks
        for task in tasks:
            task.flow = self

    def __repr__(self):
        ts = " ".join(map(lambda t: str(t), self.tasks))
        return f"{self.period:.2f} : {ts} : {self.deadline:.2f})"

    @property
    def wcrt(self):
        return self.tasks[-1].wcrt

    @property
    def slack(self):
        if self.wcrt:
            return (self.deadline - self.wcrt)/self.deadline
        else:
            return float("-inf")

    def predecessors(self, task):
        i = self.tasks.index(task)
        return [self.tasks[i-1]] if i > 0 else []

    def successors(self, task):
        i = self.tasks.index(task)
        return [self.tasks[i + 1]] if i < len(self.tasks)-1 else []

    def all_successors(self, task):
        i = self.tasks.index(task)
        return self.tasks[i+1:]

    def is_schedulable(self):
        return self.wcrt and self.wcrt <= self.deadline

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.tasks.__getitem__(item)
        elif isinstance(item, str):
            match = [task for task in self.tasks if task.name == item]
            return match[0] if len(match) > 0 else None
        return None


class TaskType(Enum):
    Activity = "Activity"
    Offset = "Offset"
    Delay = "Delay"


class Task:
    def __init__(self,
                 name: str,
                 wcet: float,
                 processor: Processor = None,
                 type: TaskType = TaskType.Activity,
                 priority: int = 1,
                 bcet: float = 0,
                 deadline=0):
        self.flow = None
        self.name = name
        self.wcet = wcet
        self.processor: Processor = processor
        self.type = type

        self.priority: int = priority
        self.deadline = deadline
        self.wcrt = None
        self.bcet = bcet

    def __repr__(self):
        return f"{self.name} ({self.processor.name if self.processor else None},{self.utilization:.2f})"

    @property
    def utilization(self):
        return self.wcet / self.period

    @property
    def period(self):
        return self.flow.period

    @property
    def sched(self) -> SchedulerType:
        return self.processor.sched

    @property
    def successors(self):
        return self.flow.successors(self)

    @property
    def predecessors(self):
        return self.flow.predecessors(self)

    @property
    def is_last(self):
        return len(self.successors) == 0

    @property
    def all_successors(self):
        return self.flow.all_successors(self)

    @property
    def jitter(self):
        wcrts = list(map(lambda t: t.wcrt, self.flow.predecessors(self)))
        return max(wcrts) if len(wcrts) > 0 else 0

    def copy(self):
        new_task = Task(name=self.name, wcet=self.wcet, processor=None,
                        priority=self.priority)
        new_task.wcrt = self.wcrt
        new_task.deadline = self.deadline
        return new_task


def save_attrs(elements: [], attrs: [str], key="_saved_") -> None:
    for element in elements:
        for attr in attrs:
            if hasattr(element, attr):
                value = getattr(element, attr)
                setattr(element, key + attr, value)
            else:
                warnings.warn(f"Warning, element {element} does not have attribute {attr}")


def restore_attrs(elements: [], attrs: [str], key="_saved_") -> None:
    for element in elements:
        for attr in attrs:
            if hasattr(element, key + attr):
                value = getattr(element, key + attr)
                setattr(element, attr, value)


def is_FP(system: System) -> bool:
    return all(proc.sched == SchedulerType.FP for proc in system.processors)


def is_EDF(system: System) -> bool:
    return all(proc.sched == SchedulerType.EDF for proc in system.processors)
