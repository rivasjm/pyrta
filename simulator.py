import simpy

import model
from model import System, Task, Flow
import math
import random


class Simulation:
    def __init__(self, system: System, verbose=True):
        self.system = system
        self.verbose = verbose
        self.max_priority = max([t.priority for t in system.tasks])  # simpy has reverse priority logic (less is more)
        self.env = simpy.Environment()
        self.resources = {proc:simpy.PreemptiveResource(self.env, capacity=1) for proc in system.processors}
        self.res = SimResults(system)

    def run(self, until):
        self._print(f"Running simulation until {until}")
        for flow in self.system.flows:
            self.env.process(self._flow_dispatcher(flow))
        self.env.run(until=until)

    @property
    def results(self):
        return self.res

    def _print(self, msg):
        if self.verbose:
            print(msg)

    def _flow_dispatcher(self, flow: Flow):
        if flow.phase > 0:
            yield self.env.timeout(delay=flow.phase)
        while True:
            self.env.process(self._process_flow(flow))
            yield self.env.timeout(flow.period)

    def _process_flow(self, flow: Flow):
        release_time = self.env.now
        self._print(f"{self.env.now}: flow {flow.name} [T={flow.period} D={flow.deadline}] RELEASED")
        for task in flow.tasks:
            yield self.env.process(self._process_task(task, release_time))

        finish_time = self.env.now
        self._print(f"{self.env.now}: flow {flow.name} [T={flow.period} D={flow.deadline}] FINISHED")
        self.res.add_flow_result(flow, release_time, finish_time)

    def _process_task(self, task: Task, flow_release_time):
        assert task.type in model.TaskType

        if task.type == model.TaskType.Delay:
            self._print(f"{self.env.now}: DELAY of {task.wcet}")
            yield self.env.timeout(delay=task.wcet)

        elif task.type == model.TaskType.Offset:
            self._print(f"{self.env.now}: OFFSET unitl {task.wcet}")
            delay = flow_release_time + task.wcet - self.env.now
            yield self.env.timeout(delay = max(0, delay))

        else:
            release_time = self.env.now
            resource = self.resources[task.processor]
            priority = self._process_priority(task, flow_release_time, release_time)
            remaining = task.wcet
            self._print(f"{self.env.now}: task {task.name} [{repr(task)} rem={remaining} prio={priority}] RELEASED")
            start = self.env.now  # I need to define a value here (outside try)

            while remaining > 0:
                with resource.request(priority=priority) as req:
                    try:
                        yield req
                        start = self.env.now
                        self._print(f"{self.env.now}: task {task.name} [{repr(task)} rem={remaining}] STARTED")
                        yield self.env.timeout(remaining)
                        remaining -= (self.env.now - start)
                        self._print(f"{self.env.now}: task {task.name} [{repr(task)} rem={remaining}] FINISHED")
                        assert remaining == 0

                        self.res.add_task_interval(task, task.processor, start, self.env.now)

                    except simpy.Interrupt:  # task was preempted
                        remaining -= (self.env.now - start)
                        self._print(f"{self.env.now}: task {task.name} [{repr(task)} rem={remaining}] PREEMPTED")
                        if self.env.now > start:  # avoid intervals in which start==end
                            self.res.add_task_interval(task, task.processor, start, self.env.now)

            finish_time = self.env.now
            self.res.add_task_result(task, flow_release_time, release_time, finish_time)

    def _process_priority(self, task: Task, flow_release_time, task_release_time):
        assert task.processor.sched in model.SchedulerType
        if task.processor.sched == model.SchedulerType.FP:
            return self.max_priority - task.priority
        elif task.processor.sched == model.SchedulerType.EDF and task.processor.local:
            return task_release_time + task.deadline
        elif task.processor.sched == model.SchedulerType.EDF and not task.processor.local:
            return flow_release_time + task.deadline


class SimResults:
    def __init__(self, system):
        self.system = system
        self.task_results = {}      # task=[(flow_release_time, finish_time, release_time), ...]
        self.task_intervals = {}    # task=[(processor, start_time, end_time), ...]
        self.flow_results = {}      # flow=[(release_time, finish_time), ...]

    def repr(self):
        lines = []
        for flow in self.system.flows:
            worts = " ".join([f"{self.task_wort(task):.2f}" for task in flow.tasks])
            line = f"{flow.period}: {worts} : {flow.deadline}"
            lines.append(line)
        return "\n".join(lines) + "\n"

    def pessimism(self):
        pairs = [t.wcrt/self.task_wort(t) for t in self.system.tasks]  # wcrt/wort
        avg = sum(pairs)/len(pairs)
        return min(pairs), avg, max(pairs)

    def add_task_result(self, task, flow_release_time, release_time, finish_time):
        add_result(self.task_results, task, (flow_release_time, finish_time, release_time))

    def add_task_interval(self, task, processor, start_time, end_time):
        add_result(self.task_intervals, task, (processor, start_time, end_time))

    def add_flow_result(self, flow, release_time, finish_time):
        add_result(self.flow_results, flow, (release_time, finish_time))

    def task_rts(self, task):
        rts = list(map(lambda r: r[1] - r[0], self.task_results[task]))
        return rts

    def task_wort(self, task):
        return wort(task, self.task_results)

    def flow_wort(self, flow):
        ret = wort(flow, self.flow_results)
        last_task = flow.tasks[len(flow.tasks)-1]
        assert ret == wort(last_task, self.task_results)
        return ret

    def intervals(self, task):
        return [(i[1],i[2]) for i in self.task_intervals[task]]


def wort(key, data):
    if key not in data:
        return None
    rts = map(lambda r: r[1] - r[0], data[key])
    return max(rts)


def add_result(dic, key, element):
    if key not in dic:
        dic[key] = []
    dic[key].append(element)


class SimRandomizer:
    def __init__(self, system: System, seed=0, callback=None, verbose=False):
        self.system = system
        self.seed = seed
        self.verbose = verbose
        self.rnd = random.Random(seed)
        self.callback = callback
        self.phases = [flow.phase for flow in self.system.flows]
        self.results = []

    def run(self, until, iterations):
        for i in range(iterations):
            if self.verbose:
                print(".", end="")
            self._randomize_flow_phases(self.system.flows)
            sim = Simulation(self.system, verbose=False)
            sim.run(until=until)
            if self.callback:
                self.callback(sim, i, until, self.seed)
            self.results.append(sim.results)
        if self.verbose:
            print("")

        # restore flow phases
        for flow, phase in zip(self.system.flows, self.phases):
            flow.phase = phase

    def _randomize_flow_phases(self, flows):
        for flow in flows:
            phase = self.rnd.randrange(0, flow.period)
            flow.phase = phase


def max_flow_wort(flow, results: []):
    ret = 0
    for res in results:
        w = res.flow_wort(flow)
        if w > ret:
            ret = w
    return ret


def max_task_wort(task, results: []):
    ret = 0
    for res in results:
        w = res.task_wort(task)
        if w > ret:
            ret = w
    return ret


def repr(task: Task) -> str:
    return f"proc={task.processor.name} prio={task.priority} wcet={task.wcet}"


def hyperperiod(system: System):
    return math.lcm(*[f.period for f in system.flows])


if __name__ == '__main__':
    import examples, generator, analysis
    system = examples.get_palencia_system()
    generator.to_int(system)

    holistic = analysis.HolisticFPAnalysis(reset=False)
    holistic.apply(system)

    sim = Simulation(system, verbose=False)
    sim.run(hyperperiod(system))
    print(sim.results.repr())
    print(analysis.repr_wcrts(system))
    print(sim.results.pessimism())
