import simpy
from model import System, Task, Flow
import math


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
        resource = self.resources[task.processor]
        priority = self.max_priority - task.priority
        remaining = task.wcet
        start = self.env.now  # I need to define a value here
        release_time = self.env.now
        self._print(f"{self.env.now}: task {task.name} [{repr(task)} rem={remaining}] RELEASED")

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
                    self.res.add_task_interval(task, task.processor, start, self.env.now)

        finish_time = self.env.now
        self.res.add_task_result(task, flow_release_time, release_time, finish_time)


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

    def task_wort(self, task):
        return wort(task, self.task_results)

    def flow_wort(self, flow):
        ret = wort(flow, self.flow_results)
        last_task = flow.tasks[len(flow.tasks)-1]
        return ret == wort(last_task, self.task_results)


def wort(key, data):
    if key not in data:
        return None
    rts = map(lambda r: r[1] - r[0], data[key])
    return max(rts)


def add_result(dic, key, element):
    if key not in dic:
        dic[key] = []
    dic[key].append(element)


def repr(task: Task) -> str:
    return f"proc={task.processor.name} prio={task.priority} wcet={task.wcet}"



if __name__ == '__main__':
    import examples, generator, analysis
    system = examples.get_palencia_system()
    generator.to_int(system)

    holistic = analysis.HolisticFPAnalysis(reset=False)
    holistic.apply(system)

    sim = Simulation(system, verbose=False)
    sim.run(math.lcm(*[f.period for f in system.flows]))
    print(sim.results.repr())
    print(analysis.repr_wcrts(system))
    print(sim.results.pessimism())
