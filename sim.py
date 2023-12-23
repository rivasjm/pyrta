import simpy
from model import System, Task, Flow


class Simulation:
    def __init__(self, system: System):
        self.system = system
        self.max_priority = max([t.priority for t in system.tasks])  # simpy has reverse priority logic (less is more)
        self.env = simpy.Environment()
        self.resources = {proc:simpy.PreemptiveResource(self.env, capacity=1) for proc in system.processors}

    def run(self, until):
        for flow in self.system.flows:
            self.env.process(self._process_flow(flow))
        self.env.run(until=until)

    def _process_flow(self, flow: Flow):
        print(f"{self.env.now}: flow {flow.name} RELEASED")
        for task in flow.tasks:
            yield self.env.process(self._process_task(task))
        print(f"{self.env.now}: flow {flow.name} FINISHED")

    def _process_task(self, task: Task):
        resource = self.resources[task.processor]
        priority = self.max_priority - task.priority
        remaining = task.wcet
        start = self.env.now  # I need to define a value here
        print(f"{self.env.now}: task {task.name} [{repr(task)} rem={remaining}] RELEASED")

        while remaining > 0:
            with resource.request(priority=priority) as req:
                try:
                    yield req
                    start = self.env.now
                    print(f"{self.env.now}: task {task.name} [{repr(task)} rem={remaining}] STARTED")

                    yield self.env.timeout(remaining)
                    remaining -= (self.env.now - start)
                    print(f"{self.env.now}: task {task.name} [{repr(task)} rem={remaining}] FINISHED")
                    assert remaining == 0

                except simpy.Interrupt:
                    remaining -= (self.env.now - start)
                    print(f"{self.env.now}: task {task.name} [{repr(task)} rem={remaining}] PREEMPTED")


def repr(task: Task) -> str:
    return f"proc={task.processor.name} prio={task.priority} wcet={task.wcet}"


if __name__ == '__main__':
    import examples, generator
    system = examples.get_palencia_system()
    generator.to_int(system)

    sim = Simulation(system)
    sim.run(100)

