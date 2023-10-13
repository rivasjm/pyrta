from model import System, Processor, Flow, Task, TaskType
import textwrap


def processing_resource_name(processor: Processor) -> str:
    return processor.name


def scheduler_name(processor: Processor) -> str:
    return "sched_" + processor.name


def scheduler_host(processor: Processor) -> str:
    return processing_resource_name(processor)


def scheduler_policy_type(processor: Processor) -> str:
    return processor.sched.value


def scheduling_server_name(task: Task) -> str:
    return "ss_" + task.name


def scheduling_server_scheduler(task: Task) -> str:
    return scheduler_name(task.processor)


def operation_name(task: Task) -> str:
    return "oper_" + task.name


def transaction_name(flow: Flow) -> str:
    return flow.name


def external_event_name(flow: Flow) -> str:
    return "ee_" + transaction_name(flow)


def internal_event_name(task: Task) -> str:
    return "ie_" + task.name


def reverse_internal_event_name(event_name) -> str:
    return event_name.lstrip("ie_")


def input_event_name(task: Task) -> str:
    return internal_event_name(task.predecessors[0]) if task.predecessors else external_event_name(task.flow)


def indent(level: int) -> str:
    return "    "*level;


class ProcessorAdapter:
    def __init__(self, processor: Processor):
        self.processor = processor

    def processing_resource(self) -> str:
        literal = textwrap.dedent(f"""\
        Processing_Resource (
           Type                   => Regular_Processor,
           Name                   => {processing_resource_name(self.processor)},
           Max_Interrupt_Priority => 512,
           Min_Interrupt_Priority => 512,
           Worst_ISR_Switch       => 0.00,
           Avg_ISR_Switch         => 0.00,
           Best_ISR_Switch        => 0.00,
           Speed_Factor           => 1.00)""")
        return literal

    def scheduler(self) -> str:
        literal = textwrap.dedent(f"""\
        Scheduler (
           Type            => Primary_Scheduler,
           Name            => {scheduler_name(self.processor)},
           Host            => {scheduler_host(self.processor)},
           Policy          => 
              ( Type                 => {self.processor.sched.value},
                Worst_Context_Switch => 0.00,
                Avg_Context_Switch   => 0.00,
                Best_Context_Switch  => 0.00,
                Max_Priority         => 511,
                Min_Priority         => 1))""")
        return literal


class TaskAdapter:
    def __init__(self, task: Task):
        self.task = task

    def scheduling_server(self):
        if self.task.type != TaskType.Activity:
            return ""
        literal = textwrap.dedent(f"""\
        Scheduling_Server (
           Type                       => Regular,
           Name                       => {scheduling_server_name(self.task)},
           Server_Sched_Parameters    => \n{textwrap.indent(self.server_sched_parameter(), indent(4))},
           Scheduler                  => {scheduler_name(self.task.processor)})""")
        return literal

    def server_sched_parameter(self):
        literal = textwrap.dedent(f"""\
        (Type         => {self.task.processor.sched.value}_Policy,
         The_Priority => {self.task.priority},
         Preassigned  => NO)""")
        return literal

    def operation(self):
        if self.task.type != TaskType.Activity:
            return ""
        literal = textwrap.dedent(f"""\
        Operation (
           Type                       => Simple,
           Name                       => {operation_name(self.task)},
           Worst_Case_Execution_Time  => {self.task.wcet},
           Best_Case_Execution_Time   => {self.task.bcet})""")
        return literal

    def internal_event(self):
        if not self.task.successors and self.task.flow.deadline:
            literal = textwrap.dedent(f"""\
            (Type => Regular,
             Name => {internal_event_name(self.task)},
             Timing_Requirements  => 
                (Type             => Hard_Global_Deadline,
                 Deadline         => {self.task.flow.deadline},
                 Referenced_Event => {external_event_name(self.task.flow)}))""")
            return literal
        else:
            literal = textwrap.dedent(f"""\
                        (Type => Regular,
                         Name => {internal_event_name(self.task)})""")
            return literal

    def event_handler(self):
        if self.task.type == TaskType.Activity:
            return self.event_handler_activity()
        elif self.task.type == TaskType.Offset:
            return self.event_handler_offset()
        elif self.task.type == TaskType.Delay:
            return self.event_handler_delay()

    def event_handler_activity(self):
        literal = textwrap.dedent(f"""\
        (Type                   => Activity,
         Input_Event            => {input_event_name(self.task)},
         Output_Event           => {internal_event_name(self.task)},
         Activity_Operation     => {operation_name(self.task)},
         Activity_Server        => {scheduling_server_name(self.task)})""")
        return literal

    def event_handler_offset(self):
        literal = textwrap.dedent(f"""\
        (Type                   => Offset,
         Input_Event            => {input_event_name(self.task)},
         Output_Event           => {internal_event_name(self.task)},
         Delay_Max_Interval     => {self.task.wcet},
         Delay_Min_Interval     => {self.task.bcet},
         Referenced_Event       => {external_event_name(self.task.flow)})""")
        return literal

    def event_handler_delay(self):
        literal = textwrap.dedent(f"""\
        (Type                   => Delay,
         Input_Event            => {input_event_name(self.task)},
         Output_Event           => {internal_event_name(self.task)},
         Delay_Max_Interval     => {self.task.wcet},
         Delay_Min_Interval     => {self.task.bcet})""")
        return literal


class FlowAdapter:
    def __init__(self, flow: Flow):
        self.flow = flow
        self.tasks_adapters = list(map(lambda t: TaskAdapter(t), self.flow.tasks))

    def external_event(self):
        literal = textwrap.dedent(f"""\
        (Type       => Periodic,
         Name       => {external_event_name(self.flow)},
         Period     => {self.flow.period},
         Max_Jitter => 0,
         Phase      => 0)""")
        return literal

    def transaction(self):
        internal_events = textwrap.indent(",\n".join(map(lambda t: t.internal_event(), self.tasks_adapters)), indent(4))
        event_handlers = textwrap.indent(",\n".join(map(lambda t: t.event_handler(), self.tasks_adapters)), indent(4))

        literal = textwrap.dedent(f"""\
        Transaction (
            Type            => Regular,
            Name            => {transaction_name(self.flow)},
            External_Events => (\n{textwrap.indent(self.external_event(), indent(4))}),
            Internal_Events => (\n{internal_events}),
            Event_Handlers  => (\n{event_handlers}))""")
        return literal


class SystemAdapter:
    def __init__(self, system: System):
        self.system = system
        self.procs = list(map(lambda p: ProcessorAdapter(p), self.system.processors))
        self.tasks = list(map(lambda t: TaskAdapter(t), self.system.tasks))
        self.flows = list(map(lambda f: FlowAdapter(f), self.system.flows))

    def serialize(self) -> str:
        pr = "\n\n".join(map(lambda p: p.processing_resource()+";", self.procs))
        scheds = "\n\n".join(map(lambda p: p.scheduler() + ";", self.procs))
        opers = "\n\n".join([t.operation()+";" for t in self.tasks if t.task.type == TaskType.Activity])
        # opers = "\n\n".join(map(lambda t: t.operation() + ";", self.tasks))
        ss = "\n\n".join([t.scheduling_server()+";" for t in self.tasks if t.task.type == TaskType.Activity])
        # ss = "\n\n".join(map(lambda t: t.scheduling_server() + ";", self.tasks))
        trans = "\n\n".join(map(lambda f: f.transaction() + ";", self.flows))
        literal = f"""{pr}\n\n{scheds}\n\n{opers}\n\n{ss}\n\n{trans}"""
        return literal


def export(system: System, file):
    adapter = SystemAdapter(system)
    txt = adapter.serialize()
    with open(file, "w") as f:
        f.write(txt)


if __name__ == '__main__':
    import examples
    system = SystemAdapter(examples.get_simple_gpu())
    print(system.serialize())

