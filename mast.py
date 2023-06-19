from model import *
from enum import Enum
import uuid
import os.path
import subprocess
from bs4 import BeautifulSoup
import tempfile
import config


class MastAnalysis(Enum):
    HOLISTIC = "holistic"
    OFFSET = "offset_based_approx"
    OFFSET_PR = "offset_based_approx_w_pr"


class MastAssignment(Enum):
    NONE = None
    PD = "pd"
    HOSPA = "hospa"
    SA = "annealing"


# region MAST Wrappers

class MastWrapper:
    def __init__(self, analysis: MastAnalysis, assignment=MastAssignment.NONE,
                 limit_factor=100):
        self.analysis = analysis
        self.assignment = assignment
        self.limit_factor = limit_factor

    def apply(self, system: System) -> None:
        analyze(system, self.analysis, self.assignment, limit=self.limit_factor)


class MastHolisticAnalysis(MastWrapper):
    def __init__(self, assignment=MastAssignment.NONE, limit_factor=100):
        super().__init__(MastAnalysis.HOLISTIC, assignment=assignment, limit_factor=limit_factor)


class MastOffsetAnalysis(MastWrapper):
    def __init__(self, assignment=MastAssignment.NONE, limit_factor=100):
        super().__init__(MastAnalysis.OFFSET, assignment=assignment, limit_factor=limit_factor)


class MastOffsetPrecedenceAnalysis(MastWrapper):
    def __init__(self, assignment=MastAssignment.NONE, limit_factor=100):
        super().__init__(MastAnalysis.OFFSET_PR, assignment=assignment, limit_factor=limit_factor)


# endregion


# region MAST Analysis

def analyze(system: System, analysis: MastAnalysis, assignment: MastAssignment = MastAssignment.NONE, limit=1e100):

    # create random temporary file names for this analysis, will be removed afterwards
    temp_dir = tempfile.TemporaryDirectory()
    name = str(uuid.uuid1())
    input = os.path.abspath(os.path.join(temp_dir.name, name + ".txt"))
    output = os.path.abspath(os.path.join(temp_dir.name, name + "-out.xml"))
    preserve = False

    try:
        # make sure priorities are correct for mast (integers higher than 0)
        sanitize_priorities(system)

        # export system to a file with mast format
        export(system, input)

        # analyze with mast, capture results
        schedulable, results = run(analysis, assignment, input, output, limit)

        # save wcrts into the system
        for task in system.tasks:
            task.wcrt = results[task.name] if task.name in results else limit

        # sanity check: system schedulability must match
        if system.is_schedulable() != schedulable:
            print("assertion error: " + input)
            preserve = True

    finally:
        # clean-up process: restore original unsanitized priorities, remove temporary files
        desanitize_priorities(system)
        if not preserve:
            temp_dir.cleanup()


def clear_files(*files):
    for file in files:
        if os.path.isfile(file):
            while True:
                try:
                    os.remove(file)
                    break
                except PermissionError:
                    pass


def run(analysis: MastAnalysis, assignment: MastAssignment, input, output=None, limit=None, timeout=None):
    c = config.get_config()
    mast_path = c['mast_path'] if 'mast_path' in c else "mast_analysis.exe"

    cmd = [mast_path, analysis.value]
    if assignment is not MastAssignment.NONE:
        cmd.append("-p")
        cmd.append("-t")
        cmd.append(assignment.value)
    if limit:
        cmd.append("-f")
        cmd.append(str(limit))
    cmd.append(input)
    if output:
        cmd.append(output)

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        print("Timeout!")

    MAST_SCHEDULABLE = "The system is schedulable"
    out = proc.stdout.read().decode() if proc and proc.stdout else ""
    proc.stdout.close()
    schedulable = MAST_SCHEDULABLE in out if out else False
    results = parse_results(output) if output and os.path.isfile(output) else {}
    return schedulable, results

# endregion


# region MAST Results

def parse_results(file):
    wcrts = {}
    with open(file, "r") as f:
        soup = BeautifulSoup(f, features="xml")

        for task_results in soup.findAll("mast_res:Timing_Result"):
            event_name = (task_results["Event_Name"])
            task_name = reverse_output_event_name(event_name)
            wcrt = find_task_wcrt(task_results)
            wcrts[task_name] = wcrt
    return wcrts


def find_task_wcrt(element):
    wcrt = max([float(x["Time_Value"]) for x in element.findAll("mast_res:Global_Response_Time")])
    return wcrt

# endregion


# region MAST Writers

def export(system: System, file):
    txt = write_system(system)
    with open(file, "w") as f:
        f.write(txt)


def write_system(system: System) -> str:
    header = """Model (
        Model_Name  => {0},
        Model_Date  => 2000-01-01);""".format("System")

    procs = "\n\n".join(map(write_processing_resource, system.processors))
    scheds = "\n\n".join(map(write_scheduler, system.processors))
    sss = "\n\n".join(map(write_scheduling_server, system.tasks))
    os = "\n\n".join(map(write_operation, system.tasks))
    ts = "\n\n".join(map(write_transaction, system.flows))
    return header + "\n\n" + procs + "\n\n" + scheds + "\n\n" + sss + "\n\n" + os + "\n\n" + ts


def write_processing_resource(processor: Processor) -> str:
    return (
        f'Processing_Resource (\n'
        f'      Type 			            => Regular_Processor,\n'
        f'      Name 			            => {processor.name},\n'
        f'      Max_Interrupt_Priority	    => 32767,\n'
        f'      Min_Interrupt_Priority	    => 32767);')


def write_scheduler(processor: Processor) -> str:
    body = (
        f'Scheduler (\n'
        f'      Type 			            => Primary_Scheduler,\n'
        f'      Name 			            => {processor.name},\n'
        f'      Host                	    => {processor.name},\n')
    if processor.sched == SchedulerType.EDF:
        body += \
            f'      Policy          =>      ( Type                 => EDF));\n'
    else:
        body += (
            f'      Policy          =>      (\n'
            f'          Type                    => Fixed_Priority,\n'
            f'          Max_Priority            => 32766,\n'
            f'          Min_Priority            => 1));\n'
        )
    return body


def write_scheduling_server(task: Task) -> str:
    body  = (
        f'Scheduling_Server (\n'
        f'        Type                       => Regular,\n'
        f'        Name    => {task.name},\n'
        f'        Server_Sched_Parameters         => (\n'
    )

    if task.sched == SchedulerType.FP:
        body += (
            f'                Type                    => Fixed_Priority_policy,\n'
            f'                The_Priority            => {task.priority},\n'
        )
    else:
        body += (
            f'                Type                    => EDF_policy,\n'
            f'                Deadline                => {task.deadline},\n'
        )
    body += (
        f'                Preassigned             => No),\n'
        f'        Scheduler      => {task.processor.name});\n\n'
    )
    return body
    # return """Scheduling_Server (
    #     Type				=> Fixed_Priority,
    #     Name 				=> {0},
    #     Server_Sched_Parameters		=> (
    #             Type		=> Fixed_Priority_policy,
    #             The_Priority	=> {1},
    #             Preassigned		=> no),
    #     Server_Processing_Resource	=> {2});""".format(task.name, task.priority, task.processor.name)


def write_operation(task: Task) -> str:
    return """Operation (
        Type        => Simple,
        Name        => {0},
        Worst_Case_Execution_Time   => {1});""".format(task.name, task.wcet)


def write_transaction(flow: Flow) -> str:
    prefix = """Transaction (
        Type	=> Regular, 
        Name	=> {0},
        External_Events => (
            (Type 		=> Periodic,
            Name 		=> {1},
            Period 	    => {2})),""".format(flow.name, external_event_name(flow), flow.period)

    ies = """
        Internal_Events => ({0}),""".format("".join(map(write_internal_event, flow.tasks)))

    ehs = """
        Event_Handlers => ({0})""".format(",".join(map(write_event_handler, flow.tasks)))

    return prefix + ies + ehs + ");"


def write_internal_event(task: Task) -> str:
    fixed = """
            (Type 	=> regular,
            name 	=> {0}""".format(output_event_name(task))
    if not task.is_last:
        return fixed + "),"
    else:
        return fixed + """,
            Timing_Requirements => (
                Type 		  => Hard_Global_Deadline,
                Deadline 	  => {0},
                referenced_event => {1}))""".format(task.flow.deadline, external_event_name(task.flow))


def write_event_handler(task: Task) -> str:
    return """
            (Type         => Activity,
            Input_Event         => {0},
            Output_Event        => {1},
            Activity_Operation  => {2},
            Activity_Server     => {3})""".format(input_event_name(task), output_event_name(task), task.name, task.name)


def output_event_name(task: Task) -> str:
    return f"o_{task.name}"


def reverse_output_event_name(event_name) -> str:
    return event_name.lstrip("o_")


def input_event_name(task: Task) -> str:
    return external_event_name(task.flow) if len(task.predecessors) == 0 else output_event_name(task.predecessors[0])


def external_event_name(flow: Flow) -> str:
    return f"e_{flow.name}"


def sanitize_priorities(system: System):
    """
    In-place sanitization of the priorities
    In MAST priorities must be integers higher than 0
    :param system:
    """
    # first save current priorities
    tasks = system.tasks
    save_attrs(tasks, ["priority"], "_sanitize_")

    # assign integer priorities in the same ordering
    prio = 1
    for task in sorted(tasks, key=lambda t: t.priority):
        task.priority = prio
        prio += 1


def desanitize_priorities(system: System):
    """
    Restore the un-sanitized priorities of the system
    :param system:
    """
    restore_attrs(system.tasks, ["priority"], "_sanitize_")

# endregion