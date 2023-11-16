
from random import Random
from model import *
from exec_time import ExecTime


def calculate_priorities(system) -> bool:
    changed = False
    for processor in system.processors:
        tasks = sorted(processor.tasks,
                       key=lambda task: task.deadline,
                       reverse=True)
        for i, task in enumerate(tasks):
            if not changed and task.priority != i + 1:
                changed = True
            task.priority = i + 1
    return changed


def globalize_deadlines(system: System):
    for flow in system.flows:
        tasks = flow.tasks
        if len(tasks) <= 1:
            continue
        for i, task in enumerate(tasks):
            if i == 0:
                continue
            task.deadline += tasks[i-1].deadline


def clear_assignment(system):
    for t in system.tasks:
        t.priority = 1
        t.deadline = None


def normalize_priorities(system):
    max_priority = max(map(lambda t: t.priority, system.tasks))
    for t in system.tasks:
        t.priority = t.priority/max_priority


def save_assignment(system: System):
    save_attrs(system.tasks, ["priority", "deadline"])


def restore_assignment(system: System):
    restore_attrs(system.tasks, ["priority", "deadline"])


class PassthroughAssignment:
    def __init__(self, normalize=False):
        self.normalize = normalize

    def apply(self, system: System):
        if self.normalize:
            normalize_priorities(system)


class RandomAssignment:
    def __init__(self, random=Random(42), normalize=False):
        self.random = random
        self.normalize = normalize

    def apply(self, system: System):
        tasks = system.tasks
        self.random.shuffle(tasks)
        for task, priority in zip(tasks, range(1, len(tasks)+1)):
            task.priority = priority
        if self.normalize:
            normalize_priorities(system)


class PDAssignment:
    def __init__(self, normalize=False, globalize=False):
        self.normalize = normalize
        self.globalize = globalize
        self.exec_time = ExecTime()

    def apply(self, system: System):
        self.exec_time.init()
        self.calculate_local_deadlines(system)
        if self.globalize:
            globalize_deadlines(system)
        calculate_priorities(system)
        if self.normalize:
            normalize_priorities(system)
        self.exec_time.stop()

    @staticmethod
    def calculate_local_deadlines(system):
        for flow in system:
            sum_wcet = sum(map(lambda t: t.wcet, flow.tasks))
            for task in flow:
                d = task.wcet * flow.deadline / sum_wcet
                task.deadline = d


class EQSAssignment:
    def apply(self):
        pass

    @staticmethod
    def compute_deadlines(system: System):
        for flow in system:
            s = 0
            n = len(flow.tasks)
            for j, task in enumerate(reversed(flow.tasks)):
                s += task.wcet
                task.deadline = task.wcet + (flow.deadline - s)/(n - j + 1)


class HOPAssignment:
    def __init__(self, analysis, iterations=40, k_pairs=None, patience=40, over_iterations=0,
                 callback=None, normalize=False, globalize=False, verbose=False):
        self.analysis = analysis
        self.k_pairs = k_pairs if k_pairs else HOPAssignment.default_k_pairs()
        self.iterations = iterations
        self.patience = patience
        self.over_iterations = over_iterations
        self.callback = callback
        self.globalize = globalize
        self.verbose = verbose
        self.normalize = normalize
        self.exec_time = ExecTime()
        self.iterations_to_sched = -1

    @staticmethod
    def default_k_pairs():
        return [(2.0, 2.0), (1.8, 1.8), (3.0, 3.0), (1.5, 1.5)]

    def apply(self, system: System):
        self.exec_time.init()
        self.iterations_to_sched = -1
        iteration = 0
        patience = self.patience if self.patience >= 0 else 100
        over_iterations = self.over_iterations
        stop = False
        optimizing = False
        best_slack = float("-inf")

        PDAssignment.calculate_local_deadlines(system)
        if self.globalize:
            globalize_deadlines(system)
        save_assignment(system)

        for ka, kr in self.k_pairs:
            restore_assignment(system)  # always start each new k-pair iteration with the best

            for i in range(self.iterations):
                iteration += 1
                if self.verbose:
                    print(f"Iteration={i}, ka={ka}, kr={kr} ", end="")

                changed = calculate_priorities(system)  # update priorities
                patience = patience-1 if not changed else self.patience

                system.apply(self.analysis)  # update response times
                self.clean_response_times(system)
                if self.callback:
                    self.callback.apply(system)

                slack = system.slack
                if slack > best_slack:
                    best_slack = slack
                    save_assignment(system)

                if self.verbose:
                    sched = "SCHEDULABLE" if system.is_schedulable() else "NOT SCHEDULABLE"
                    print(f"slack={system.slack} {sched}")

                if system.is_schedulable() and self.iterations_to_sched < 0:
                    self.iterations_to_sched = iteration

                if system.is_schedulable() and over_iterations > 0:
                    optimizing = True

                if optimizing:
                    over_iterations -= 1

                if (not optimizing and system.is_schedulable()) or patience <= 0:
                    stop = True
                    break
                elif optimizing and over_iterations < 0 or patience <= 0:
                    stop = True
                    break

                self.update_local_deadlines(system, ka, kr)
                if self.globalize:
                    globalize_deadlines(system)

            if stop:
                break

        self.delete_excesses(system)
        restore_assignment(system)
        self.exec_time.stop()
        system.apply(self.analysis)
        if self.verbose:
            sched = "SCHEDULABLE" if system.is_schedulable() else "NOT SCHEDULABLE"
            print(f"Returning best assignment: slack={system.slack} {sched}")
        if self.normalize:
            normalize_priorities(system)

    def update_local_deadlines(self, system: System, ka, kr):
        # update excesses with last response times
        for task in system.tasks: self.save_task_excess(task)
        for proc in system.processors: self.save_proc_excess(proc)
        for flow in system.flows: self.save_flow_mex(flow)
        self.save_proc_mex(system)

        # calculate unadjusted local deadlines
        for task in system.tasks:
            self.save_local_deadline(task, ka, kr)

        # adjust local deadlines
        self.adjust_local_deadlines(system)

    @staticmethod
    def save_local_deadline(task: Task, ka, kr):
        mex_pr = task.flow.system.mex_pr
        second = 1 + task.processor.excess/(kr * mex_pr) if kr * mex_pr != 0 else sys.float_info.max
        third = 1 + task.excess/(ka * task.flow.excess) if ka * task.flow.excess != 0 else sys.float_info.max
        task.deadline = task.deadline * second * third

    @staticmethod
    def save_task_excess(task: Task):
        d = task.deadline
        e = 0
        if d <= task.period:
            e = (task.wcrt-d)*task.flow.wcrt/task.flow.deadline
        elif d > task.period:
            e = (task.wcrt+task.jitter-d)*task.flow.wcrt/task.flow.deadline
        task.excess = e

    @staticmethod
    def save_proc_excess(proc: Processor):
        proc.excess = sum([task.excess for task in proc.tasks])

    @staticmethod
    def save_flow_mex(flow: Flow):
        excesses = [abs(task.excess) for task in flow.tasks]
        flow.excess = max(excesses) if len(excesses) > 0 else 0

    @staticmethod
    def save_proc_mex(system: System):
        excesses = [abs(proc.excess) for proc in system.processors]
        system.mex_pr = max(excesses) if len(excesses) > 0 else 0

    @staticmethod
    def delete_excesses(system: System):
        for task in system.tasks:
            if hasattr(task, "excess"): del task.excess
        for flow in system.flows:
            if hasattr(flow, "excess"): del flow.excess
        for proc in system.processors:
            if hasattr(proc, "excess"): del proc.excess
        if hasattr(system, "mex_pr"): del system.mex_pr

    @staticmethod
    def adjust_local_deadlines(system: System):
        for flow in system.flows:
            d_sum = sum([task.deadline for task in flow])
            for task in flow:
                task.deadline = task.deadline * flow.deadline / d_sum

    @staticmethod
    def clean_response_times(system):
        for task in system.tasks:
            if task.wcrt is None:
                task.wcrt = sys.float_info.max


def walk_random_priorities(system: System, breadth, depth, callback, verbose=False, seed=None):
    # it must have at least one processor with more than 1 task
    procs = [p for p in system.processors if len(p.tasks) > 1]
    if len(procs) < 1:
        return

    random = Random(seed)
    save_assignment(system)  # back up current priorities

    if verbose:
        print(f"Starting random priority walk [breadth={breadth}, depth={depth}]")

    for b in range(breadth):
        restore_assignment(system)

        for d in range(depth):
            # pick a random processor that has more than 1 task
            p = random.choice(procs)

            # randomly pick 2 tasks in this processor
            t1, t2 = random.sample(p.tasks, 2)

            # swap their priorities
            t1.priority, t2.priority = t2.priority, t1.priority

            # apply callback on this system
            if verbose:
                print(f"Random priority walk: breadth={b}, depth={d}")

            if callback:
                callback.apply(system)

    restore_assignment(system)  # restore initial priorities


def walk_random_priorities_processors(system: System, breadth, depth, callback, verbose=False):
    # it must have more than one processor with more than 1 task
    procs = [p for p in system.processors if len(p.tasks) > 1]
    if len(procs) <= 1:
        return

    random = Random()
    save_attrs(system.tasks, ["processor", "priority"])

    if verbose:
        print(f"Starting random priority/processor walk [breadth={breadth}, depth={depth}]")

    for b in range(breadth):
        restore_attrs(system.tasks, ["processor", "priority"])

        for d in range(depth):
            # there is a 50% change of either swapping priorities
            # or swapping processors

            if random.random() < 0.5:
                # swap priorities
                p = random.choice(procs)
                t1, t2 = random.sample(p.tasks, 2)
                t1.priority, t2.priority = t2.priority, t1.priority

            else:
                # swap processor mapping
                p1, p2 = random.sample(procs, 2)
                t1, t2 = random.choice(p1.tasks), random.choice(p2.tasks)
                t1.processor, t2.processor = t2.processor, t1.processor

            # apply callback on this system
            if verbose:
                print(f"Random priority/processor walk: breadth={b}, depth={d}")

            if callback:
                callback(system)

    restore_attrs(system.tasks, ["processor", "priority"])


def random_priority_jump(system: System, random=Random()):
    tasks = system.tasks
    a = random.randint(0, len(tasks)-1)
    b = a
    while b == a:
        b = random.randint(0, len(tasks) - 1)
    tasks[a].priority, tasks[b].priority = tasks[b].priority, tasks[a].priority


def repr_priorities(system: System) -> str:
    msg = ""
    for flow in system.flows:
        ts = " ".join(map(lambda t: f"{t.priority:.2f}[{t.processor.name}]", flow.tasks))
        msg += f"{flow.period:.2f}: {ts} : {flow.deadline:.2f}\n"
    return msg


def repr_deadlines(system: System) -> str:
    msg = ""
    for flow in system.flows:
        ts = " ".join(map(lambda t: f"{t.deadline:.2f}[{t.processor.name}]", flow.tasks))
        msg += f"{flow.period:.2f}: {ts} : {flow.deadline:.2f}\n"
    return msg
