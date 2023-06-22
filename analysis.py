from model import Task, System, Processor, save_attrs, restore_attrs
import math


class HolisticGlobalEDFAnalysis:
    def __init__(self, limit_factor=10, reset=True, verbose=False):
        self.limit_factor = limit_factor
        self.reset = reset
        self.verbose = verbose

    @staticmethod
    def _activations(task: Task, length: float) -> int:
        return math.ceil((length+task.jitter)/task.period)

    def _longest_busy_period(self, proc: Processor, l_prev: float) -> float:
        length = sum(map(lambda t: math.ceil((l_prev+t.jitter)/t.period)*t.wcet, proc.tasks))
        if math.isclose(length, l_prev):
            return length
        else:
            return self._longest_busy_period(proc, length)

    def _set_psi(self, proc: Processor, busy_period: float):
        psi = [(p-1)*task.period - task.jitter + task.deadline
               for task in proc.tasks for p in range(1, self._activations(task, busy_period)+1)]
        return psi

    def apply(self, system: System) -> None:
        init_wcrt(system)
        try:
            while True:
                changed = False
                for proc in system.processors:
                    changed |= self._proc_analysis(proc)
                if not changed:
                    break
        except LimitFactorReachedException as e:
            if self.verbose:
                print(e.message)
            if self.reset:
                reset_wcrt(system)
            else:
                e.task.wcrt = e.response_time
                for task in e.task.all_successors:
                    task.wcrt = e.response_time

    def _proc_analysis(self, proc: Processor):
        length = self._longest_busy_period(proc, 0)
        changed = False
        for task in proc.tasks:
            changed |= self._task_analysis(task, length)
        return changed

    def _task_analysis(self, task: Task, length: float) -> bool:
        max_r = 0
        all_psi = self._set_psi(task.processor, length)
        for p in range(1, self._activations(task, length) + 1):
            activations = [psi - (p-1) * task.period + task.jitter - task.deadline for psi in all_psi
                           if (p-1)*task.period-task.jitter+task.deadline <= psi < p*task.period-task.jitter+task.deadline]
            for activation in activations:
                r = self._ra(task, activation, p)
                if r > max_r:
                    max_r = r
                if r > task.flow.deadline * self.limit_factor:
                    raise LimitFactorReachedException(task, r, task.flow.deadline * self.limit_factor)

        if max_r > task.wcrt:
            task.wcrt = max_r
            return True
        else:
            return False

    def _ra(self, task, activation, p):
        deadline_activation = activation - task.jitter + (p-1)*task.period + task.deadline
        wa = self._wa(task, deadline_activation, p, 0)
        ra = wa - activation + task.jitter - (p-1)*task.period
        return ra

    def _wa(self, task, deadline_activation, p, wa_prev):
        wa = p*task.wcet + sum(map(lambda t: self._wi(t, wa_prev, deadline_activation),
                                   [t for t in task.processor.tasks if t != task]))
        if math.isclose(wa, wa_prev):
            return wa
        else:
            return self._wa(task, deadline_activation, p, wa)


    @staticmethod
    def _wi(task: Task, t: float, D: float) -> float:
        value = min(math.ceil((t+task.jitter)/task.period), math.floor((task.jitter + D - task.deadline)/task.period)+1)
        return value * task.wcet if value > 0 else 0


class HolisticFPAnalysis:
    def __init__(self, limit_factor=10, reset=True, verbose=False):
        self.limit_factor = limit_factor
        self.reset = reset
        self.verbose = verbose

    def apply(self, system: System) -> None:
        init_wcrt(system)

        try:
            while True:
                changed = False
                for task in system.tasks:
                    changed |= self._task_analysis(task)
                if not changed:
                    break

        except LimitFactorReachedException as e:
            if self.verbose:
                print(e.message)
            if self.reset:
                reset_wcrt(system)
            else:
                e.task.wcrt = e.response_time
                for task in e.task.all_successors:
                    task.wcrt = e.response_time

    def _task_analysis(self, task: Task) -> bool:
        p = 1
        rmax = 0
        while True:
            wip = self._wip(p, task)
            r = wip - (p-1)*task.period + task.jitter
            if r > rmax:
                rmax = r

            if r > task.flow.deadline * self.limit_factor:
                raise LimitFactorReachedException(task, r, task.flow.deadline * self.limit_factor)

            # print(f"id={task.name}, wip={wip}, r={r}, rmax={rmax}, p*T={p*task.period}")
            if wip <= p * task.period:
                break
            p += 1

        if math.isclose(task.wcrt, rmax):
            return False
        else:
            task.wcrt = rmax
            return True

    def _wip(self, p: int, task: Task) -> float:
        w_ini = p * task.wcet
        return self._wi(p, w_ini, task)

    def _wi(self, p: int, w_prev: float, task: Task) -> float:
        hp = higher_priority(task)
        w = sum(map(lambda t: math.ceil((t.jitter + w_prev)/t.period)*t.wcet, hp)) + p*task.wcet

        provisional_r = w - (p-1)*task.period + task.jitter
        if provisional_r > task.flow.deadline * self.limit_factor:
            raise LimitFactorReachedException(task, provisional_r, task.flow.deadline * self.limit_factor)

        if math.isclose(w, w_prev):
            return w
        else:
            return self._wi(p, w, task)


class JosephPandyaAnalysis:
    def __init__(self, reset=True, verbose=False):
        self.reset = reset
        self.verbose = verbose

    def apply(self, system: System):
        init_wcrt(system)

        # first pass to calculate local response times
        for task in system.tasks:
            try:
                r = self._ri(task, task.wcet, higher_priority(task))
                task.wcrt = r
            except LimitFactorReachedException as e:
                if self.verbose:
                    print(e.message)
                if self.reset:
                    reset_wcrt(system)  # set as non schedulable
                    break
                else:
                    e.task.wcrt = e.response_time

        # second pass to convert to global response times
        for flow in system.flows:
            for i, task in enumerate(flow.tasks):
                if task.wcrt and i > 0:
                    task.wcrt += sum(map(lambda t: t.period, flow.tasks[1:i+1]))

    def _ri(self, task: Task, r, hp: list):
        _r = task.wcet + sum(map(lambda t: math.ceil(r/t.period)*t.wcet, hp))
        if self.verbose:
            print(f"  task={task.name}: {_r:.2f}")
        if _r > task.period:
            raise LimitFactorReachedException(task, _r, task.period)
        if _r != r:
            return self._ri(task, _r, higher_priority(task))
        else:
            return _r


def higher_priority(task: Task) -> [Task]:
    return [t for t in task.processor.tasks
            if t.priority >= task.priority and t != task]


def init_wcrt(system: System):
    for flow in system.flows:
        tasks = flow.tasks
        for i, task in enumerate(tasks):
            task.wcrt = task.wcet
            if i > 0:
                task.wcrt += tasks[i-1].wcrt


def reset_wcrt(system: System):
    for task in system.tasks:
        task.wcrt = None


def save_wcrt(system: System):
    save_attrs(system.tasks, ["wcrt"])


def restore_wcrt(system: System):
    restore_attrs(system.tasks, ["wcrt"])


def repr_wcrts(system: System) -> str:
    msg = ""
    for flow in system.flows:
        ts = " ".join(map(lambda t: f"{t.wcrt if t.wcrt else 0:.2f}", flow.tasks))
        msg += f"{flow.period}: {ts} : {flow.deadline}\n"
    return msg


class LimitFactorReachedException(Exception):
    def __init__(self, task, response_time, limit):
        self.task = task
        self.response_time = response_time
        self.limit = limit
        self.message = f"Analysis stopped because provisional response time for task {task.name} (R={response_time}) " \
                       f"reached the limit (limit={limit})"
        super().__init__(self.message)