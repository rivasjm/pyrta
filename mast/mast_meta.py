from model import Flow, Task, TaskType
from mast_constants import MAX_PRIORITY


def unavailable_flow(period, window, cpu):
    flow = Flow(name="unavailable", period=period, deadline=period*10)
    flow.add_tasks(
        Task(name="unavailable_1", type=TaskType.Offset, wcet=window, bcet=window),
        Task(name="unavailable_2", wcet=period-window, bcet=period-window, priority=MAX_PRIORITY, processor=cpu)
    )
    return flow

