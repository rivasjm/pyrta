from model import System


class DeadlineExtractor:
    @staticmethod
    def extract(system: System) -> [float]:
        r = [t.deadline for t in system.tasks]
        return r

    @staticmethod
    def apply(system: System, vector: [float]):
        tasks = system.tasks
        assert len(tasks) == len(vector)
        for v, t in zip(vector, tasks):
            t.deadline = v


class PriorityExtractor:
    @staticmethod
    def extract(system: System) -> [float]:
        r = [t.priority for t in system.tasks]
        return r

    @staticmethod
    def apply(system: System, vector: [float]):
        tasks = system.tasks
        assert len(tasks) == len(vector)
        for v, t in zip(vector, tasks):
            t.deadline = v