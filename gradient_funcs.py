from model import System


class GradientDescentFunction:
    def apply(self, S: System) -> [float]:
        pass


class Extractor:
    def extract(self, S: System) -> [float]:
        pass

    def insert(self, S: System, x: [float]) -> None:
        pass


class CostFunction:
    def apply(self, S: System, x: [float]) -> float:
        pass


class BatchCostFunction:
    def apply(self, S: System, inputs: [[float]]) -> [float]:
        pass


class StopFunction:
    def apply(self, S: System, x: [float], cost: float, t: int) -> bool:
        pass


class DeltaFunction:
    def apply(self, S: System, x: [float]) -> [float]:
        pass


class GradientFunction:
    def apply(self, S: System, x: [float]) -> [float]:
        pass


class UpdateFunction:
    def apply(self, S: System, x: [float], nabla: [float], t: int) -> [float]:
        pass
