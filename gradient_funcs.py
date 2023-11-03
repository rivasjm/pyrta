from model import System


class GradientDescentFunction:
    """Class to perform Gradient Descent. Usually composed of building blocks such as Extractor, CostFunction, etc."""
    def apply(self, S: System) -> [float]:
        """Apply gradient descent on the given system. Returns the optimized parameters. These parameters
        might get inserted into the system before this method is returned"""
        pass


class Extractor:
    """Class to extract an insert parameters into a system.
    The parameters is stored as a flat list of floats"""
    def extract(self, S: System) -> [float]:
        pass

    def insert(self, S: System, x: [float]) -> None:
        pass


class CostFunction:
    """Class to compute the cost value of the given parameters"""
    def apply(self, S: System, x: [float]) -> float:
        pass


class BatchCostFunction:
    """Class to compute the cost values of a list of parameters.
    One cost value is returned per parameter list"""
    def apply(self, S: System, inputs: [[float]]) -> [float]:
        pass


class StopFunction:
    """Class to determine if the optimization process must stop for the given state, represented by the
    given parameters, cost value and iteration number"""
    def apply(self, S: System, x: [float], cost: float, t: int) -> bool:
        """Returns true if the optimization process must stop given the current state"""
        pass

class DeltaFunction:
    """Class to calculate the delta with which compute the gradient"""
    def apply(self, S: System, x: [float]) -> [float]:
        """Returns true if the optimization process must stop given the current state"""
        pass

class GradientFunction:
    """Class to compute the gradient at the given parameters"""
    def apply(self, S: System, x: [float]) -> [float]:
        pass


class UpdateFunction:
    """Class t update the given input parameters, according to the gradient and iteration number"""
    def apply(self, S: System, x: [float], nabla: [float], t: int) -> [float]:
        pass
