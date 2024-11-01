from model import System


class Reseteable():
    def reset(self):
        pass

    def __repr__(self):
        return type(self).__name__


class GradientDescentFunction(Reseteable):
    """Class to perform Gradient Descent. Usually composed of building blocks such as Extractor, CostFunction, etc."""
    def apply(self, S: System) -> [float]:
        """Apply gradient descent on the given system. Returns the optimized parameters. These parameters
        might get inserted into the system before this method is returned"""
        pass


class Extractor(Reseteable):
    """Class to extract an insert parameters into a system.
    The parameters is stored as a flat list of floats"""
    def extract(self, S: System) -> [float]:
        pass

    def insert(self, S: System, x: [float]) -> None:
        pass

    def mask(self, S: System, x: [float], t: int) -> [bool]:
        return [True]*len(x)


class CostFunction(Reseteable):
    """Class to compute the cost value of the given parameters"""
    def apply(self, S: System, x: [float]) -> float:
        pass


class BatchCostFunction(Reseteable):
    """Class to compute the cost values of a list of parameters.
    One cost value is returned per parameter list"""
    def apply(self, S: System, inputs: [[float]]) -> [float]:
        pass


class StopFunction(Reseteable):
    """Class to determine if the optimization process must stop for the given state, represented by the
    given parameters, cost value and iteration number"""
    def should_stop(self, S: System, x: [float], cost: float, t: int) -> bool:
        """Returns true if the optimization process must stop given the current state"""
        pass

    def solution(self, S: System):
        """Returns the solution it considers best. May return several solutions"""
        pass

    def solution_cost(self):
        """Returns the cost value of the solution"""
        pass


class DeltaFunction(Reseteable):
    """Class to calculate the delta with which compute the gradient"""
    def apply(self, S: System, x: [float]) -> [float]:
        """Returns true if the optimization process must stop given the current state"""
        pass


class GradientFunction(Reseteable):
    """Class to compute the gradient at the given parameters"""
    def apply(self, S: System, x: [float]) -> [float]:
        pass


class UpdateFunction(Reseteable):
    """Class t update the given input parameters, according to the gradient and iteration number"""
    def apply(self, S: System, x: [float], nabla: [float], t: int) -> [float]:
        pass
