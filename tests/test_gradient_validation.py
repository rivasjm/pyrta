import unittest

from analysis import HolisticFPAnalysis
from assignment import PDAssignment
from examples import get_validation_example
from gradient import PriorityExtractor, InvslackCost, StandardStop, AvgSeparationDelta, StandardGradient, NoisyAdam, \
    StandardGradientDescent
from model import System
from vector import VectorHolisticFPBatchCosts, PrioritiesMatrix


class GradientTest(unittest.TestCase):

    def test_validation(self):
        system = get_validation_example()
        print(system)
        gdpa_pd_fp_vector(system)


if __name__ == '__main__':
    unittest.main()


def gdpa_pd_fp_vector(system: System) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    extractor = PriorityExtractor()
    cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=100)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = VectorHolisticFPBatchCosts(PrioritiesMatrix())
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam()
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=True)

    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    analysis.apply(system)
    return system.is_schedulable()