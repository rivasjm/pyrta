import unittest
from gradient import *
from analysis import HolisticFPAnalysis, HolisticGlobalEDFAnalysis
from examples import get_small_system
from random import Random
from assignment import PDAssignment, HOPAssignment
from model import SchedulerType


class StandardGradientDescentTest(unittest.TestCase):
    def test_fp(self):
        analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
        extractor = PriorityExtractor()
        cost_function = InvslackCost(extractor=extractor, analysis=analysis)
        stop_function = StandardStop(limit=100)
        delta_function = AvgSeparationDelta(factor=1.5)
        batch_cost_function = SequentialBatchCostFunction(cost_function=cost_function)
        gradient_function = StandardGradient(delta_function=delta_function,
                                             batch_cost_function=batch_cost_function)
        update_function = NoisyAdam()
        optimizer = StandardGradientDescent(extractor=extractor,
                                            cost_function=cost_function,
                                            stop_function=stop_function,
                                            gradient_function=gradient_function,
                                            update_function=update_function,
                                            verbose=True)

        r = Random(42)
        pd = PDAssignment(normalize=True)
        system = get_small_system(random=r, utilization=0.7, balanced=True)
        pd.apply(system)

        x = optimizer.apply(system)
        extractor.insert(system, x)
        analysis.apply(system)
        self.assertTrue(system.is_schedulable())

    def test_gedf(self):
        analysis = HolisticGlobalEDFAnalysis(limit_factor=10, reset=False)
        extractor = DeadlineExtractor()
        cost_function = InvslackCost(extractor=extractor, analysis=analysis)
        stop_function = StandardStop(limit=100)
        delta_function = AvgSeparationDelta(factor=1.5)
        batch_cost_function = SequentialBatchCostFunction(cost_function=cost_function)
        gradient_function = StandardGradient(delta_function=delta_function,
                                             batch_cost_function=batch_cost_function)
        update_function = NoisyAdam()
        optimizer = StandardGradientDescent(extractor=extractor,
                                            cost_function=cost_function,
                                            stop_function=stop_function,
                                            gradient_function=gradient_function,
                                            update_function=update_function,
                                            verbose=True)

        r = Random(42)
        pd = PDAssignment(normalize=True, globalize=True)
        system = get_small_system(random=r, utilization=0.81, balanced=True, sched=SchedulerType.EDF)
        pd.apply(system)

        x = optimizer.apply(system)
        extractor.insert(system, x)
        analysis.apply(system)
        self.assertTrue(system.is_schedulable())


