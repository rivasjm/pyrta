import unittest

import examples
from gradient import *
from analysis import HolisticFPAnalysis, HolisticGlobalEDFAnalysis
from examples import get_small_system, get_three_tasks
from random import Random
from assignment import PDAssignment, HOPAssignment, repr_priorities
from model import SchedulerType
from vector import VectorHolisticFPBatchCosts, PrioritiesMatrix


class StandardGradientDescentTest(unittest.TestCase):
    def test_fp_vector(self):
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
                                            callback=mapping_prio_callback,
                                            verbose=True)

        r = Random(42)
        pd = PDAssignment(normalize=True)
        system = get_small_system(random=r, utilization=0.7, balanced=True)
        pd.apply(system)

        optimizer.apply(system)
        analysis.apply(system)
        self.assertTrue(system.is_schedulable())

    # def test_mapping_fp_vector(self):
    #     # MAPPING EXTRACTOR DOES NOT WORK WITH VECTOR ANALYSIS
    #     analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    #     extractor = MappingPriorityExtractor()
    #     cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    #     stop_function = StandardStop(limit=100)
    #     delta_function = AvgSeparationDelta(factor=1.5)
    #     batch_cost_function = VectorHolisticFPBatchCosts()
    #     gradient_function = StandardGradient(delta_function=delta_function,
    #                                          batch_cost_function=batch_cost_function)
    #     update_function = NoisyAdam()
    #     optimizer = StandardGradientDescent(extractor=extractor,
    #                                         cost_function=cost_function,
    #                                         stop_function=stop_function,
    #                                         gradient_function=gradient_function,
    #                                         update_function=update_function,
    #                                         callback=mapping_prio_callback,
    #                                         verbose=True)
    #
    #     r = Random(42)
    #     pd = PDAssignment(normalize=True)
    #     system = get_small_system(random=r, utilization=0.7, balanced=False)
    #     pd.apply(system)
    #
    #     optimizer.apply(system)
    #     analysis.apply(system)
    #     self.assertTrue(system.is_schedulable())

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

    def test_mapping_fp(self):
        analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
        extractor = MappingPriorityExtractor()
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
                                            callback=mapping_prio_callback,
                                            verbose=True)

        r = Random(42)
        pd = PDAssignment(normalize=True)
        system = get_small_system(random=r, utilization=0.7, balanced=False)
        pd.apply(system)

        optimizer.apply(system)
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


class FixedAccumIterationsStopTest(unittest.TestCase):
    def test_fixed_accum_iterations_stop(self):
        sut = FixedAccumIterationsStop()
        s = examples.get_three_tasks()
        sut.should_stop(s, [1, 5, 10], cost=0, t=1)
        sut.should_stop(s, [10, 5, 1], cost=0, t=0)
        sut.should_stop(s, [5, 10, 1], cost=0, t=1)
        sol = sut.solution(s)
        cost = sut.solution_cost()
        self.assertEqual(sol, [10, 5, 1])
        self.assertAlmostEqual(cost, -0.6)


def mapping_prio_callback(t, S: System, x, xb, cost, best):
    print(repr_priorities(S))