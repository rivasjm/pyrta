import unittest
from random import Random

from assignment import PDAssignment, HOPAssignment, RandomAssignment, clear_assignment, repr_deadlines
from model import *
from examples import *
from analysis import (HolisticFPAnalysis, JosephPandyaAnalysis, HolisticGlobalEDFAnalysis, repr_wcrts,
                      HolisticLocalEDFAnalysis)
from generator import generate_system, to_edf
from mast_tools import MastHolisticAnalysis


class HolisticTest(unittest.TestCase):
    def test_palencia(self):
        system = get_palencia_system()
        flow1 = system['flow1']
        flow2 = system['flow2']
        cpu1 = next((p for p in system.processors if p.name == "cpu1"), None)
        cpu2 = next((p for p in system.processors if p.name == "cpu2"), None)
        network = next((p for p in system.processors if p.name == "network"), None)

        # analyze
        system.apply(HolisticFPAnalysis())

        #
        self.assertEqual(flow1.wcrt, 42)
        self.assertEqual(flow2.wcrt, 30)
        self.assertTrue(system.is_schedulable())
        self.assertAlmostEqual(cpu1.utilization, 0.416, delta=0.001)
        self.assertAlmostEqual(cpu2.utilization, 0.791, delta=0.001)
        self.assertAlmostEqual(network.utilization, 0.316, delta=0.001)
        self.assertAlmostEqual(system.max_utilization, 0.791, delta=0.001)
        self.assertAlmostEqual(system.utilization, 0.508, delta=0.001)

        #
        # print(system.processors)

    def test_random(self):
        random = Random(10)
        pd = PDAssignment()
        holistic = HolisticFPAnalysis()
        hopa = HOPAssignment(analysis=HolisticFPAnalysis(reset=False), over_iterations=10, verbose=True)

        utilization = 0.85
        system = generate_system(random,
                                 n_flows=random.randint(1, 10),
                                 n_tasks=random.randint(1, 10),
                                 n_procs=random.randint(1, 5),
                                 sched=SchedulerType.FP,
                                 utilization=utilization,
                                 period_min=100, period_max=100 * random.uniform(2.0, 1000.0),
                                 deadline_factor_min=0.5, deadline_factor_max=2)

        # this one should be schedulable at the third iteration
        # in the first iteration it triggers the stop factor
        system.apply(hopa)
        system.apply(holistic)
        print(system.is_schedulable())
        print(system.slack)
        self.assertAlmostEqual(system.slack, 0.28803, delta=0.00001)  # should be from iteration 8 of 12

    def test_joseph_pandya(self):
        random = Random(10)
        pandya = JosephPandyaAnalysis(verbose=False, reset=True)
        pd = PDAssignment()
        ra = RandomAssignment()
        hopa = HOPAssignment(analysis=pandya)

        utilization = 0.1
        while utilization < 1:
            system = generate_system(random,
                                     n_flows=4,
                                     n_tasks=5,
                                     n_procs=4,
                                     sched=SchedulerType.FP,
                                     utilization=utilization,
                                     period_min=100, period_max=100 * random.uniform(2.0, 1000.0),
                                     deadline_factor_min=0.86, deadline_factor_max=0.86)

            system.apply(ra)
            system.apply(pandya)
            sched_ra = system.is_schedulable()
            clear_assignment(system)

            system.apply(pd)
            system.apply(pandya)
            sched_pd = system.is_schedulable()
            clear_assignment(system)

            system.apply(hopa)
            system.apply(pandya)
            sched_hopa = system.is_schedulable()
            clear_assignment(system)

            print(f"{utilization:.1f}: ra={sched_ra}, pd={sched_pd}, hopa={sched_hopa}")
            utilization += 0.1


if __name__ == '__main__':
    unittest.main()
