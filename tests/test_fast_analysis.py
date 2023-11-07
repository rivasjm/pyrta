import unittest
from analysis import HolisticFPAnalysis, reset_wcrt
from fast_analysis import FastHolisticFPAnalysis
from assignment import PDAssignment
import examples
from random import Random
from generator import set_utilization


class MyTestCase(unittest.TestCase):
    def test_fast_holistic(self):
        # test that the real Holistic Analysis and Fast Holistic Analysis return the same results
        # the fast analysis is configured to mimic the real analysis
        holistic = HolisticFPAnalysis(limit_factor=10, reset=False)
        fast_holistic = FastHolisticFPAnalysis(limit_factor=10, limit_i=-1, limit_p=-1, ceiling=True, fast=False)

        for i in range(100):
            rnd = Random(i)
            s = examples.get_medium_system(random=rnd, utilization=0.5, balanced=True)
            pd = PDAssignment()
            pd.apply(s)

            reset_wcrt(s)
            holistic.apply(s)
            r_holistic = [task.wcrt for task in s.tasks]

            reset_wcrt(s)
            fast_holistic.apply(s)
            r_fast_holistic = [task.wcrt for task in s.tasks]

            self.assertListEqual(r_holistic, r_fast_holistic)

    def test_limited_case(self):
        """Test a case that reaches the limit"""
        rnd = Random(42)
        size = (3, 4, 3)  # flows, tasks, procs
        system = examples.get_system(size, rnd, balanced=True, name=str(0),
                                     deadline_factor_min=0.5, deadline_factor_max=1)

        set_utilization(system, 0.8)
        prio = [0.5, 0.05681818181818182, 0.25, 0.75, 0.75, 0.25, 0.25, 1.0, 1.0, 0.5, 1.0, 0.75]

        for p, t in zip(prio, system.tasks):
            t.priority = p

        holistic = HolisticFPAnalysis(limit_factor=10, reset=False)
        fast_holistic = FastHolisticFPAnalysis(limit_factor=10, limit_p=-1, limit_i=-1, ceiling=True, fast=False)

        reset_wcrt(system)
        holistic.apply(system)
        a = [task.wcrt for task in system.tasks]

        reset_wcrt(system)
        fast_holistic.apply(system)
        b = [task.wcrt for task in system.tasks]

        # there are differences in the way both analysis behave when reaching the limit. but the tasks
        # that do not reach the limit should have the same wcrt
        self.assertAlmostEqual(a[0], b[0])

if __name__ == '__main__':
    unittest.main()
