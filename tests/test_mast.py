import random
import unittest
from mast import MastHolisticAnalysis, MastAssignment
from generator import to_edf
from examples import get_palencia_system, get_small_system, get_medium_system, get_big_system
from analysis import repr_wcrts, reset_wcrt, HolisticFPAnalysis, HolisticGlobalEDFAnalysis
from assignment import PDAssignment

class MASTHolisticTest(unittest.TestCase):
    def test_holistic_fp_palencia(self):
        system = get_palencia_system()

        # analyze with MAST
        holistic_mast = MastHolisticAnalysis(limit_factor=100)
        system.apply(holistic_mast)
        mast_wcrts = [task.wcrt for task in system.tasks]

        # analyze with python
        reset_wcrt(system)
        holistic_py = HolisticFPAnalysis(limit_factor=100)
        system.apply(holistic_py)
        py_wcrts = [task.wcrt for task in system.tasks]

        # compare results: should be the same
        self.assertListEqual(mast_wcrts, py_wcrts)

    def test_holistic_global_edf_palencia(self):
        system = get_palencia_system()
        to_edf(system)

        # assign PD deadlines
        pd = PDAssignment(globalize=True)
        pd.apply(system)

        # analyze with MAST
        holistic_mast = MastHolisticAnalysis(limit_factor=100)
        system.apply(holistic_mast)
        mast_wcrts = [task.wcrt for task in system.tasks]

        # analyze with python
        reset_wcrt(system)
        holistic_py = HolisticGlobalEDFAnalysis(limit_factor=100)
        system.apply(holistic_py)
        py_wcrts = [task.wcrt for task in system.tasks]

        # compare results: should be the same
        self.assertListEqual(mast_wcrts, py_wcrts)

    def test_holistic_global_edf_small(self):
        r = random.Random(42)
        system = get_small_system(r, utilization=0.7, balanced=True)
        to_edf(system)

        # assign PD deadlines
        pd = PDAssignment(globalize=True)
        pd.apply(system)

        # analyze with MAST
        holistic_mast = MastHolisticAnalysis(limit_factor=100)
        system.apply(holistic_mast)
        mast_wcrts = [task.wcrt for task in system.tasks]

        # analyze with python
        reset_wcrt(system)
        holistic_py = HolisticGlobalEDFAnalysis(limit_factor=100)
        system.apply(holistic_py)
        py_wcrts = [task.wcrt for task in system.tasks]

        # compare results: should be the same
        for m, p in zip(mast_wcrts, py_wcrts):
            self.assertAlmostEqual(m, p, delta=0.001)

    def test_holistic_global_edf_medium(self):
        r = random.Random(42)
        system = get_medium_system(r, utilization=0.5, balanced=True)
        to_edf(system)

        # assign PD deadlines
        pd = PDAssignment(globalize=True)
        pd.apply(system)

        # analyze with MAST
        holistic_mast = MastHolisticAnalysis(limit_factor=100)
        system.apply(holistic_mast)
        mast_wcrts = [task.wcrt for task in system.tasks]

        # analyze with python
        reset_wcrt(system)
        holistic_py = HolisticGlobalEDFAnalysis(limit_factor=100)
        system.apply(holistic_py)
        py_wcrts = [task.wcrt for task in system.tasks]

        # compare results: should be the same
        for m, p in zip(mast_wcrts, py_wcrts):
            self.assertAlmostEqual(m, p, delta=0.001)


if __name__ == '__main__':
    unittest.main()
