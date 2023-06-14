import unittest
from mast import MastHolisticAnalysis, MastAssignment
from generator import to_edf
from examples import get_palencia_system
from analysis import repr_wcrts, reset_wcrt, HolisticFPAnalysis, HolisticGlobalEDFAnalysis
from assignment import PDAssignment


MAST_PATH = 'D:\dev\pymast\mast\mast-1-5-1-0-bin\mast_analysis.exe'


class MASTHolisticTest(unittest.TestCase):
    def test_holistic_fp_palencia(self):
        system = get_palencia_system()

        # analyze with MAST
        holistic_mast = MastHolisticAnalysis(mast_path=MAST_PATH, limit_factor=100)
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
        holistic_mast = MastHolisticAnalysis(mast_path=MAST_PATH, limit_factor=100)
        system.apply(holistic_mast)
        mast_wcrts = [task.wcrt for task in system.tasks]

        # analyze with python
        reset_wcrt(system)
        holistic_py = HolisticGlobalEDFAnalysis(limit_factor=100)
        system.apply(holistic_py)
        py_wcrts = [task.wcrt for task in system.tasks]

        # compare results: should be the same
        self.assertListEqual(mast_wcrts, py_wcrts)



if __name__ == '__main__':
    unittest.main()
