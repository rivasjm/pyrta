import unittest
from examples import get_palencia_system
from assignment import PDAssignment
from generator import to_edf


class PDTest(unittest.TestCase):
    def test_local_edf(self):
        system = to_edf(get_palencia_system())
        pd = PDAssignment()
        pd.apply(system)

        expected = [11.11111, 4.44444, 44.44444, 16, 32, 32]
        real = [task.deadline for task in system.tasks]

        for e, r in zip(expected, real):
            self.assertAlmostEqual(e, r, delta=0.0001)

    def test_global_edf(self):
        system = to_edf(get_palencia_system())
        pd = PDAssignment(globalize=True)
        pd.apply(system)

        expected = [11.11111, 15.55555, 60, 16, 48, 80]
        real = [task.deadline for task in system.tasks]

        for e, r in zip(expected, real):
            self.assertAlmostEqual(e, r, delta=0.0001)



if __name__ == '__main__':
    unittest.main()
