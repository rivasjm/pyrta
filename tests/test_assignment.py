import unittest

import examples
from examples import get_palencia_system
from assignment import PDAssignment, EQFAssignment, EQSAssignment
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


class EQFTest(unittest.TestCase):
    def test_eqf(self):
        system = examples.get_palencia_system()
        eqf = EQFAssignment()
        eqf.apply(system)

        task11 = system[0][0]
        task12 = system[0][1]
        task13 = system[0][2]

        self.assertAlmostEqual(11.111111111, task11.deadline)
        self.assertAlmostEqual(5.4545454545, task12.deadline)
        self.assertAlmostEqual(60, task13.deadline)


class EQSTest(unittest.TestCase):
    def test_eqf(self):
        system = examples.get_palencia_system()
        eqs = EQSAssignment()
        eqs.apply(system)

        task11 = system[0][0]
        task12 = system[0][1]
        task13 = system[0][2]

        self.assertAlmostEqual(16, task11.deadline)
        self.assertAlmostEqual(21, task12.deadline)
        self.assertAlmostEqual(60, task13.deadline)


if __name__ == '__main__':
    unittest.main()
