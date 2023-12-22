import unittest

import generator
import examples
from random import Random


class GeneratorTest(unittest.TestCase):
    def test_unbalance_palencia(self):
        system = examples.get_palencia_system()
        u_ini = sum([proc.utilization for proc in system.processors])
        generator.unbalance(system)
        u_end = sum([proc.utilization for proc in system.processors])
        self.assertAlmostEqual(u_ini, u_end)

    def test_unbalance_medium(self):
        r = Random(42)
        system = examples.get_medium_system(random=r)
        u_ini = sum([proc.utilization for proc in system.processors])

        generator.unbalance(system)
        us_end = [proc.utilization for proc in system.processors]
        u_end = sum(us_end)

        print(us_end)
        self.assertTrue(all([u < 1 for u in us_end]))
        self.assertAlmostEqual(u_ini, u_end)

    def test_system_utilization(self):
        r = Random(42)
        system = examples.get_medium_system(random=r, utilization=0.5)
        tasks = system.tasks

        target_utilization = 0.7
        generator.set_system_utilization(system, target_utilization)
        u = sum([t.utilization for t in tasks])

        self.assertEqual(target_utilization*len(system.processors), u)

