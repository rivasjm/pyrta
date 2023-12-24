import unittest
import sim
import examples
import generator
from random import Random
import analysis
import math
import assignment


class SimulatorTest(unittest.TestCase):
    rnd = Random(42)
    system = examples.get_medium_system(random=rnd, utilization=0.5)

    pd = assignment.PDAssignment()
    pd.apply(system)
    generator.to_int(system)

    holistic = analysis.HolisticFPAnalysis(reset=False)
    holistic.apply(system)

    sim = sim.Simulation(system, verbose=False)
    time = math.lcm(*[f.period for f in system.flows])
    sim.run(until=2000000)

    print(sim.results.repr())
    print(analysis.repr_wcrts(system))
    print(sim.results.pessimism())