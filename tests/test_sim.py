import unittest
import simulator
import examples
import generator
from random import Random
import analysis
import math
import assignment
from model import *


class SimulatorTest(unittest.TestCase):
    def sanity_check(self):
        rnd = Random(42)
        system = examples.get_medium_system(random=rnd, utilization=0.5)

        pd = assignment.PDAssignment()
        pd.apply(system)
        generator.to_int(system)

        holistic = analysis.HolisticFPAnalysis(reset=False)
        holistic.apply(system)

        sim = simulator.Simulation(system, verbose=False)
        time = math.lcm(*[f.period for f in system.flows])
        sim.run(until=time)

        pessimism= sim.results.pessimism()
        self.assertAlmostEqual(pessimism[0], 0)

    def test_fp(self):
        proc = Processor("p1", sched=SchedulerType.FP)
        a = Task("a", wcet=2, processor=proc, type=TaskType.Activity, priority=1)
        fa = Flow("fa", period=10, deadline=10)
        fa.add_tasks(a)

        b = Task("b", wcet=2, processor=proc, type=TaskType.Activity, priority=10)
        fb = Flow("fb", period=10, deadline=10)
        fb.phase = 1
        fb.add_tasks(b)

        system = System()
        system.add_procs(proc)
        system.add_flows(fa, fb)

        sim = simulator.Simulation(system, verbose=False)
        sim.run(5)

        self.assertEqual(sim.results.intervals(a), [(0, 1), (3, 4)])
        self.assertEqual(sim.results.intervals(b), [(1, 3)])

    def test_edfl(self):
        proc = Processor("p1", sched=SchedulerType.EDF, local=True)
        a = Task("a", wcet=2, processor=proc, type=TaskType.Activity, deadline=10)
        fa = Flow("fa", period=10, deadline=10)
        fa.add_tasks(a)

        b = Task("b", wcet=2, processor=proc, type=TaskType.Activity, deadline=1)
        fb = Flow("fb", period=10, deadline=10)
        fb.phase = 1
        fb.add_tasks(b)

        system = System()
        system.add_procs(proc)
        system.add_flows(fa, fb)

        sim = simulator.Simulation(system, verbose=False)
        sim.run(5)

        self.assertEqual(sim.results.intervals(a), [(0, 1), (3, 4)])
        self.assertEqual(sim.results.intervals(b), [(1, 3)])

    def test_edfg(self):
        proc = Processor("p1", sched=SchedulerType.EDF, local=False)
        a = Task("a", wcet=2, processor=proc, type=TaskType.Activity, deadline=10)
        fa = Flow("fa", period=10, deadline=10)
        fa.add_tasks(a)

        b = Task("b", wcet=2, processor=proc, type=TaskType.Activity, deadline=1)
        fb = Flow("fb", period=10, deadline=10)
        fb.phase = 1
        fb.add_tasks(b)

        system = System()
        system.add_procs(proc)
        system.add_flows(fa, fb)

        sim = simulator.Simulation(system, verbose=False)
        sim.run(5)

        self.assertEqual(sim.results.intervals(a), [(0, 1), (3, 4)])
        self.assertEqual(sim.results.intervals(b), [(1, 3)])