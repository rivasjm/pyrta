import sys

import assignment
import examples
import generator
import analysis
import simulator
import random
from model import *


# EDF-G seems to be too good to be true
# Select a schedulable system with utilization close to 1, simulate for very look and see
# if any simulated task overruns its wcrt
# try to find a system as small as possible


def callback(sim: simulator.Simulation, iteration, until, seed):
    print(sim.results.pessimism())
    min_pessimism = sim.results.pessimism()[0]
    if min_pessimism < 1:
        print(f"Anomaly detected with system={system.name}, iteration={iteration}, until={until}, seed={seed}")
        sys.exit()


def run(system: System):
    h = simulator.hyperperiod(system)
    print(f", hyperperiod={h} : ", end="")
    sim = simulator.SimRandomizer(system, seed=0, verbose=True)
    sim.run(until=h, iterations=100)


if __name__ == '__main__':
    for seed in range(1000):
        rnd = random.Random(seed)
        system = examples.get_small_system(random=rnd, utilization=0.99)
        system.name = f"seed={seed}"
        generator.to_edf(system, local=False)

        pd = assignment.PDAssignment(globalize=True)
        pd.apply(system)
        generator.to_int(system)

        holistic = analysis.HolisticGlobalEDFAnalysis()
        holistic.apply(system)
        print(f"System={system.name}, schedulable={system.is_schedulable()}", end="")

        run(system)

