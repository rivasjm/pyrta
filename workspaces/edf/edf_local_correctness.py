import pickle
import sys

import assignment
import examples
import generator
import analysis
import simulator
import gradient
import random
from model import *


# GDPA with current Holistic analysis determines that negative local deadlines are best
# We suspect that the analysis does not take into account correctly the negative deadlines
# Generate many systems, apply GDPA, obtain WCRT's with Holistic and WORTS with simulation
# to try to find simulated values that are higher than the analysis, to indicate that the
# analysis is not correct


def edf_local_gdpa(system: System):
    holistic = analysis.HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    extractor = gradient.DeadlineExtractor()
    cost_function = gradient.InvslackCost(extractor=extractor, analysis=holistic)
    stop_function = gradient.StandardStop(limit=100)
    delta_function = gradient.AvgSeparationDelta(factor=1.5)
    batch_cost_function = gradient.SequentialBatchCostFunction(cost_function=cost_function)
    gradient_function = gradient.StandardGradient(delta_function=delta_function,
                                                  batch_cost_function=batch_cost_function)
    update_function = gradient.NoisyAdam()
    optimizer = gradient.StandardGradientDescent(extractor=extractor,
                                                 cost_function=cost_function,
                                                 stop_function=stop_function,
                                                 gradient_function=gradient_function,
                                                 update_function=update_function,
                                                 verbose=False)

    assignment.PDAssignment().apply(system)
    optimizer.apply(system)


def callback(sim: simulator.Simulation, iteration, until, seed):
    # print(sim.results.pessimism())
    print(".", end="")
    min_pessimism = sim.results.pessimism()[0]
    if min_pessimism < 1:
        print(f"Anomaly detected with system={sim.system.name}, iteration={iteration}, until={until}, seed={seed}")
        sys.exit()


def apply_anomaly_deadlines(system):
    with open("negatives.pkl", "rb") as f:
        deadlines = pickle.load(f)
        for t,d in zip(system.tasks, deadlines):
            t.deadline = d


def run_anomaly():
    seed = 7
    rnd = random.Random(seed)
    system = examples.get_small_system(random=rnd, utilization=0.8, balanced=True)
    system.name = f"seed({seed})"
    generator.to_edf(system, local=True)
    generator.to_int(system)
    print(f"System={system.name}")

    # assign local deadlines with GDPA (hopefully they become negative)
    # edf_local_gdpa(system)
    apply_anomaly_deadlines(system)
    print("   GDPA " + assignment.repr_deadlines_mini(system))

    # apply holistic analysis to obtain WCRT's
    holistic = analysis.HolisticLocalEDFAnalysis(reset=False)
    holistic.apply(system)
    print(f"   holistic schedulable={system.is_schedulable()}")

    # apply simulator to obtain WORT's.
    # the simulator is applied 100 times until the hyperperiod. Each time with a different transaction phases
    # the callback is called after one of these 100 runs
    h = simulator.hyperperiod(system)
    print(f"   simulation hyperperiod={h} : ", end="")
    sim = simulator.SimRandomizer(system, seed=0, verbose=True, callback=callback)
    sim.run(until=h, iterations=100)
    print("")


def run():
    for seed in range(1000):
        rnd = random.Random(seed)
        system = examples.get_small_system(random=rnd, utilization=0.8, balanced=True)
        system.name = f"seed({seed})"
        generator.to_edf(system, local=True)
        generator.to_int(system)
        print(f"System={system.name}")

        # assign local deadlines with GDPA (hopefully they become negative)
        edf_local_gdpa(system)
        print("   GDPA " + assignment.repr_deadlines_mini(system))

        # apply holistic analysis to obtain WCRT's
        holistic = analysis.HolisticLocalEDFAnalysis(reset=False)
        holistic.apply(system)
        print(f"   holistic schedulable={system.is_schedulable()}")

        # apply simulator to obtain WORT's.
        # the simulator is applied 100 times until the hyperperiod. Each time with a different transaction phases
        # the callback is called after one of these 100 runs
        h = simulator.hyperperiod(system)
        print(f"   simulation hyperperiod={h} : ", end="")
        sim = simulator.SimRandomizer(system, seed=0, verbose=True, callback=callback)
        sim.run(until=h, iterations=100)
        print("")


if __name__ == '__main__':
    run()