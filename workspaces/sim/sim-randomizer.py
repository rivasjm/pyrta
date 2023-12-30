import simulator
import examples
import random
import generator

def run():
    rnd = random.Random(42)
    system = examples.get_medium_system(random=rnd, balanced=True)
    generator.to_int(system)

    sim = simulator.Simulation(system=system, verbose=False)
    sim.run(until=10000000)
    flow_results1 = [sim.results.flow_wort(flow) for flow in system.flows]
    print(flow_results1)

    sim = simulator.SimRandomizer(system, seed=0)
    sim.run(until=1000000, iterations=10)  # 100000 * 10 = 1000000
    flow_results2 = [simulator.max_flow_wort(flow, sim.results) for flow in system.flows]
    print(flow_results2)


if __name__ == '__main__':
    run()
