from random import Random

from analysis import HolisticFPAnalysis
from assignment import PDAssignment
from examples import get_small_system
from fast_analysis import FastHolisticFPAnalysis
from gradient import PriorityExtractor, InvslackCost, StandardStop, AvgSeparationDelta, SequentialBatchCostFunction, \
    StandardGradient, NoisyAdam, StandardGradientDescent


if __name__ == '__main__':
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    fast_analysis = FastHolisticFPAnalysis(limit_factor=10, limit_p=1, limit_i=1, ceiling=False)
    extractor = PriorityExtractor()
    fast_cost_function = InvslackCost(extractor=extractor, analysis=fast_analysis)
    real_cost_function = InvslackCost(extractor=extractor, analysis=analysis)
    stop_function = StandardStop(limit=100)
    delta_function = AvgSeparationDelta(factor=1.5)
    batch_cost_function = SequentialBatchCostFunction(cost_function=fast_cost_function)
    gradient_function = StandardGradient(delta_function=delta_function,
                                         batch_cost_function=batch_cost_function)
    update_function = NoisyAdam()
    optimizer = StandardGradientDescent(extractor=extractor,
                                        cost_function=real_cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        ref_cost_function=real_cost_function,
                                        verbose=True)

    r = Random(42)
    pd = PDAssignment(normalize=True)
    system = get_small_system(random=r, utilization=0.7, balanced=True)
    pd.apply(system)

    x = optimizer.apply(system)
    extractor.insert(system, x)
    analysis.apply(system)
    print(system.is_schedulable())
