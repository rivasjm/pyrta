from examples import get_big_system
from generator import to_edf
import random
from assignment import HOPAssignment
from model import SchedulerType
from analysis import HolisticGlobalEDFAnalysis, reset_wcrt


def testing():
    rnd = random.Random(1)
    s = get_big_system(rnd, utilization=0.9, balanced=False, sched=SchedulerType.EDF)
    holistic = HolisticGlobalEDFAnalysis()
    hopa = HOPAssignment(analysis=holistic, globalize=True, verbose=True)

    hopa.apply(s)
    reset_wcrt(s)
    holistic.apply(s)
    print(s.is_schedulable())


if __name__ == '__main__':
    testing()
