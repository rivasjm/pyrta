import unittest

import numpy as np

from examples import get_three_tasks
from fast_gradient import FastGDPA
from fast_analysis import func_w, fast_converge, FastHolisticFPAnalysis
from functools import partial
from analysis import init_wcrt
from gradient import InvslackCost, PriorityExtractor


class FastGDPATest(unittest.TestCase):
    def test_three_tasks(self):
        s = get_three_tasks()
        deadlines = np.array([t.flow.deadline for t in s.tasks])

        # conventional
        init_wcrt(s)
        fast_analysis = FastHolisticFPAnalysis(limit_p=1, limit_i=1, ceiling=False, fast=True)
        fast_analysis.apply(s)
        res1 = np.array([t.wcrt for t in s.tasks]).reshape((-1, 1))

        # vectorized
        init_wcrt(s)
        fgdpa = FastGDPA()
        res2 = fgdpa.apply(s)

        print(res1)
        print(res2)

        i1 = np.max((res1 - deadlines.T) / deadlines.T)
        i2 = np.max((res2 - deadlines.T) / deadlines.T)
        print(i1)
        print(i2)
