import unittest

import numpy as np

from examples import get_three_tasks
from fast_gradient import FastGDPA
from fast_analysis import func_w, fast_converge
from functools import partial
from analysis import init_wcrt


class FastGDPATest(unittest.TestCase):
    def test_three_tasks(self):
        s = get_three_tasks()
        init_wcrt(s)

        # conventional
        res1 = []
        for task in s.tasks:
            f = partial(func_w, task, 1, ceiling=False)
            res1.append(fast_converge(f, task.flow.deadline - task.jitter))

        # vectorized
        fgdpa = FastGDPA()
        res2 = fgdpa.apply(s)

        res1 = np.array(res1).reshape((3,1))
        print(res1)
        print(res2)
        self.assertTrue(np.allclose(res1, res2))
