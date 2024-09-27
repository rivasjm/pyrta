import examples
from random import Random
from assignment import PDAssignment, repr_priorities
from analysis import HolisticFPAnalysis, reset_wcrt
import matplotlib.pyplot as plt
from metrics import invslack
import numpy as np

r = Random(42)
s = examples.get_small_system(r)
pd = PDAssignment()
holistic = HolisticFPAnalysis(reset=False)

s = pd(s)
print(repr_priorities(s))

min_prio = min([t.priority for t in s.tasks])*0.9
max_prio = max([t.priority for t in s.tasks])*1.1

fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 8))
for ax, task in zip(axs.flat, s.tasks):
    x = []
    y = []
    c = []
    back_prio = task.priority
    for p in np.linspace(min_prio, max_prio, 100):
        reset_wcrt(s)
        task.priority = p
        s = holistic(s)
        cost = invslack(s)
        x.append(p)
        y.append(task.wcrt)
        c.append(cost)
    ax.plot(x, y, color='blue')
    ax2 = ax.twinx()
    ax2.plot(x, c, color='red')
    task.priority = back_prio

plt.show()








