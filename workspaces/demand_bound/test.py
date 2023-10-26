import matplotlib.pyplot as plt
import numpy as np
from functools import partial


c = np.array([5,  10, 15])
t = np.array([50, 50, 50])
j = np.array([5,  10, 15])
prio = np.array([10, 5, 1]).reshape((3, 1))


def func_w(p, i, hp, w):
    return p*c[i] + sum([np.ceil((j[task]+w)/t[task])*c[task]*hp[task] for task in range(len(c))])


def func_r(p, i, hp, r):
    return (p*c[i] - (p-1)*t[i] + j[i] +
            sum([np.ceil((j[task]+r+(p-1)*t[i]+j[i])/t[task])*c[task]*hp[task] for task in range(len(c))]))


def converge(f, x):
    v = f(x)
    if v != x:
        return converge(f, v)
    else:
        return v


def plot_w(i, p, ax):
    pm = prio > prio.T
    hp = pm[:, i]

    x = np.linspace(0, 200, 1000)
    f = partial(func_w, p, i, hp)

    sol = converge(f, 0)

    # DBF
    ax.plot(x, f(x), color='red')

    # w=w
    ax.plot(x, x, color='blue')

    # limit
    ax.axhline(y=sol, color='g', linestyle='-')

    # p limit
    ax.axhline(y=p*t[i], color='orange')

    ax.text(sol, sol-20, f"{sol}")
    ax.text(0.05, 0.9, f"p={p}", transform=ax.transAxes)


def plot_r(i, p, ax):
    pm = prio > prio.T
    hp = pm[:, i]

    x = np.linspace(0, 200, 1000)
    f = partial(func_r, p, i, hp)

    sol = converge(f, 0)

    # DBF
    ax.plot(x, f(x), color='red')

    # w=w
    ax.plot(x, x, color='blue')

    # limit
    ax.axhline(y=sol, color='g', linestyle='-')

    # p limit
    ax.axhline(y=p*t[i]-(p-1)*t[i]+j[i], color='orange')

    ax.text(sol, sol-20, f"{sol}")
    ax.text(0.05, 0.9, f"p={p}", transform=ax.transAxes)


if __name__ == '__main__':
    i = 2

    fig, axs = plt.subplots(2, 2)
    for p, ax in enumerate(axs.flat):
        plot_r(i, p+1, ax)
        p = p+1

    plt.show()
