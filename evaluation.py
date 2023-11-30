from analysis import init_wcrt
from generator import set_utilization
from model import System
import numpy as np
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime


class SchedRatioEval:
    """Class to perform a Schedulability Ratio evaluation"""
    def __init__(self, name, labels, funcs, systems, utilizations, threads):
        self.name = name                    # name of the study. used to name output files
        self.labels = labels                # label for each function (same length as funcs)
        self.funcs = funcs                  # test functions: each function maps system -> true/false
        self.systems = systems              # population of systems
        self.utilizations = utilizations    # utilizations array (usually: each number between [0,1], and increasing)
        self.threads = threads              # number of CPU threads to use
        self.start = 0                      # starting time
        assert len(labels) == len(funcs)

    def run(self):
        self.start = time.time()
        job = 0
        results = np.zeros((len(self.utilizations), len(self.labels)))  # schedulability ratio

        for u_index, u in enumerate(self.utilizations):
            # set utilization to every system
            for s in self.systems:
                set_utilization(s, u)

            # finish all the executions for one utilization before advancing to the next utilization
            # use a thread pool to accelerate the execution
            with Pool(self.threads) as pool:
                f = partial(self._step, u_index=u_index)
                for scheds in pool.imap_unordered(f, self.systems):
                    job += 1
                    results[u_index, :] += scheds
                    print(f"{datetime.now()} : u={u} job={job}")

            # update results file
            self._save(results)

    def _step(self, system: System, u_index: int):
        results = np.zeros(len(self.funcs), dtype=np.int8)
        for f, func in enumerate(self.funcs):
            init_wcrt(system)
            sched = func(system)
            if sched:
                results[f] = 1
        return results

    def _save(self, data):
        label = f"{self.name}_scheds"
        self._chart(label, data, ylabel="Schedulables", save=True, show=True)
        self._excel(label, data)

    def _chart(self, label, data, ylabel, save=True, show=True):
        plt.clf()
        df = pd.DataFrame(data=data,
                          index=self.utilizations,
                          columns=self.labels)
        fig, ax = plt.subplots()
        df.plot(ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Average utilization")

        # print system size
        ax.annotate(self.name, xy=(0, -0.1), xycoords='axes fraction', ha='left', va="center", fontsize=8)

        # register execution vector_times
        time_label = f"{time.time() - self.start:.2f} seconds"
        ax.annotate(time_label, xy=(1, -0.1), xycoords='axes fraction', ha='right', va="center", fontsize=8)
        fig.tight_layout()
        if save:
            fig.savefig(f"{label}.png")
        if show:
            plt.show()

    def _excel(self, label, data):
        df = pd.DataFrame(data=data,
                          index=self.utilizations,
                          columns=self.labels)
        df.to_excel(f"{label}.xlsx")
