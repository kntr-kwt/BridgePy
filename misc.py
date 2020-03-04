import os, sys, math, random, models, matplotlib
matplotlib.use('Agg')

from itertools import repeat
from collections import Sequence
import numpy as np
import matplotlib.pylab as plt


def genIndividual(container, func, n, lower, upper):
    return container(func(lower[i],upper[i]) for i in xrange(n))


def mutGaussianLimit(individual, mu, sigma, indpb, lower, upper):
    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))
    for i, m, s in zip(xrange(size), mu, sigma):
        if random.random() < indpb:
            previous = individual[i]
            individual[i] += random.gauss(m, s)
            while individual[i] <= lower[i] or upper[i] <= individual[i]:
                if individual[i] <= lower[i]:
                    individual[i] = previous
                if upper[i] <= individual[i]:
                    individual[i] = previous
                individual[i] += random.gauss(m, s)
    return individual,


def plot_data(time, df, prefix):
    pnum  = -1
    names = df.index.values.tolist()
    tdir  = prefix + "/Figures/"
    for i in range(len(df)):
        pnum += 1
        fgnum = (pnum / 40) + 1
        axnum = (pnum % 40)

        name  = names[i].split(":")[1]
        svals = df.iloc[i,0:9]
        dvals = df.iloc[i,9:18]
        ks    = df.iloc[i,18]
        kd    = df.iloc[i,19]
        t0    = df.iloc[i,20]
        a     = df.iloc[i,21]
        b1    = df.iloc[i,22]
        b2    = df.iloc[i,23]
        fdr   = df.iloc[i,29]
        sest  = [models.S_model(t, t0, ks, kd, b1) for t in range(time[-1])]
        dest  = [models.D_model(t, a, kd, b2)      for t in range(time[-1])]

        if axnum == 0:
            fig   = plt.figure(fgnum, figsize=(19.20, 10.80))
            axes  = []

        ax1 = fig.add_subplot(5, 8, axnum+1)
        ax1.scatter(time, svals, c="red")
        ax1.plot(range(time[-1]), sest, linewidth=1, color="red")
        ax2 = ax1.twinx()
        ax2.scatter(time, dvals, c="blue")
        ax2.plot(range(time[-1]), dest, linewidth=1, color="blue")
        plt.xlim(0, 720)
        plt.xticks(np.arange(0, 720 + 1, 180))
        plt.title(name + " : " + '{:.3e}'.format(fdr))

        sys.stderr.write("Name: " + name + "\tPlot: " + str(pnum) + "\tFigure: " + str(fgnum) + "\tAxis: " + str(axnum) + "\n")

        if axnum == 39 or pnum == len(df)-1:
            if not os.path.isdir(tdir): os.mkdir(tdir)
            plt.tight_layout()
            plt.savefig(prefix + "/Figures/plot_" + str(fgnum).zfill(5) + ".png")
            plt.close(fig)
