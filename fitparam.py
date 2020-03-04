import os, sys, random, math, inspect, matplotlib
import models, misc, evaluation
matplotlib.use('Agg')

from deap       import base
from deap       import creator
from deap       import tools
from scipy      import optimize
from scipy      import stats
from evaluation import RSS

import numpy  as np
import pandas as pd
import matplotlib.pylab as plt


## Default fitting parameters
CXPB, MUTPB, NGEN, NPOP = 0.5, 0.2, 200, 50

##
## Default Upper and lower boundaries of the six parameters estimated in this script.
##
##  ks: transcription rate
##  kd: degradation rates
##  t0: time delay     for transcription time series
##  a : scaling factor for degradation timeserie
##  b1: basal values   for transcription timeseries
##  b2: basal values   for degradation timeseries
##
upper = [10**3  , 10**-0.3, 30, 5000, 10.0, 10.0] # upper bounds of the parameters
lower = [10**-10, 10**-8  ,  0,    1,  0.0,  0.0] # lower bounds of the parameters


def NLP(time, sdat, ddat, init, lower=lower, upper=upper):
    name = sdat.name
    sys.stderr.write(name + ":   Start of nonlinear programming\n")
    sys.stderr.write("----------------------------------------------------------------------\n")
    boundary = np.array([[lower[i], upper[i]] for i in range(len(init))])
    param = optimize.minimize(RSS, init, args=(time, sdat, ddat), bounds=boundary, method='L-BFGS-B')
    sys.stderr.write("%s\n" % param)
    sys.stderr.write("----------------------------------------------------------------------\n")
    sys.stderr.write(name + ":   Finish of nonlinear programming\n")
    return param.x


def evolutionary(time, sdat, ddat, prefix, lower=lower, upper=upper, CXPB=CXPB, MUTPB=MUTPB, NGEN=NGEN, NPOP=NPOP):
    random.seed()
    name = sdat.name

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_random", random.uniform)
    toolbox.register("individual", misc.genIndividual, creator.Individual, toolbox.attr_random, 6, lower, upper)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda p, t, s, d: (RSS(p, t, s, d), ), t=time, s=sdat, d=ddat)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", misc.mutGaussianLimit, mu=[0.0]*len(upper), sigma=map(lambda x: x/1, upper), indpb=0.5, lower=lower, upper=upper)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=NPOP)
    sys.stderr.write(name + ":   Start of evolution\n")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    sys.stderr.write(name + ":   Evaluated %i individuals\n" % len(pop))
    
    trj = ["NaN"] * NGEN
    for g in range(NGEN):
        sys.stderr.write(name + ": -- Generation %i --\n" % g)
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        sys.stderr.write(name + ":   Evaluated %i individuals\n" % len(invalid_ind))
        
        pop[:] = offspring
        fits   = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean   = sum(fits) / length
        sum2   = sum(x*x for x in fits)
        std    = abs(sum2 / length - mean**2)**0.5
        trj[g] = min(fits)

        sys.stderr.write(name + ":   Min %s\n" % min(fits))
        sys.stderr.write(name + ":   Max %s\n" % max(fits))
        sys.stderr.write(name + ":   Avg %s\n" % mean)
        sys.stderr.write(name + ":   Std %s\n" % std)
    
    sys.stderr.write(name + ": -- End of (successful) evolution --\n")

    tdir = prefix + "/Trajectory/"
    if not os.path.isdir(tdir): os.mkdir(tdir)
    plt.figure(figsize=(8.0, 5.0))
    plt.plot(range(NGEN), trj, drawstyle='steps')
    plt.savefig(tdir + name + ".png")
    plt.close()    

    best_ind = tools.selBest(pop, 1)[0]
    sys.stderr.write(name + ": Best individual is %s, %s\n" % (best_ind, best_ind.fitness.values))    

    return best_ind

    ## https://qiita.com/neka-nat@github/items/0cb8955bd85027d58c8e
    ## http://darden.hatenablog.com/entry/2017/04/18/225459
    ## http://darden.hatenablog.com/entry/2017/03/29/213948#%E3%82%B9%E3%83%86%E3%83%83%E3%83%973-%E4%BA%A4%E5%8F%89

