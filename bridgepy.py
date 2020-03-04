#!/bin/python

import sys, fitparam, models, statistics, misc
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pylab as plt

## Loading the options
argvs = sys.argv
if len(argvs) == 1 or argvs[1] == "-h" or argvs[1] == "--help":
    error = "\n bridgepy v1.0 [08-31-2019, implemented by Kentaro KAWATA]                                                   \n" +\
            "                                                                                                                   \n" +\
            "  Usage:                                                                                                           \n" +\
            "    python calcrate.py -t [Time points] -S [Synthesis table] -D [Degradation table]                                \n" +\
            "                                                                                                                   \n" +\
            "  Input:                                                                                                           \n" +\
            "    expression_table :                                                                                             \n" +\
            "                                                                                                                   \n" +\
            "  Parameters:                                                                                                      \n" +\
            "    --synthesis   (-S )  Synthesis table                       (mandatory)    [ e.g. ./synthesis.table   ]         \n" +\
            "    --degradation (-D )  Degradation table                     (mandatory)    [ e.g. ./degradation.table ]         \n" +\
            "    --timepoints  (-t )  Time points separated with comma.     (mandatory)    [ e.g. 0,30,60,120,240     ]         \n" +\
            "    --outdir      (-o )  Directory for saving output figures.                 [ e.g. ./figure            ]         \n" +\
            "    --label1      (-l1)  Directory for saving output figures.                 [ e.g. ./figure            ]         \n" +\
            "    --label2      (-l2)  Directory for saving output figures.                 [ e.g. ./figure            ]         \n" +\
            "    --threads     (-p )  Number of processors/core                            [ e.g. 4                   ]         \n" +\
            "                                                                                                                   \n" +\
            "  Example:                                                                                                         \n" +\
            "    $ python calcrates.py -x transcription -t 0,15,30,45,60,120,240,480,720 -o ./Result/fitting/ cpm.tcreads.table \n" +\
            "                                                                                                                   \n"
    sys.exit(error)
else:
    sf   = "None"
    df   = "None"
    time = "None"
    l1   = "Data1"
    l2   = "Data2"
    pfx  = "./"
    cnum = 1
    for i in range(len(argvs)): 
        if argvs[i] == "-S"  or argvs[i] == "--synthesis":
            sf   = argvs[i+1]
        if argvs[i] == "-D"  or argvs[i] == "--degradation":
            df   = argvs[i+1]
        if argvs[i] == "-t"  or argvs[i] == "--timepoints":
            time = map(int, argvs[i+1].split(","))
        if argvs[i] == "-o"  or argvs[i] == "--outdir":
            pfx  = argvs[i+1]
        if argvs[i] == "-l1" or argvs[i] == "--label1":
            l1   = argvs[i+1]
        if argvs[i] == "-l2" or argvs[i] == "--label2":
            l2   = argvs[i+1]
        if argvs[i] == "-p"  or argvs[i] == "--threads":
            cnum = int(argvs[i+1])

    if sf == "None" or df == "None" or time == "None":
        error = "Error!! : Synthesis (-S), degradation (-D), and time points (-t) are mandatry.\n" +\
                "          Please make sure that these files do exist certainly.               \n"
        sys.exit(error)


class OptimizeParams():
    def __init__(self, time, sdat, ddat):
        self.time = time
        self.sdat = sdat
        self.ddat = ddat
        self.pfx  = pfx
    def optparam(self):
        self.param = fitparam.evolutionary(self.time, self.sdat, self.ddat, self.pfx)
        self.param = fitparam.NLP(self.time, self.sdat, self.ddat, self.param)
        return self.param
    def statparam(self):
        ks, kd, t0, a, b1, b2 = self.param        
        self.sest  = [models.S_model(t, t0, ks, kd, b1) for t in self.time]
        self.dest  = [models.D_model(t, a, kd, b2)      for t in self.time]
        self.value = np.vstack((self.sdat.values, self.ddat.values))
        self.pred  = np.vstack((self.sest, self.dest))
        self.stats = statistics.testcorr(self.sdat.name, self.value, self.pred, self.pfx)
        return self.stats


def fitting(ds):
    sdat   = ds.filter(regex=l1)
    ddat   = ds.filter(regex=l2)
    optmz  = OptimizeParams(time, sdat, ddat)
    param  = optmz.optparam()
    stats  = optmz.statparam()
    result = pd.concat([sdat, ddat])
    result = result.append(pd.Series(param   , index=['ks', 'kd', 't0', 'a', 'b1', 'b2']))
    result = result.append(pd.Series(stats, index=["Coef", "StdErr", "R2", "t-value", "p-value"]))
    return result


def run():
    sdf = pd.read_csv(sf, sep='\t', index_col=0, names=["ID"] + [l1 + "_" + str(t) for t in time])
    ddf = pd.read_csv(df, sep='\t', index_col=0, names=["ID"] + [l2 + "_" + str(t) for t in time])
    sln = len(sdf.columns)
    dln = len(ddf.columns)
    tln = len(time)
    if tln != sln or tln != dln:
        error = "Error!!: The length of the synthesis and degradation table must be the same as the number of time points.\n" +\
                "   - length of synthesis table   : " + str(sln) + "\n"                                                       +\
                "   - length of degradation table : " + str(dln) + "\n"                                                       +\
                "   - number of time points       : " + str(tln)
        sys.exit(error)

    cdf = pd.merge(sdf, ddf, on="ID", how='inner')
    daf = dd.from_pandas(cdf, npartitions=cnum)
    result = daf.apply(fitting, axis=1).compute(scheduler='processes')
    result = result.assign(FDR=statistics.fdr(result["p-value"]))
    result = result.sort_values(by=["p-value"], ascending=True)
    result.to_csv(pfx + "/parameters.txt", sep="\t")
    misc.plot_data(time, result, pfx)

    sidx = set(sdf.index)
    didx = set(ddf.index)
    for idx in sidx - didx:
        sys.stderr.write("Caution !!: " + idx + "\tis unique for synthesis table specified by --sinthesis (-S) option.\n")
    for idx in didx - sidx:
        sys.stderr.write("Caution !!: " + idx + "\tis unique for degradation table specified by --degradation (-D) option.\n")


if __name__ == "__main__":
    run()

