import sys, os, matplotlib, pyper, random
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pylab as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def fdr(pvals, method="BH"):
    r = pyper.R()
    r.assign("pvals", pvals.values.tolist())
    if method == "Storey":
        r('library(qvalue)')
        r('qobj  <- qvalue(pvals)')
        r('qvals <- qobj$qvalue')
    else:
        r('qvals <- p.adjust(pvals, method="fdr")')
    qvals = r.get('qvals')
    return qvals


def normalization(x1, x2, axis=1):
    xmean = x1.mean(axis=axis, keepdims=True)
    xstd  = np.std(x1, axis=axis, keepdims=True)
    return (x1-xmean)/xstd, (x2-xmean)/xstd


def testcorr(name, value, pred, prefix):
    value, pred = normalization(value, pred)
    x       = pd.Series(np.ravel(value))
    y       = pd.Series(np.ravel(pred ))

    if ~np.any(np.isinf(x)) and ~ np.any(np.isinf(y)):
        model   = sm.OLS(x, y)
        results = model.fit()
        coef    = results.params[0]
        stderr  = results.bse[0]
        rsq     = results.rsquared
        tval    = results.tvalues[0]
        pval    = results.pvalues[0] if not np.isnan(results.pvalues[0]) else np.float(1)
    else:
        coef    = np.nan
        stderr  = np.nan
        rsq     = np.nan
        tval    = np.nan
        pval    = np.float(1)

    sys.stderr.write(name + ":   Start of statistical test for estimated values\n")
    sys.stderr.write("----------------------------------------------------------------------\n")
    sys.stderr.write("   Coefficient. : %s\n" % coef  )
    sys.stderr.write(" Standard error : %s\n" % stderr)
    sys.stderr.write("       R-square : %s\n" % rsq   )
    sys.stderr.write("        t-value : %s\n" % tval  )
    sys.stderr.write("        p-value : %s\n" % pval  )
    sys.stderr.write("----------------------------------------------------------------------\n")
    sys.stderr.write(name + ":   Finish of statistical test for estimated values\n")

    tdir = prefix + "/Evaluation/"
    if not os.path.isdir(tdir): os.mkdir(tdir)
    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(1,1,1)    
    ax.scatter(value[0], pred[0], c='red' )
    ax.scatter(value[1], pred[1], c='blue')
    ax.plot(x, coef*x, "k:")
    ax.text(0.7,0.1, "y = " + "{0:.2f}".format(coef) + "x\n" +\
                     "R2 = " + "{0:.4f}".format(rsq)  +  "\n" +\
                     "p-value = " + "{0:.4f}".format(pval)
            ,transform=ax.transAxes)
    plt.savefig(tdir + name + ".png")
    plt.close()

    return coef, stderr, rsq, tval, pval
    

if __name__ == '__main__':
    pvals = pd.Series([random.random() for _ in range(20)])
    qvals = fdr(pvals, method="Storey")
    print pvals, type(pvals), len(pvals)
    print qvals, type(qvals), len(qvals)
