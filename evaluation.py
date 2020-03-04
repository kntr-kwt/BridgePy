import models
import pandas as pd
from operator import mul


def RSS(param, time, S_dat, D_dat, weights=(1.0, 1.0)):
    if not isinstance(S_dat, pd.Series): S_dat=pd.Series(S_dat)
    if not isinstance(D_dat, pd.Series): D_dat=pd.Series(D_dat)
    ks, kd, t0, a, b1, b2 = param
    S_est = [models.S_model(t, t0, ks, kd, b1) for t in time]
    D_est = [models.D_model(t, a, kd, b2)      for t in time]
    S_RSS = sum((S_dat - S_est)**2)
    D_RSS = sum((D_dat - D_est)**2)
    return sum(map(mul, [S_RSS, D_RSS], weights))


if __name__=='__main__':
    param = [10, 0.01, 0, 10, 0.1, 0.1]
    time  = [0,15,30,45,60,120,240,480,720]
    S_dat = [ 0.113, 9.392, 9.282, 2.472, 1.288, 8.906, 9.382, 1.870, 9.353]
    D_dat = [10.100, 8.707, 7.508, 6.476, 5.588, 3.111, 1.007, 0.182, 0.107]
    print RSS(param, time, S_dat, D_dat)
