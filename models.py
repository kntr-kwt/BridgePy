import math

def S_model(t, t0, ks, kd, b1):
    return b1 + (ks/kd)*(1 - math.e**(-1*kd*(t-t0)*(t>t0)))

def D_model(t, a, kd, b2):
    return b2 + a*(math.e**(-1*kd*t))


