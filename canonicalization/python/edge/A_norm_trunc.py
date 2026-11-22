# Agent A: dependence of Table normalizable rows 2 and 3 on the 400-lag truncation.
import numpy as np
from A_normalizable import companion_roots
for L in [400, 800, 1600, 3200]:
    j = np.arange(1, L + 1, dtype=float)
    out = []
    for wts in [np.exp(-j / 100), j ** -1.5]:
        wts = wts / wts.sum()
        r = companion_roots(wts, 0.99)
        out.append(np.abs(r).max())
    print("lags=%d  exponential width 100: %.5f   power law j^-1.5: %.5f" % (L, out[0], out[1]))
