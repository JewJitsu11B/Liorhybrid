# Agent E side note (NUMERIC, not exact): attribute the Table normalizable "width 100" gap.
import numpy as np
a = 0.99
for width in (10, 100):
    for N in (400, 4000):
        for renorm in (True, False):
            j = np.arange(1, N+1)
            w = np.exp(-j/width); w = w/w.sum() if renorm else w*(1-np.exp(-1/width))/np.exp(-1/width)
            c = np.zeros(N+2); c[0] = 1; c[1] = -a; c[2:] = (1-a)*w      # z^{N+1} - a z^N + (1-a) sum w_j z^{N-j}
            r = np.roots(c)
            print("width", width, "lags", N, "renormalised" if renorm else "untruncated-normalisation", "sum w=%.6f" % w.sum(), "max|z| = %.5f" % abs(r).max())
