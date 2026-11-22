# Agent A: Thm normalizable / Table normalizable. ghat = 1 (worst case).
import mpmath as mp
import numpy as np

mp.mp.dps = 40
ns = mp.nstr
L = 400


def kernels(Lags):
    j = np.arange(1, Lags + 1, dtype=float)
    out = {}
    for name, wts in [('exponential, width 10', np.exp(-j / 10)), ('exponential, width 100', np.exp(-j / 100)),
                      ('power law j^-1.5', j ** -1.5), ('Dirac', (j == 1).astype(float))]:
        out[name] = wts / wts.sum()
    return out


def companion_roots(wts, alpha, ghat=1.0):
    # z^{L+1} - alpha z^L + (1-alpha) ghat sum_j w_j z^{L-j} = 0
    Lg = len(wts)
    coeffs = np.zeros(Lg + 2)
    coeffs[0] = 1.0
    coeffs[1] = -alpha
    coeffs[2:] = (1 - alpha) * ghat * wts
    return np.roots(coeffs)


def polish(z0, wts, alpha):
    wm = [mp.mpf(float(x)) for x in wts]
    al = mp.mpf(str(alpha))
    def F(z):
        s = mp.mpc(0)
        zi = 1 / z
        p = mp.mpc(1)
        for x in wm:
            p *= zi
            s += x * p
        return z - al + (1 - al) * s
    return mp.findroot(F, mp.mpc(complex(z0)), tol=mp.mpf(10) ** (-30), verify=False)


paper = {('exponential, width 10', 1e-2): '0.97677', ('exponential, width 100', 1e-2): '0.99205',
         ('power law j^-1.5', 1e-2): '0.98656', ('Dirac', 1e-2): '0.97979'}
print("=== Table normalizable: kernel truncated at 400 lags and normalized over those lags; companion-matrix roots, polished in mpmath ===")
for eps in [1e-2, 1e-4]:
    alpha = 1 - eps
    for name, wts in kernels(L).items():
        r = companion_roots(wts, alpha)
        zmax = r[np.argmax(np.abs(r))]
        zp = polish(zmax, wts, alpha)
        print("  %-24s eps=%g  largest |z| = %s (z=%s)  decay/update = %s   paper: %s   all inside: %s" % (
            name, eps, ns(abs(zp), 8), ns(zp, 8), ns(-mp.log(abs(zp)), 4), paper.get((name, eps), '0.99980 (all four)'), bool(np.all(np.abs(r) < 1))))

print()
print("=== same kernels WITHOUT truncation (infinite sums, exact) ===")
for eps in ['1e-2', '1e-4']:
    e = mp.mpf(eps); al = 1 - e
    for width in [10, 100]:
        q = mp.exp(-mp.mpf(1) / width)
        # W = (1-q)/(z-q):  (z-al)(z-q) + (1-al)(1-q) = 0
        rts = mp.polyroots([1, -(al + q), al * q + (1 - al) * (1 - q)], maxsteps=200, extraprec=200)
        zm = max(rts, key=abs)
        print("  exponential width %-4d eps=%s  roots %s  largest |z| = %s" % (width, eps, [ns(x, 8) for x in rts], ns(abs(zm), 8)))
    rts = mp.polyroots([1, -al, 1 - al])
    print("  Dirac                  eps=%s  largest |z| = %s" % (eps, ns(max(abs(x) for x in rts), 8)))
    # power law j^-1.5 with full tail: W = Li_1.5(1/z)/zeta(1.5); polylog continued by mpmath for |1/z|>1
    z15 = mp.zeta(mp.mpf('1.5'))
    F = lambda z: z - al + (1 - al) * mp.polylog(mp.mpf('1.5'), 1 / z) / z15
    for start in [mp.mpc('0.98', '0.03'), mp.mpc('0.99', '0.005'), mp.mpc('0.9998', '0.0001')]:
        try:
            zr = mp.findroot(F, start, tol=mp.mpf(10) ** (-25), verify=False)
            print("  power law j^-1.5 (infinite) eps=%s start %s -> root %s  |z|=%s  |F|=%s" % (eps, ns(start, 4), ns(zr, 8), ns(abs(zr), 8), ns(abs(F(zr)), 3)))
        except Exception as ex:
            print("  power law (infinite) start %s failed: %s" % (ns(start, 4), type(ex).__name__))

print()
print("=== Remark powerlaw: 'damps at about 2(1-alpha) per update when (1-alpha) x mean lag is small' ===")
for eps in [1e-2, 1e-4, 1e-6]:
    alpha = 1 - eps
    for name, wts in kernels(L).items():
        r = companion_roots(wts, alpha)
        zmax = r[np.argmax(np.abs(r))]
        zp = polish(zmax, wts, alpha)
        mean_lag = float(np.dot(np.arange(1, L + 1), wts))
        print("  %-24s eps=%g  decay/update / (1-alpha) = %s   (1-alpha)*mean lag = %.3g   root is real: %s" % (
            name, eps, ns(-mp.log(abs(zp)) / eps, 6), eps * mean_lag, abs(zp.imag) < 1e-25))

print()
print("=== Thm normalizable: search for a counterexample (random nonnegative kernels, sum 1, 60 lags) ===")
rng = np.random.default_rng(12345)
worst = 0.0; n = 0
for trial in range(3000):
    Lg = int(rng.integers(1, 61))
    kind = trial % 4
    if kind == 0:
        wts = rng.random(Lg)
    elif kind == 1:
        wts = np.zeros(Lg); wts[rng.integers(0, Lg)] = 1.0          # single lag (delta at lag m)
    elif kind == 2:
        wts = rng.random(Lg) ** 8                                  # sparse-ish
    else:
        wts = np.zeros(Lg); wts[-1] = 1.0; wts[0] += rng.random() * 1e-3
    wts = wts / wts.sum()
    alpha = float(rng.choice([0.01, 0.3, 0.5, 0.9, 0.99, 0.9999, 1 - 1e-8]))
    ghat = float(rng.choice([1.0, 1.0, 0.7, 0.0]))
    r = companion_roots(wts, alpha, ghat)
    m = np.abs(r).max(); n += 1
    worst = max(worst, m)
print("  trials: %d, largest |root| seen: %.12f (theorem: < 1)" % (n, worst))
# the proof chain at its tightest: z=1, ghat=1 gives F(1) = 2(1-alpha); and delta at lag m near z=-1 (m even/odd)
for m in [1, 2, 3, 7]:
    for alpha in [0.5, 0.999999]:
        wts = np.zeros(m); wts[-1] = 1.0
        r = companion_roots(wts, alpha)
        print("  delta at lag %d, alpha=%s: max|z| = %.12f ; 1 - max|z| = %.3e ; (1-alpha) = %.1e" % (m, alpha, np.abs(r).max(), 1 - np.abs(r).max(), 1 - alpha))
