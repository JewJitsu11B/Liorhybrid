# Agent A: Thm feedback / Table feedback.  alpha > 1, k = 0 (ghat = 1).
# Characteristic function with brake b and fading lambda:
#   h(z) = z - (1-b) (alpha + (alpha-1) W(z e^lambda)),   W_lambda(z) = W(z e^lambda).
import mpmath as mp
import numpy as np
from A_common import *

mp.mp.dps = 40
I = mp.mpc(0, 1)
ns = mp.nstr


def Wreal(t, alpha):
    """W(e^t) for real t > 0, by the continuation formula (cross-checked vs direct sums)."""
    return W_jonq(-I * t, alpha).real


def h_t(t, alpha, b, lam):
    return mp.exp(t) - (1 - b) * (alpha + (alpha - 1) * Wreal(t + lam, alpha))


def real_root(alpha, b, lam):
    """Unique real root z = e^t with t > -lam (h increases from -inf to +inf there)."""
    lo = -lam + mp.mpf(10) ** (-15)
    hi = mp.mpf(5)
    assert h_t(lo, alpha, b, lam) < 0 < h_t(hi, alpha, b, lam)
    for _ in range(60):
        mid = (lo + hi) / 2
        if h_t(mid, alpha, b, lam) < 0:
            lo = mid
        else:
            hi = mid
    t = mp.findroot(lambda t: h_t(t, alpha, b, lam), (lo, hi), solver='secant', tol=mp.mpf(10) ** (-30), verify=False)
    return t


print("=== sanity: W(e^t) for alpha>1 by continuation vs direct sum ===")
for alpha in [mp.mpf('1.01'), mp.mpf('1.001')]:
    for t in [mp.mpf('0.03'), mp.mpf('0.1'), mp.mpf('0.5')]:
        print("  alpha=%s t=%s diff=%s" % (ns(alpha, 5), ns(t, 3), ns(abs(Wreal(t, alpha) - W_direct(mp.exp(t), alpha).real), 3)))

print()
print("=== Thm feedback (2): real root z*>1, growth ln z* vs sqrt(alpha-1) ===")
for d in ['1e-1', '1e-2', '1e-3', '1e-4', '1e-6', '1e-8']:
    alpha = 1 + mp.mpf(d)
    t = real_root(alpha, mp.mpf(0), mp.mpf(0))
    print("  alpha-1=%s  ln z* = %s  sqrt(alpha-1) = %s  ratio = %s   (flat-weight s=acosh((1+alpha)/2) = %s)" % (
        d, ns(t, 10), ns(mp.sqrt(mp.mpf(d)), 10), ns(t / mp.sqrt(mp.mpf(d)), 10), ns(mp.acosh((1 + alpha) / 2), 10)))
print("  paper: 0.1010 at alpha=1.01, 0.03167 at alpha=1.001")

print()
print("=== Table feedback, alpha = 1.01: growth per update = ln(real root) ===")
alpha = mp.mpf('1.01')
paper = {'0': ('+0.1010', '+0.0647', '+0.0964'), '0.05': ('+0.0775', '+0.0293', '+0.0715'), '0.2': ('+0.0356', '-0.0456', '+0.0270'),
         '0.5': ('+0.0103', '-0.0874', '+0.0005'), '0.9': ('+0.0012', '-0.0987', '-0.0088')}
mine = {}
for b in ['0', '0.05', '0.2', '0.5', '0.9']:
    row = []
    for lam in ['0', '0.1', '0.01']:
        t = real_root(alpha, mp.mpf(b), mp.mpf(lam))
        row.append(t)
    mine[b] = row
    print("  b=%-5s mine: %s | paper: %s" % (b, ' '.join('%+.6f' % x for x in row), ' '.join(paper[b])))

print()
print("=== criterion thresholds: (1-b)(alpha + (alpha-1) W_lambda(1)) < 1  <=>  b > 1 - 1/(alpha+(alpha-1)W(e^lambda)) ===")
for a in ['1.01', '1.001']:
    for lam in ['0.1', '0.01']:
        al = mp.mpf(a)
        Wl = Wreal(mp.mpf(lam), al)
        print("  alpha=%s lambda=%s  W_lambda(1)=%s  threshold b* = %s" % (a, lam, ns(Wl, 10), ns(1 - 1 / (al + (al - 1) * Wl), 8)))
print("  paper (alpha=1.01): b>0.095 (lambda=0.1), b>0.501 (lambda=0.01)")
print("  flat weights w_j=1 instead (W_lambda(1)=1/(e^lambda-1)), alpha=1.01: thresholds",
      [ns(1 - 1 / (alpha + (alpha - 1) / (mp.exp(mp.mpf(l)) - 1)), 6) for l in ['0.1', '0.01']])
print("  which threshold is real? ln(real root) just above the paper's value and around mine (power-law weights, alpha=1.01):")
for lam, bs in [('0.1', ['0.0955', '0.0965', '0.0969', '0.0970']), ('0.01', ['0.502', '0.510', '0.5126', '0.5127'])]:
    print("    lambda=%s: " % lam + '  '.join("b=%s: %s" % (b, ns(real_root(alpha, mp.mpf(b), mp.mpf(lam)), 4)) for b in bs))
print("  lambda=0: W(e^t) as t->0+ :", [ns(Wreal(mp.mpf(10) ** (-k), alpha), 6) for k in [1, 3, 6, 9]], "(diverges => no brake b<1 settles)")
for b in ['0.9', '0.99', '0.999']:
    print("    lambda=0, b=%s: ln z* = %s" % (b, ns(real_root(alpha, mp.mpf(b), mp.mpf(0)), 8)))

print()
print("=== criterion necessary AND sufficient on a grid: sign(ln z_real) vs criterion; plus root count in |z|>=1 by argument principle ===")
from A_argp import winding
mp.mp.dps = 25
bad = 0; n = 0; nwind = 0; badw = 0
for a in ['1.001', '1.01', '1.1', '1.5']:
    al = mp.mpf(a)
    for lam in ['0.003', '0.01', '0.03', '0.1', '0.3', '1']:
        lm = mp.mpf(lam)
        Wl = Wreal(lm, al)
        for ib in range(0, 20):
            b = mp.mpf(ib) / 20
            crit = (1 - b) * (al + (al - 1) * Wl) < 1
            t = real_root(al, b, lm)
            n += 1
            if crit != (t < 0):
                bad += 1
                print("  MISMATCH", a, lam, b, crit, t)
            if ib % 4 == 0 and a in ('1.01', '1.5') and lam in ('0.01', '0.1', '1'):
                # zeros of h in |z|>1  =  1 - winding of h around the unit circle (h ~ z at infinity, analytic for |z|>e^-lambda)
                f = lambda th: mp.exp(I * th) - (1 - b) * (al + (al - 1) * W_jonq(th - I * lm, al))
                wn = winding(f, [(-mp.pi, mp.pi)])
                outside = 1 - wn
                nwind += 1
                if (outside == 0) != crit:
                    badw += 1
                    print("  WINDING MISMATCH", a, lam, b, crit, outside)
print("  grid points: %d, mismatches between criterion and sign of ln z_real: %d" % (n, bad))
print("  argument-principle checks: %d, mismatches (criterion true <=> zero roots in |z|>1): %d" % (nwind, badw))

print()
print("=== direct iteration (numpy, own code): slope of log|J_n| over second half of 6000 updates, zero past ===")
def iterate(alpha, b, lam, N=6000):
    from math import gamma
    j = np.arange(1, N + 2, dtype=float)
    wts = (j ** alpha - (j - 1) ** alpha) / gamma(alpha + 1) * np.exp(-lam * j)
    J = np.zeros(N + 1); J[0] = 1.0
    for n_ in range(N):
        mem = np.dot(wts[:n_], J[n_ - 1::-1][:n_]) if n_ > 0 else 0.0
        J[n_ + 1] = (1 - b) * (alpha * J[n_] + (alpha - 1) * mem)
    x = np.arange(N // 2, N + 1)
    y = np.log(np.abs(J[N // 2:]))
    return np.polyfit(x, y, 1)[0]
for b in [0.0, 0.05, 0.2, 0.5, 0.9]:
    out = [iterate(1.01, b, lam) for lam in [0.0, 0.1, 0.01]]
    print("  b=%-5s iteration: %s | root: %s | paper: %s" % (
        b, ' '.join('%+.4f' % x for x in out), ' '.join('%+.4f' % x for x in mine[str(b) if b else '0']), ' '.join(paper[str(b) if b else '0'])))
print("  b=0.505, lambda=0.01 (paper criterion says settles, mine says runs away): iteration slope %+.3e over 6000, %+.3e over 40000 updates; root %s" % (
    iterate(1.01, 0.505, 0.01), iterate(1.01, 0.505, 0.01, N=40000), ns(real_root(mp.mpf('1.01'), mp.mpf('0.505'), mp.mpf('0.01')), 5)))
print("  alpha=1.001, b=0, lambda=0: iteration slope %+.5f ; root %s ; paper 0.03167" % (
    iterate(1.001, 0.0, 0.0), ns(real_root(mp.mpf('1.001'), mp.mpf(0), mp.mpf(0)), 6)))
