# Agent A: Thm near2 and Table near2. Independent code.
# Units hbar = c = 1, tau_c = 1/sqrt 2 (mu = 1 at alpha = 2).
import math
import numpy as np
import mpmath as mp
from A_common import check, info, finish, table_check

mp.mp.dps = 50
TAUC = 1 / mp.sqrt(2)


def D(om, alpha, k):
    """Symbol of T = alpha J - (1-alpha) K_alpha J on exp(-i om t + i k x), principal branch
    (analytic continuation of the Laplace integral from Im om > 0 across |Re om| > k)."""
    alpha = mp.mpf(alpha)
    s = -1j * mp.mpmathify(om)
    if k == 0:
        return alpha + (alpha - 1) * (TAUC * s) ** (-alpha)
    b = mp.mpf(k)
    return alpha + TAUC ** (-alpha) * ((s - 1j * b) ** (1 - alpha) - (s + 1j * b) ** (1 - alpha)) / (2j * b)


def root(alpha, k, branch=+1):
    alpha = mp.mpf(alpha)
    k = mp.mpf(k)
    om0 = mp.sqrt(k * k + 1)
    guess = branch * om0 + 1j * mp.pi / 4 * (alpha - 2) / om0
    f = lambda w: D(w, alpha, k)
    r = mp.findroot(f, (guess, guess * (1 + mp.mpf('1e-3')), guess * (1 - mp.mpf('1e-3')) + mp.mpf('1e-4')), solver='muller', tol=mp.mpf(10) ** (-45), maxsteps=200)
    assert abs(f(r)) < mp.mpf(10) ** (-40), abs(f(r))
    return r


# ------------------------------------------------------------ item (1): alpha = 2 roots real
for ks in ['0', '0.1', '1', '5']:
    k = mp.mpf(ks)
    for br in (+1, -1):
        r = root(2, k, br)
        check("Thm near2 (1) k=%s branch %+d" % (ks, br), abs(r - br * mp.sqrt(k * k + 1)) < mp.mpf(10) ** (-35),
              "alpha=2 root %s, Im = %s" % (mp.nstr(mp.re(r), 15), mp.nstr(mp.im(r), 3)))

# ------------------------------------------------------------ exact k = 0 roots for any alpha
for al in ['1.5', '1.9', '2.1', '2.7']:
    a = mp.mpf(al)
    rho = ((a - 1) / a) ** (1 / a) / TAUC
    exact = 1j * rho * mp.exp(-1j * mp.pi / a)  # omega = i s, s = rho exp(-i pi/alpha): Re omega > 0
    r = root(al, 0, +1)
    check("Thm near2 k=0 exact alpha=%s" % al, abs(r - exact) < mp.mpf(10) ** (-35),
          "root %s vs i*rho*exp(-i pi/alpha); Im omega = rho cos(pi/alpha) = %s" % (mp.nstr(r, 12), mp.nstr(rho * mp.cos(mp.pi / a), 12)))

# ------------------------------------------------------------ Table near2: roots and rates at k = 1
tab = {'1.90': ('1.370022', '-5.435e-2', '5.03e-2'), '1.99': ('1.410296', '-5.546e-3', '5.51e-3'),
       '2.01': ('1.418025', '5.560e-3', '5.60e-3'), '2.05': ('1.432250', '2.790e-2', '2.88e-2'),
       '2.10': ('1.447853', '5.590e-2', '5.95e-2')}
for al, (pre, pim, pform) in tab.items():
    rp = root(al, 1, +1)
    rm = root(al, 1, -1)
    table_check("Table near2 alpha=%s root (+)" % al, pre, mp.re(rp))
    table_check("Table near2 alpha=%s Im omega" % al, pim, mp.im(rp))
    check("Table near2 alpha=%s both branches" % al, abs(rm + mp.conj(rp)) < mp.mpf(10) ** (-35), "minus-branch root = -conj(plus-branch root): same Im omega")
    d = abs(mp.mpf(al) - 2)
    fA = mp.pi / 4 * d * (mp.re(rp) ** 2 - 1) / mp.re(rp)          # mu^2 = (Re omega)^2 - k^2, |omega| -> Re omega
    fC = mp.pi / 4 * d * abs(rp * rp - 1) / abs(rp)                # complex moduli
    fB = mp.pi / 4 * d / mp.sqrt(2)                                # unperturbed mu = 1, omega = sqrt 2
    okA = table_check("Table near2 alpha=%s formula column" % al, pform, fA)
    info("Table near2 alpha=%s formula variants" % al, "with (Re omega)^2-k^2 and Re omega: %s; with moduli: %s; with alpha=2 values mu=1, omega=sqrt2: %s; true |Im omega| %s"
         % (mp.nstr(fA, 5), mp.nstr(fC, 5), mp.nstr(fB, 5), mp.nstr(abs(mp.im(rp)), 5)))

# ------------------------------------------------------------ rate versus delta, 1e-2 ... 1e-8
print("INFO Thm near2 rate sweep: ratio = Im(omega) / [(pi/4) delta mu0^2/omega0], mu0 = 1, omega0 = sqrt(k^2+1); c1 = (ratio-1)/delta")
c1 = {}
allok = True
for ks in ['0', '0.1', '1', '5']:
    k = mp.mpf(ks)
    om0 = mp.sqrt(k * k + 1)
    for sgn in (+1, -1):
        prev = None
        for j in range(2, 9):
            delta = sgn * mp.mpf(10) ** (-j)
            r = root(2 + delta, k, +1)
            lead = mp.pi / 4 * delta / om0
            ratio = mp.im(r) / lead
            cc = (ratio - 1) / delta
            ratioA = mp.im(r) / (mp.pi / 4 * delta * (mp.re(r) ** 2 - k * k) / mp.re(r))
            ccA = (ratioA - 1) / delta
            print("INFO   k=%-3s delta=%+.0e  Re omega-omega0=%s  Im omega=%s  ratio-1=%s  c1=%s  | paper's convention (root's own mu, omega): ratio-1=%s c1=%s"
                  % (ks, float(delta), mp.nstr(mp.re(r) - om0, 6), mp.nstr(mp.im(r), 10), mp.nstr(ratio - 1, 6), mp.nstr(cc, 8), mp.nstr(ratioA - 1, 6), mp.nstr(ccA, 8)))
            # sign of Im omega = sign of delta, and |ratio-1| <= 2|delta|
            if not (mp.sign(mp.im(r)) == sgn and abs(ratio - 1) <= 2 * abs(delta) and abs(ratioA - 1) <= 2 * abs(delta)):
                allok = False
            prev = (cc, ccA)
        c1[(ks, sgn)] = prev
check("Thm near2 (2) first-order rate", allok, "Im omega = (pi/4) delta mu^2/|omega| (1+O(delta)) for delta = +-1e-2..+-1e-8, k = 0, 0.1, 1, 5; decay below 2, growth above; |ratio-1| <= 2|delta| throughout")
for ks in ['0', '0.1', '1', '5']:
    a, b = c1[(ks, +1)], c1[(ks, -1)]
    check("Thm near2 next correction k=%s" % ks, abs(a[0] - b[0]) < mp.mpf('1e-6') and abs(a[1] - b[1]) < mp.mpf('1e-6'),
          "relative correction is c1*delta: c1 = %s (alpha=2 values mu=1, omega0), c1 = %s (root's own mu^2, omega as in the paper's table); same from both sides" % (mp.nstr(a[0], 8), mp.nstr(a[1], 8)))
k0 = (mp.log(2) - 1) / 4
check("Thm near2 next correction k=0 closed form", abs(c1[('0', 1)][0] - k0) < mp.mpf('1e-6'), "c1(k=0) = (ln 2 - 1)/4 = %s" % mp.nstr(k0, 10))

# ------------------------------------------------------------ growing modes: argument principle (numpy, double)
SQ2 = math.sqrt(2.0)


def D_np(om, alpha, k):
    s = -1j * om
    if k == 0:
        return alpha + (alpha - 1) * (s / SQ2) ** (-alpha)
    return alpha + SQ2 ** alpha * ((s - 1j * k) ** (1 - alpha) - (s + 1j * k) ** (1 - alpha)) / (2j * k)


def winding(alpha, k, y0, Y=12.0, X=40.0):
    corners = [complex(-X, y0), complex(X, y0), complex(X, Y), complex(-X, Y), complex(-X, y0)]
    total = 0.0
    far = 0.0
    for i in range(4):
        a, b = corners[i], corners[i + 1]
        t = np.linspace(0.0, 1.0, 40001)
        if i == 0:
            # cluster points around the branch points omega = +-k (and 0), which sit y0 below this edge,
            # so that a phase swing of a multiple of 2 pi inside one coarse interval cannot be aliased away
            off = y0 * 1.3 ** np.arange(0, 70)
            xs = np.concatenate([c0 + sg * off for c0 in (-k, k) for sg in (-1, 1)] + [np.array([-k, k])])
            xs = xs[(xs > -X) & (xs < X)]
            t = np.unique(np.concatenate([t, (xs + X) / (2 * X)]))
        for it in range(80):
            z = a + (b - a) * t
            v = D_np(z, alpha, k)
            dphi = np.angle(v[1:] / v[:-1])
            bad = np.abs(dphi) > 0.2
            if not bad.any():
                break
            mids = 0.5 * (t[1:] + t[:-1])[bad]
            t = np.sort(np.concatenate([t, mids]))
        else:
            raise RuntimeError("no convergence")
        total += dphi.sum()
        if i > 0:
            far = max(far, float(np.max(np.abs(v - alpha))))
    return total / (2 * math.pi), far


paper_counts = {1.90: (0, 0, 0), 1.99: (0, 0, 0), 2.01: (2, 2, 0), 2.05: (2, 2, 2), 2.10: (2, 2, 2)}
for al, pc in paper_counts.items():
    mine = []
    for k in (0.1, 1.0, 5.0):
        w, far = winding(al, k, 0.002, 12.0)
        mine.append(int(round(w)))
        assert abs(w - round(w)) < 1e-6
    check("Table near2 alpha=%.2f growing modes (paper's rectangle Im in [0.002,12])" % al, tuple(mine) == pc, "printed %s, mine %s" % (pc, tuple(mine)))
true_counts = {}
for al in (1.90, 1.99, 2.01, 2.05, 2.10):
    mine = []
    for k in (0.0, 0.1, 1.0, 5.0):
        w, far = winding(al, k, 1e-6, 40.0)
        assert abs(w - round(w)) < 1e-6
        mine.append(int(round(w)))
    true_counts[al] = tuple(mine)
    exp = (2, 2, 2, 2) if al > 2 else (0, 0, 0, 0)
    check("Thm near2 growing-mode count alpha=%.2f (lower edge 1e-6)" % al, tuple(mine) == exp, "zeros in Im omega > 1e-6, |Re|<40, Im<40 for k=0,0.1,1,5: %s" % (tuple(mine),))
r5 = root('2.01', 5, +1)
table_check("Table near2 caption alpha=2.01 k=5 growth rate", "1.5e-3", mp.im(r5))
info("Table near2 caption alpha=2.01 k=5", "exact Im omega = %s rounds to 1.6e-3; the printed 1.5e-3 is the first-order value (pi/4)(0.01)/sqrt(26) = %s. Either way it is below the 0.002 edge, so the printed count 0 is a threshold artifact; the true count is 2." % (mp.nstr(mp.im(r5), 8), mp.nstr(mp.pi / 4 * mp.mpf('0.01') / mp.sqrt(26), 8)))

# exploration away from 2 (not a claim of the paper)
for al in (0.5, 1.2, 1.5, 1.8, 2.2, 2.5, 3.0, 4.0, 5.0, 7.0):
    row = []
    for k in (0.0, 0.1, 1.0, 5.0):
        w, far = winding(al, k, 1e-6, 60.0, 60.0)
        row.append(int(round(w)))
    info("extra roots alpha=%.1f" % al, "number of growing modes (Im omega > 0) at k=0,0.1,1,5: %s" % (tuple(row),))
a = mp.mpf('0.5')
sreal = ((1 - a) / a) ** (1 / a) / TAUC
info("extra roots alpha<1", "for 0<alpha<1 the same law has a real growing mode at k=0, s = ((1-alpha)/alpha)^(1/alpha)/tau_c (= %s at alpha=0.5; D there = %s). Outside the theorem's range 'alpha near 2'." % (mp.nstr(sreal, 8), mp.nstr(abs(D(1j * sreal, a, 0)), 3)))

# ------------------------------------------------------------ time-stepped runs (numpy double, own solver)
def source(t):
    return np.where((t > 0) & (t < 2), np.sin(np.pi * t / 2) ** 2, 0.0)


def volterra(alpha, k, h, T):
    n = int(round(T / h))
    t = h * np.arange(n + 1)
    tau = t.copy()
    ker = np.zeros(n + 1)
    ker[1:] = tau[1:] ** (alpha - 1) * np.sinc(k * tau[1:] / np.pi) * SQ2 ** alpha / math.gamma(alpha)
    S = source(t)
    J = np.zeros(n + 1)
    J[0] = S[0] / alpha
    for i in range(1, n + 1):
        integ = h * (np.dot(ker[1:i], J[i - 1:0:-1]) + 0.5 * ker[i] * J[0])
        J[i] = (S[i] - (alpha - 1) * integ) / alpha
    return t, J


def peaks(t, J, t0, t1):
    a = np.abs(J)
    idx = np.where((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]) & (t[1:-1] >= t0) & (t[1:-1] <= t1))[0] + 1
    return t[idx], a[idx]


h = 0.02
# alpha = 2: compare with Thm KG (1) solution J = S/2 - (mu^2/2) int sin(w(t-t'))/w S(t') dt'
t, J = volterra(2.0, 1.0, h, 60.0)
w = SQ2
Sx = source(t)
ex = np.zeros_like(t)
cs = np.cumsum
# exact via trapezoid of smooth integrand on the same grid (source is C^1), error O(h^2)
cosw, sinw = np.cos(w * t), np.sin(w * t)
A = np.concatenate([[0], np.cumsum(0.5 * h * ((Sx * cosw)[1:] + (Sx * cosw)[:-1]))])
B = np.concatenate([[0], np.cumsum(0.5 * h * ((Sx * sinw)[1:] + (Sx * sinw)[:-1]))])
ex = 0.5 * Sx - 0.5 * (sinw * A - cosw * B) / w
err2 = float(np.max(np.abs(J - ex)))
check("Thm KG (1) time-stepped alpha=2", err2 < 2e-3, "own Volterra stepping (h=0.02, k=1) vs S/2 - (mu^2/2) G_mu*S: max diff %.2e on [0,60]; max|J| after source %.4f (bounded)" % (err2, float(np.max(np.abs(J[t > 30])))))

for al, T in ((1.95, 220.0), (2.05, 220.0)):
    t, J = volterra(al, 1.0, h, T)
    tp, ap = peaks(t, J, 50.0, 200.0)
    slope = np.polyfit(tp, np.log(ap), 1)[0]
    r = root(al, 1, +1)
    check("Thm near2 time-stepped alpha=%.2f" % al, abs(slope - float(mp.im(r))) < 0.03 * abs(float(mp.im(r))),
          "fitted envelope rate on t in [50,200]: %.5f; root Im omega: %.5f; first-order formula (alpha=2 values): %.5f" % (slope, float(mp.im(r)), math.pi / 4 * (al - 2) / SQ2))
    i50, i200 = int(round(50 / h)), int(round(200 / h))
    info("Table near2 caption alpha=%.2f" % al, "my run (k=1, source sin^2(pi t/2) on [0,2]): envelope %.3e at t=50, %.3e at t=200. Paper prints %s; its source, k and the meaning of |J| are not stated, so the two amplitudes cannot be reproduced from the text. Rate implied by the printed pair: %.5f"
         % (float(np.interp(50, tp, ap)) if len(tp) else float('nan'), float(np.interp(200, tp, ap)) if len(tp) else float('nan'),
            "8.0e-2 -> 1.4e-3" if al < 2 else "6.3e-1 -> 5.9e1", (math.log(1.4e-3 / 8.0e-2) if al < 2 else math.log(5.9e1 / 6.3e-1)) / 150.0))

# long-time behaviour below 2: branch-point tail
al = 1.9
kk = 1.0
T = 700.0
t, J = volterra(al, kk, h, T)
r = root(al, 1, +1)
mp.mp.dps = 30
Stil = mp.quad(lambda x: mp.sin(mp.pi * x / 2) ** 2 * mp.exp(1j * kk * x), [0, 1, 2])  # S~(s) at s = -i k
coef = complex(Stil * (-2j * kk) * mp.mpf(2) ** (-mp.mpf(al) / 2) / mp.gamma(1 - mp.mpf(al)))
mask = (t >= 450) & (t <= 700)
pred = 2 * np.real(coef * np.exp(-1j * kk * t[mask])) * t[mask] ** (-al)
num = J[mask]
rel = float(np.linalg.norm(num - pred) / np.linalg.norm(pred))
expo = float(np.exp(float(mp.im(r)) * 600.0))
tp, ap = peaks(t, J, 300.0, 700.0)
pl = np.polyfit(np.log(tp), np.log(ap), 1)[0]
check("Thm near2 long-time tail alpha=1.9 (not in paper)", rel < 0.05,
      "for t in [450,700], J(t) matches the branch-point tail 2 Re[C e^{-ikt}] t^{-alpha}, C = S~(-ik)(-2ik) tau_c^alpha... /Gamma(1-alpha), to rel L2 %.3f; fitted power-law exponent of the envelope on [300,700]: %.3f (alpha = %.1f); |J| near t=600 is %.2e while exp(Im omega * 600) = %.1e"
      % (rel, pl, al, float(np.max(np.abs(J[(t > 590) & (t < 610)]))), expo))
tp, ap = peaks(t, J, 20.0, 700.0)
for tt in (50, 100, 150, 200, 250, 300, 400, 600):
    info("tail alpha=1.9 envelope", "t=%4d  |J|env=%.3e   exp-only estimate %.3e" % (tt, float(np.interp(tt, tp, ap)), float(np.interp(50, tp, ap)) * math.exp(float(mp.im(r)) * (tt - 50))))

finish("A_near2")
