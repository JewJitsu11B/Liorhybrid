# Agent A: Thm pathsum, Prop checker, Prop descent3, Tables pathsum / checker / descent.
# Independent code (mpmath for the mathematics, numpy/scipy only to rebuild the paper's grid tables).
import itertools
import math
import numpy as np
from scipy import special as sp
import mpmath as mp
from A_common import check, info, finish, table_check

mp.mp.dps = 45
TOL = mp.mpf(10) ** (-30)
GL = dict(method='gauss-legendre')


# ------------------------------------------------------------------ claimed closed forms
def G1(n, t, x, c):
    """Prop checker (1): G^{*(n+1)} in 1+1."""
    u, v = c * t - x, c * t + x
    if u < 0 or v < 0:
        return mp.mpf(0)
    return (u * v / (4 * c * c)) ** n / mp.factorial(n) ** 2 / (2 * c)


def G3(n, t, r, c):
    """Prop descent3: G^{*(n+1)} in 3+1 for n >= 1 (regular part)."""
    w = c * c * t * t - r * r
    if w < 0:
        return mp.mpf(0)
    return n * w ** (n - 1) / (2 * mp.pi * c * (4 * c * c) ** n * mp.factorial(n) ** 2)


# ------------------------------------------------------------------ 1+1: terms by direct convolution in (t, x)
def conv1(n, t, x, c):
    def inner(tp):
        lo = max(-c * tp, x - c * (t - tp))
        hi = min(c * tp, x + c * (t - tp))
        if hi <= lo:
            return mp.mpf(0)
        return mp.quad(lambda xp: G1(n, tp, xp, c) / (2 * c), [lo, hi], **GL)
    brk = sorted(set([mp.mpf(0), (c * t - x) / (2 * c), (c * t + x) / (2 * c), t]))
    return mp.quad(inner, brk, **GL)


c = mp.mpf('1.7')
pts = [(mp.mpf('2.0'), mp.mpf('0.9')), (mp.mpf('3.1'), mp.mpf('-4.4')), (mp.mpf('0.7'), mp.mpf('0'))]
for n in range(0, 5):
    worst = mp.mpf(0)
    for (t, x) in pts:
        cv = conv1(n, t, x, c)
        ex = G1(n + 1, t, x, c)
        worst = max(worst, abs(cv - ex) / abs(ex))
    check("Prop checker (1) term n=%d" % (n + 1), worst < TOL, "G * G^{*(%d)} by 2D quadrature in (t,x), c=1.7, vs theta theta/(2c) (uv/4c^2)^n/(n!)^2: max rel diff %s" % (n + 1, mp.nstr(worst, 3)))

# mass of each term (saturates the bound of Thm pathsum (1) for S = theta(t))
worst = mp.mpf(0)
for n in range(0, 7):
    t = mp.mpf('2.3')
    m1 = mp.quad(lambda x: G1(n, t, x, c), [-c * t, 0, c * t])
    worst = max(worst, abs(m1 - t ** (2 * n + 1) / mp.factorial(2 * n + 1)) / m1)
    if n >= 1:
        m3 = mp.quad(lambda r: 4 * mp.pi * r * r * G3(n, t, r, c), [0, c * t])
        worst = max(worst, abs(m3 - t ** (2 * n + 1) / mp.factorial(2 * n + 1)) / m3)
check("Thm pathsum (1) bound", worst < TOL, "total mass of G^{*(n+1)}(t,.) = t^(2n+1)/(2n+1)! in d=1 and d=3, n<=6, hence sup|G^{*n}*theta| = t^(2n)/(2n)! (bound attained): max rel diff %s" % mp.nstr(worst, 3))

# ------------------------------------------------------------------ Neumann series vs Bessel closed forms
mp.mp.dps = 90
worst1 = worstI = worst3 = worstI3 = mp.mpf(0)
nterms_needed = {}
for (cc, mu, t, x) in [(1, 1, 3, 1), ('1.7', '0.8', 5, '-2.5'), (1, 1, 12, 0), ('0.3', '2.5', 20, '1.1'), (1, 1, 60, 5)]:
    cc, mu, t, x = mp.mpf(cc), mp.mpf(mu), mp.mpf(t), mp.mpf(x)
    z = mu * mp.sqrt(t * t - x * x / (cc * cc))
    ssum = sabs = mp.mpf(0)
    s3 = s3abs = mp.mpf(0)
    n = 0
    while True:
        term = (-mu * mu) ** n * G1(n, t, x, cc)
        ssum += term
        sabs += abs(term)
        if n >= 1:
            t3 = (-mu * mu) ** n * G3(n, t, abs(x), cc)
            s3 += t3
            s3abs += abs(t3)
        n += 1
        if abs(term) < mp.mpf(10) ** (-60) and n > z:
            break
    nterms_needed[mp.nstr(z, 5)] = n
    ex = mp.besselj(0, z) / (2 * cc)
    worst1 = max(worst1, abs(ssum - ex))
    worstI = max(worstI, abs(sabs - mp.besseli(0, z) / (2 * cc)) / sabs)
    w = cc * cc * t * t - x * x
    ex3 = -mu / (4 * mp.pi * cc * cc) * mp.besselj(1, mu * mp.sqrt(w) / cc) / mp.sqrt(w)
    worst3 = max(worst3, abs(s3 - ex3))
    worstI3 = max(worstI3, abs(s3abs - mu / (4 * mp.pi * cc * cc) * mp.besseli(1, mu * mp.sqrt(w) / cc) / mp.sqrt(w)) / s3abs)
check("Prop checker (1) resummation 1+1", worst1 < mp.mpf(10) ** (-40), "sum_n (-mu^2)^n G^{*(n+1)} vs theta theta J0(mu sqrt(t^2-x^2/c^2))/(2c), 5 points up to z=59.8: max abs diff %s" % mp.nstr(worst1, 3))
check("Prop checker (1) absolute sum", worstI < mp.mpf(10) ** (-40), "sum of absolute values vs I0/(2c): max rel diff %s" % mp.nstr(worstI, 3))
check("Prop descent3 resummation 3+1 tail", worst3 < mp.mpf(10) ** (-40), "sum_{n>=1} (-mu^2)^n G^{*(n+1)} vs -(mu/4 pi c^2) J1(mu sqrt w/c)/sqrt w: max abs diff %s" % mp.nstr(worst3, 3))
check("Prop descent3 absolute sum", worstI3 < mp.mpf(10) ** (-40), "sum of absolute values vs (mu/4 pi c^2) I1/sqrt w: max rel diff %s" % mp.nstr(worstI3, 3))
info("Thm pathsum convergence", "terms needed for 1e-60 absolute accuracy as a function of z = mu sqrt(t^2-x^2/c^2): %s; the alternating series loses about z/2.3 digits to cancellation (terms reach I0(z))" % nterms_needed)
mp.mp.dps = 45

# ------------------------------------------------------------------ symbols: spatial Fourier transform of G_mu(t,.) = sin(omega t)/omega
worst1 = worst3 = mp.mpf(0)
for (cc, mu, t, k) in [(1, 1, 3, '0.7'), ('1.7', '0.8', 2, '2.2'), ('0.6', '2.5', 4, '1.3')]:
    cc, mu, t, k = mp.mpf(cc), mp.mpf(mu), mp.mpf(t), mp.mpf(k)
    om = mp.sqrt(cc * cc * k * k + mu * mu)
    ex = mp.sin(om * t) / om
    nsub = 8
    grid = [mp.pi / 2 * j / nsub for j in range(nsub + 1)]
    f1 = 2 * mp.quad(lambda th: (t / 2) * mp.cos(th) * mp.besselj(0, mu * t * mp.cos(th)) * mp.cos(k * cc * t * mp.sin(th)), grid, **GL)
    tail = -(mu / (cc * cc * k)) * mp.quad(lambda th: cc * t * mp.sin(th) * mp.sin(k * cc * t * mp.sin(th)) * mp.besselj(1, mu * t * mp.cos(th)), grid, **GL)
    f3 = mp.sin(cc * k * t) / (cc * k) + tail
    worst1 = max(worst1, abs(f1 - ex))
    worst3 = max(worst3, abs(f3 - ex))
check("Thm pathsum (3) symbol 1+1", worst1 < TOL, "spatial FT of theta J0/(2c) by quadrature vs sin(omega t)/omega, omega^2 = c^2k^2+mu^2: max diff %s" % mp.nstr(worst1, 3))
check("Thm pathsum (3) / Prop descent3 symbol 3+1", worst3 < TOL, "spatial FT of [shell - (mu/4 pi c^2) J1/sqrt w] by quadrature vs sin(omega t)/omega (sign and normalization of shell and tail): max diff %s" % mp.nstr(worst3, 3))

# ------------------------------------------------------------------ 3+1 terms by direct convolution with the shell
cc, t = mp.mpf('1.3'), mp.mpf('2.0')
lhs = mp.quad(lambda tau: tau * (t - tau) * mp.quad(lambda zt: mp.exp(-((cc * tau) ** 2 + (cc * (t - tau)) ** 2 + 2 * cc * cc * tau * (t - tau) * zt)) / 2, [-1, 1], **GL), [0, t / 2, t], **GL)
rhs = mp.quad(lambda r: mp.exp(-r * r) * 4 * mp.pi * r * r / (8 * mp.pi * cc ** 3), [0, cc * t], **GL)
check("Prop descent3 term n=1 (two null legs)", abs(lhs - rhs) < TOL, "G*G in 3+1 paired with exp(-|x|^2), from two shells with weights tau (Thm pathsum (4), n=1), vs theta(ct-r)/(8 pi c^3): diff %s" % mp.nstr(abs(lhs - rhs), 3))


def conv3(n, t, r, c):
    def inner(tau):
        if tau == 0:
            return mp.mpf(0)
        zmin = (r * r - c * c * t * t + 2 * c * c * t * tau) / (2 * r * c * tau)
        zmin = max(zmin, mp.mpf(-1))
        if zmin >= 1:
            return mp.mpf(0)
        return tau * mp.quad(lambda zt: G3(n, t - tau, mp.sqrt(r * r + c * c * tau * tau - 2 * r * c * tau * zt), c) / 2, [zmin, 1], **GL)
    brk = [mp.mpf(0), (c * t - r) / (2 * c), (c * t + r) / (2 * c)]
    return mp.quad(inner, brk, **GL)


for n in (1, 2, 3):
    worst = mp.mpf(0)
    for (t, r) in [(mp.mpf('2.0'), mp.mpf('0.9')), (mp.mpf('3.0'), mp.mpf('3.5'))]:
        cv = conv3(n, t, r, cc)
        ex = G3(n + 1, t, r, cc)
        worst = max(worst, abs(cv - ex) / ex)
    check("Prop descent3 term n=%d" % (n + 1), worst < TOL, "shell-convolution of G^{*(%d)} by quadrature vs n w^(n-1)/(2 pi c (4c^2)^n (n!)^2): max rel diff %s" % (n + 1, mp.nstr(worst, 3)))

# ------------------------------------------------------------------ checkerboard: own corner counting
ok = True
for N in range(2, 15):
    counts = {}
    for steps in itertools.product((1, -1), repeat=N - 2):
        path = (1,) + steps + (-1,)
        corners = sum(1 for a, b in zip(path, path[1:]) if a != b)
        NR = path.count(1)
        counts[(NR, corners)] = counts.get((NR, corners), 0) + 1
    for (NR, corners), cnt in counts.items():
        NL = N - NR
        if corners % 2 == 0:
            ok = False
            continue
        n = (corners - 1) // 2
        if cnt != math.comb(NR - 1, n) * math.comb(NL - 1, n):
            ok = False
    # and nothing missing
    for NR in range(1, N):
        NL = N - NR
        for n in range(0, min(NR, NL)):
            if counts.get((NR, 2 * n + 1), 0) != math.comb(NR - 1, n) * math.comb(NL - 1, n):
                ok = False
check("Prop checker (3) corner count, brute force", ok, "all 2^(N-2) paths, N=2..14, first step right, last step left: number with N_R right steps and 2n+1 corners is C(N_R-1,n) C(N_L-1,n); no even-corner paths")

# transfer recursion with exact integer corner polynomials, larger lattice
N = 60
state = {(1, 1): {0: 1}}
for step in range(2, N + 1):
    new = {}
    for (pos, d), poly in state.items():
        for d2 in (1, -1):
            key = (pos + d2, d2)
            tgt = new.setdefault(key, {})
            for cnr, cnt in poly.items():
                c2 = cnr + (1 if d2 != d else 0)
                tgt[c2] = tgt.get(c2, 0) + cnt
    state = new
ok = True
for (pos, d), poly in state.items():
    if d != -1:
        continue
    NR, NL = (N + pos) // 2, (N - pos) // 2
    for cnr, cnt in poly.items():
        if cnr % 2 == 0 or cnt != math.comb(NR - 1, (cnr - 1) // 2) * math.comb(NL - 1, (cnr - 1) // 2):
            ok = False
check("Prop checker (3) corner count, exact transfer recursion N=60", ok, "integer corner polynomials at every site agree with C(N_R-1,n) C(N_L-1,n)")

# term by term continuum limit, with the factor 1/(2 i c mu); c = 1 on the lattice
mu, t, x = mp.mpf('1.3'), mp.mpf(2), mp.mpf('0.5')
u, v = t - x, t + x
print("INFO Prop checker (2),(3) term by term: rel diff between lattice term (i eps mu)^(2n+1) C C / eps / (2 i mu) and (-mu^2)^n G^{*(n+1)} at (t,x)=(2,0.5), mu=1.3")
okterm = True
prev = None
for N in (2 ** 7, 2 ** 10, 2 ** 13, 2 ** 16):
    eps = t / N
    NR, NL = int(N * 5 // 8), int(N * 3 // 8)
    row = []
    for n in range(0, 7):
        lat = (1j * eps * mu) ** (2 * n + 1) * mp.binomial(NR - 1, n) * mp.binomial(NL - 1, n) / eps / (2j * mu)
        cont = (-mu * mu) ** n * G1(n, t, x, mp.mpf(1))
        row.append((lat - cont) / cont)
        if mp.im(lat) != 0 and abs(mp.im(lat)) > mp.mpf(10) ** (-40):
            okterm = False
    print("INFO   N=%6d  " % N + " ".join("n=%d:%s" % (n, mp.nstr(mp.re(row[n]), 4)) for n in range(7)))
    if prev is not None and N >= 2 ** 13:
        for n in range(1, 7):
            ratio = prev[n] / row[n]
            if not (7.0 < mp.re(ratio) < 9.5):
                okterm = False
    prev = row
# predicted first-order coefficient: -n(n+1) eps (1/v + 1/u)
pred = [-n * (n + 1) * eps * (1 / v + 1 / u) for n in range(7)]
okpred = all(abs(mp.re(prev[n]) - pred[n]) < abs(pred[n]) * mp.mpf('1e-2') for n in range(1, 7)) and abs(prev[0]) < TOL
check("Prop checker (2) term by term", okterm and okpred, "each (2n+1)-corner lattice term / (2 i c mu) -> (-mu^2)^n G^{*(n+1)}, n=0..6; n=0 exact; error first order in eps with coefficient -n(n+1)(1/u+1/v) eps (my derivation), ratio 8 per 8-fold refinement")
okid = True
for cgen in ('0.4', '2.5'):
    cg = mp.mpf(cgen)
    for n in range(0, 6):
        ug, vg = cg * t - x, cg * t + x
        Phi = (1j * mu) ** (2 * n + 1) * (ug * vg / (4 * cg * cg)) ** n / mp.factorial(n) ** 2
        if abs(Phi / (2j * cg * mu) - (-mu * mu) ** n * G1(n, t, x, cg)) > TOL:
            okid = False
check("Prop checker (2) factor 1/(2 i c mu)", okid, "Phi_(2n+1)/(2 i c mu) = (-mu^2)^n G^{*(n+1)} with the paper's Phi, c = 0.4 and 2.5, n <= 5")

print("INFO Prop checker (3) full lattice sum at (t,x)=(2,0.5), mu=1.3: err = L_eps/(i eps mu) - J0(mu sqrt(t^2-x^2))")
errs = []
for N in (2 ** 6, 2 ** 8, 2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18):
    eps = t / N
    NR, NL = int(N * 5 // 8), int(N * 3 // 8)
    tot = mp.mpc(0)
    for n in range(0, min(NR, NL)):
        term = (1j * eps * mu) ** (2 * n + 1) * mp.binomial(NR - 1, n) * mp.binomial(NL - 1, n)
        tot += term
        if abs(term) < mp.mpf(10) ** (-50) * eps:
            break
    err = tot / (1j * eps * mu) - mp.besselj(0, mu * mp.sqrt(t * t - x * x))
    errs.append((N, err))
    print("INFO   N=%7d eps=%s err=%s  err/eps=%s" % (N, mp.nstr(eps, 4), mp.nstr(mp.re(err), 8), mp.nstr(mp.re(err) / eps, 10)))
lim = mp.re(errs[-1][1]) / (t / errs[-1][0])
check("Prop checker (3) lattice limit", abs(errs[-1][1]) < mp.mpf('1e-4') and abs(mp.re(errs[-2][1]) / mp.re(errs[-1][1]) - 4) < mp.mpf('0.01') and abs(mp.im(errs[-1][1])) < TOL,
      "L_eps/(i eps mu) -> J0, i.e. L_eps/eps -> 2 i c mu G_mu; error O(eps), err/eps -> %s" % mp.nstr(lim, 8))

# ------------------------------------------------------------------ Table checker (own lattice propagation, numpy complex128)
def lattice(eps, t=8.0, mu=1.0):
    N = int(round(t / eps))
    R = np.zeros(2 * N + 3, dtype=complex)
    L = np.zeros(2 * N + 3, dtype=complex)
    R[N + 2] = 1.0  # after the first step, at x = +eps (index N+1+pos)
    a = 1j * eps * mu
    for _ in range(N - 1):
        nR = np.zeros_like(R)
        nL = np.zeros_like(L)
        nR[1:] = R[:-1] + a * L[:-1]
        nL[:-1] = L[1:] + a * R[1:]
        R, L = nR, nL
    pos = np.arange(-N - 1, N + 2)
    xs = pos * eps
    sel = (np.abs(xs) <= 7.5 + 1e-12) & ((pos - N) % 2 == 0)
    val = L[sel] / a
    ex = sp.j0(mu * np.sqrt(t * t - xs[sel] ** 2))
    err = np.abs(val - ex)
    i = int(np.argmax(err))
    return float(err[i]), float(xs[sel][i]), N


tabc = {0.02: ('3.697e-2', None), 0.01: ('1.810e-2', '2.04'), 0.005: ('8.950e-3', '2.02'), 0.0025: ('4.451e-3', '2.01')}
prev = None
for eps, (pe, pr) in tabc.items():
    e, xm, N = lattice(eps)
    table_check("Table checker eps=%g max error" % eps, pe, e)
    # confirm the same site with the binomial sum at 45 digits
    NR, NL = int(round((8 + xm) / (2 * eps))), int(round((8 - xm) / (2 * eps)))
    me = mp.mpf(eps)
    tot = mp.mpc(0)
    for n in range(0, min(NR, NL)):
        term = (1j * me) ** (2 * n + 1) * mp.binomial(NR - 1, n) * mp.binomial(NL - 1, n)
        tot += term
        if abs(term) < mp.mpf(10) ** (-45):
            break
    emp = abs(tot / (1j * me) - mp.besselj(0, mp.sqrt(64 - mp.mpf(xm) ** 2)))
    check("Table checker eps=%g mp cross-check" % eps, abs(emp - e) < 1e-9, "binomial sum at the worst site x=%.4f in 45-digit arithmetic: %s (lattice run %.6e)" % (xm, mp.nstr(emp, 8), e))
    if prev is not None:
        table_check("Table checker eps=%g ratio" % eps, pr, prev / e)
    prev = e

# ------------------------------------------------------------------ Table pathsum (paper's grid construction, rebuilt)
def one_d_iterates(h, nmax, umax=12.0, use_mp=False):
    npts = int(round(umax / float(h))) + 1
    if use_mp:
        hh = mp.mpf(h)
        a = [mp.mpf(1)] * npts
        out = [a]
        for _ in range(nmax):
            b = [mp.mpf(0)] * npts
            for i in range(1, npts):
                b[i] = b[i - 1] + hh * (a[i] + a[i - 1]) / 2
            out.append(b)
            a = b
        return out
    a = np.ones(npts)
    out = [a]
    for _ in range(nmax):
        b = np.concatenate([[0.0], np.cumsum(0.5 * h * (a[1:] + a[:-1]))])
        out.append(b)
        a = b
    return out


tabp = {'0.020': ['1.44e2', '1.15e3', '3.28e1', '3.35e-5', '3.35e-5', '3.35e-5'],
        '0.010': ['1.44e2', '1.15e3', '3.28e1', '8.38e-6', '8.38e-6', '8.38e-6']}
ns = [2, 5, 10, 20, 40, 60]
final = {}
grids = {}
for hs, printed in tabp.items():
    h = float(hs)
    its = one_d_iterates(h, 60)
    uu = h * np.arange(len(its[0]))
    exact = 0.5 * sp.j0(np.sqrt(np.outer(uu, uu)))
    psum = np.zeros_like(exact)
    res = {}
    for n in range(0, 61):
        psum += 0.5 * (-0.25) ** n * np.outer(its[n], its[n])
        if n in ns:
            e = np.abs(psum - exact)
            idx = np.unravel_index(np.argmax(e), e.shape)
            res[n] = (float(e[idx]), idx)
    grids[hs] = psum.copy()
    final[hs] = res[60][0]
    # 30-digit confirmation of the converged residual at its argmax
    mp.mp.dps = 40
    itm = one_d_iterates(hs, 60, use_mp=True)
    i, j = res[60][1]
    val = sum(mp.mpf('0.5') * (mp.mpf('-0.25')) ** n * itm[n][i] * itm[n][j] for n in range(61))
    ui, vj = mp.mpf(hs) * i, mp.mpf(hs) * j
    emp = abs(val - mp.besselj(0, mp.sqrt(ui * vj)) / 2)
    mp.mp.dps = 45
    for n, pr in zip(ns, printed):
        table_check("Table pathsum h=%s n=%d" % (hs, n), pr, res[n][0])
    check("Table pathsum h=%s mp cross-check" % hs, abs(emp - res[60][0]) < 1e-9, "n=60 residual at its worst grid point (u,v)=(%s,%s) recomputed with 40-digit trapezoid iterates: %s" % (mp.nstr(ui, 5), mp.nstr(vj, 5), mp.nstr(emp, 8)))
check("Table pathsum caption ratio 4.0", abs(final['0.020'] / final['0.010'] - 4.0) < 0.05, "residual ratio when h halves: %.4f (printed 4.0)" % (final['0.020'] / final['0.010']))

# ------------------------------------------------------------------ Table descent (paper's construction, rebuilt)
h = 0.01
its = one_d_iterates(h, 60)
uu = h * np.arange(len(its[0]))
tail = np.zeros((len(uu), len(uu)))
for n in range(1, 61):
    tail += 0.5 * (-0.25) ** n * np.outer(its[n], its[n])
tabd = {4: ('5.28e-7', '3.090e-2', 351), 8: ('2.46e-6', '2.330e-3', 376)}
gu, gv = np.gradient(tail, h)  # central inside, one-sided (first order) on the grid edge
for t, (perr, pmax, ppts) in tabd.items():
    cen = int(round(t / h))
    rr, vdiag, vgrad, ex, edge = [], [], [], [], []
    for m in range(1, cen):
        i, j = cen - m, cen + m  # u = t - r, v = t + r, r = m h
        if j >= len(uu) or i < 0:
            break
        r = m * h
        if r < 0.25 - 1e-12 or (t - r) < 0.25 - 1e-12:
            continue
        on_edge = (j == len(uu) - 1)
        dA = float('nan') if on_edge else (tail[i - 1, j + 1] - tail[i + 1, j - 1]) / (2 * h)  # central, along the diagonal
        dB = gv[i, j] - gu[i, j]                                                             # d_v - d_u
        w = t * t - r * r
        rr.append(r)
        vdiag.append(-dA / (2 * math.pi * r))
        vgrad.append(-dB / (2 * math.pi * r))
        ex.append(-sp.j1(math.sqrt(w)) / (4 * math.pi * math.sqrt(w)))
        edge.append(on_edge)
    rr, vdiag, vgrad, ex, edge = map(np.array, (rr, vdiag, vgrad, ex, edge))
    eg = np.abs(vgrad - ex)
    table_check("Table descent t=%d max error (d_v - d_u, one-sided at the grid edge)" % t, perr, float(eg.max()))
    info("Table descent t=%d" % t, "worst point r=%.2f (on grid edge v=12: %s); max error over points not on the grid edge: %.3e; with a central difference along the diagonal instead: %.3e; r range covered [%.2f, %.2f]"
         % (rr[int(eg.argmax())], bool(edge[int(eg.argmax())]), float(eg[~edge].max()), float(np.nanmax(np.abs(vdiag - ex))), rr[0], rr[-1]))
    table_check("Table descent t=%d max|exact|" % t, pmax, float(np.max(np.abs(ex))))
    check("Table descent t=%d points" % t, len(rr) == ppts, "printed %d, mine %d" % (ppts, len(rr)))
mp.mp.dps = 30
full = max(abs(mp.besselj(1, mp.sqrt(64 - r * r)) / (4 * mp.pi * mp.sqrt(64 - r * r))) for r in [mp.mpf('0.25') + mp.mpf('0.01') * j for j in range(0, 751)])
info("Table descent t=8 coverage", "the stated window r>=0.25, t-r>=0.25 has 751 grid points at t=8; the grid u,v<=12 only reaches r<=4, hence 376 points. Over the full stated window max|exact| = %s, ten times the printed 2.330e-3; the outer half 4<r<=7.75 of the cone interior at t=8 is not tested by the table." % mp.nstr(full, 5))

finish("A_pathsum")
