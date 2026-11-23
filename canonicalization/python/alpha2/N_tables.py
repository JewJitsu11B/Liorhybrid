"""Agent N: Sec. 9 tables rebuilt from the paper's own method description (Appendix B),
written before opening the author's script.  Prints N-TABLE lines for later comparison,
and PASS|FAIL against the numbers printed in the paper (to the printed digits)."""
import sys, math
import numpy as np
from scipy.special import j0, j1

FAILS = []


def check(ok, label, text):
    print(("PASS " if ok else "FAIL ") + label + " " + text, flush=True)
    if not ok:
        FAILS.append(label)


def agree(val, printed, fmt):
    """True if val formatted with fmt equals the printed string (allow +-1 in last digit)."""
    s = fmt % val
    if s == printed:
        return True
    try:
        a, b = float(s), float(printed)
        m = printed.lower().split('e')[0]
        dec = len(m.split('.')[1]) if '.' in m else 0
        ex = int(printed.lower().split('e')[1]) if 'e' in printed.lower() else 0
        return abs(a - b) <= 1.01 * 10 ** (ex - dec)
    except Exception:
        return False


mu = 1.0; tauc = 1 / math.sqrt(2)

# ---- Table symbol
print("== Table 1 (tab:symbol) ==")
uu = np.linspace(-14, 8, 400001); tau = np.exp(uu)
paper = {1.5: ("1.054093", ["1.0050", "1.0468", "1.1395", "1.3075"]), 2.0: ("1.000000", ["1.0000"] * 4),
         2.5: ("0.913547", ["0.9917", "0.9249", "0.7903", "0.5840"]), 3.0: ("0.800000", ["0.9801", "0.8253", "0.5403", "0.1700"])}
s_, b_ = 0.8, 0.6
h_ = lambda a, p: (p / math.sin(p)) if a == 1 else math.sin((a - 1) * p) / ((a - 1) * math.sin(p))
for a, (pn, ph) in paper.items():
    f = tau ** (a - 1) / math.gamma(a) * np.exp(-s_ * tau) * np.sinc(b_ * tau / np.pi) * tau
    num = np.sum((f[1:] + f[:-1]) / 2 * np.diff(uu))
    cf = ((s_ - 1j * b_) ** (1 - a) - (s_ + 1j * b_) ** (1 - a)) / (2j * b_ * (a - 1))
    hv = [h_(a, p) for p in (0.2, 0.6, 1.0, 1.4)]
    print("N-TABLE symbol alpha=%.1f numeric %.6f closed %.6f h: %s" % (a, num, cf.real, " ".join("%.4f" % v for v in hv)))
    check(agree(num, pn, "%.6f") and agree(cf.real, pn, "%.6f") and all(agree(v, p, "%.4f") for v, p in zip(hv, ph)),
          "Table 1 row alpha=%.1f" % a, "numeric %.6f closed %.6f (paper %s); h %s (paper %s)" % (num, cf.real, pn, ["%.4f" % v for v in hv], ph))
phi = np.linspace(1e-6, np.pi / 2, 200001)
pd = {0.5: "4.1e-01", 1.5: "4.1e-01", 1.9: "9.7e-02", 2.0: "0", 2.1: "1.0e-01", 3.0: "1.0e+00"}
for a, pp in pd.items():
    d = np.abs(np.sin((a - 1) * phi) / ((a - 1) * np.sin(phi)) - 1).max()
    print("N-TABLE defect alpha=%.1f %.3e" % (a, d))
    check((d < 1e-12) if pp == "0" else agree(d, pp, "%.1e"), "Sec 9 defect alpha=%.1f" % a, "%.3e (paper %s)" % (d, pp))

# ---- Table disp
print("== Table 2 (tab:disp) ==")
pdisp = {0.0: "1.0000000000", 0.5: "1.1180339887", 1.0: "1.4142135624", 2.0: "2.2360679775", 4.0: "4.1231056256"}
for k, pp in pdisp.items():
    F = lambda E: (2 + tauc ** -2 / ((1e-12 - 1j * E) ** 2 + k * k)).real
    lo, hi = abs(k), 50.0
    if k == 0:
        lo = 1e-9
    flo = F(lo + 1e-9)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (F(mid) > 0) == (flo > 0):
            lo = mid
        else:
            hi = mid
    E = 0.5 * (lo + hi)
    print("N-TABLE disp k=%.1f root %.10f exact %.10f tachyon E^2 %.2f" % (k, E, math.sqrt(k * k + 1), k * k - 1))
    check(agree(E, pp, "%.10f"), "Table 2 row k=%.1f" % k, "root %.10f (paper %s)" % (E, pp))

# ---- Table near2
print("== Table 3 (tab:near2) ==")


def symT(alpha, w, b):
    s = -1j * w
    if abs(b) < 1e-14:
        return alpha + (alpha - 1) * (tauc * s) ** (-alpha)
    return alpha + tauc ** -alpha * ((s - 1j * b) ** (1 - alpha) - (s + 1j * b) ** (1 - alpha)) / (2j * b)


def newton(alpha, b, w0):
    w = complex(w0)
    for _ in range(200):
        d = 1e-7
        st = symT(alpha, w, b) / ((symT(alpha, w + d, b) - symT(alpha, w - d, b)) / (2 * d))
        w -= st
        if abs(st) < 1e-14:
            break
    return w


def winding(alpha, b, y0=0.002, y1=12.0, X=12.0, n=200000):
    xs = np.linspace(-X, X, n); ys = np.linspace(y0, y1, n)
    path = np.concatenate([xs + 1j * y0, X + 1j * ys, xs[::-1] + 1j * y1, -X + 1j * ys[::-1]])
    f = symT(alpha, path, b)
    ang = np.unwrap(np.angle(f))
    return (ang[-1] - ang[0]) / (2 * np.pi)


pn2 = {1.90: ("1.370022", "-5.435e-02", "5.03e-02", "0,0,0"), 1.99: ("1.410296", "-5.546e-03", "5.51e-03", "0,0,0"),
       2.01: ("1.418025", "5.560e-03", "5.60e-03", "2,2,0"), 2.05: ("1.432250", "2.790e-02", "2.88e-02", "2,2,2"),
       2.10: ("1.447853", "5.590e-02", "5.95e-02", "2,2,2")}
for a, (pr, pi_, pf, pg) in pn2.items():
    r = newton(a, 1.0, math.sqrt(2)); r2 = newton(a, 1.0, -math.sqrt(2))
    form = (math.pi / 4) * abs(a - 2) * (r.real ** 2 - 1) / abs(r.real)
    wn = [winding(a, kk) for kk in (0.1, 1.0, 5.0)]
    g = ",".join("%d" % round(v) for v in wn)
    print("N-TABLE near2 alpha=%.2f root %.6f Im %+.3e (other branch %+.3e) formula %.2e winding %s" % (a, r.real, r.imag, r2.imag, form, g))
    check(agree(r.real, pr, "%.6f") and agree(r.imag, pi_, "%.3e") and agree(form, pf, "%.2e") and g == pg,
          "Table 3 row alpha=%.2f" % a, "root %.6f%+.3ei formula %.2e growing %s (paper %s %s %s %s)" % (r.real, r.imag, form, g, pr, pi_, pf, pg))
r5 = newton(2.01, 5.0, math.sqrt(26))
print("N-TABLE near2 alpha=2.01 k=5 Im %.3e" % r5.imag)
check(agree(r5.imag, "1.5e-03", "%.1e"), "Table 3 caption growth rate alpha=2.01 k=5", "%.2e (paper 1.5e-3)" % r5.imag)
# wider search for growing modes: larger rectangle, lower edge closer to the axis
for a in (2.01, 2.05, 2.10):
    w2 = [winding(a, kk, y0=1e-4, y1=200.0, X=200.0, n=400000) for kk in (0.1, 1.0, 5.0)]
    print("N-TABLE near2 alpha=%.2f winding on Im in [1e-4,200], |Re|<=200: %s" % (a, ",".join("%.2f" % v for v in w2)))

# ---- Table pathsum
print("== Table 4 (tab:pathsum) ==")
pps = {0.02: ["1.44e+02", "1.15e+03", "3.28e+01", "3.35e-05", "3.35e-05", "3.35e-05"],
       0.01: ["1.44e+02", "1.15e+03", "3.28e+01", "8.38e-06", "8.38e-06", "8.38e-06"]}
sums = {}
for h in (0.02, 0.01):
    n = int(round(12 / h)) + 1
    u = np.linspace(0, 12, n)
    U, V = np.meshgrid(u, u, indexing='ij')
    exact = 0.5 * j0(np.sqrt(U * V))

    def cum(a, ax):
        out = np.zeros_like(a)
        sl1 = [slice(None)] * 2; sl0 = [slice(None)] * 2
        sl1[ax] = slice(1, None); sl0[ax] = slice(None, -1)
        out[tuple(sl1)] = np.cumsum((a[tuple(sl1)] + a[tuple(sl0)]) * (h / 2), axis=ax)
        return out
    term = np.full((n, n), 0.5); tot = term.copy(); errs = {}; tail = np.zeros((n, n))
    for m in range(1, 61):
        term = 0.25 * cum(cum(term, 0), 1)
        tot += (-1) ** m * term
        tail += (-1) ** m * term
        if m in (2, 5, 10, 20, 40, 60):
            errs[m] = np.abs(tot - exact).max()
    sums[h] = (u, tail)
    vals = [errs[m] for m in (2, 5, 10, 20, 40, 60)]
    print("N-TABLE pathsum h=%.3f " % h + " ".join("%.2e" % v for v in vals))
    check(all(agree(v, p, "%.2e") for v, p in zip(vals, pps[h])), "Table 4 row h=%.3f" % h, "%s (paper %s)" % (["%.2e" % v for v in vals], pps[h]))

# ---- Table checker
print("== Table 5 (tab:checker) ==")
pck = {0.02: "3.697e-02", 0.01: "1.810e-02", 0.005: "8.950e-03", 0.0025: "4.451e-03"}
prev = None
for eps, pp in pck.items():
    N = int(round(8 / eps))
    R = np.zeros(2 * N + 3, complex); L = np.zeros(2 * N + 3, complex)
    o = N + 1
    R[o] = 1.0   # one amplitude at the origin about to move right
    for _ in range(N):
        Rn = np.zeros_like(R); Ln = np.zeros_like(L)
        Rn[1:] = R[:-1] + 1j * eps * mu * L[:-1]
        Ln[:-1] = L[1:] + 1j * eps * mu * R[1:]
        R, L = Rn, Ln
    j = np.arange(-N - 1, N + 2); x = j * eps
    sel = (np.abs(x) <= 7.5 + 1e-12) & ((j + N) % 2 == 0)
    err = np.abs(L[sel] / (1j * eps * mu) - j0(mu * np.sqrt(64 - x[sel] ** 2))).max()
    # also the closed binomial-free variant: exact lattice site parity is (j+N) even
    print("N-TABLE checker eps=%.4f err %.3e ratio %s" % (eps, err, "--" if prev is None else "%.2f" % (prev / err)))
    check(agree(err, pp, "%.3e"), "Table 5 row eps=%g" % eps, "%.3e (paper %s)" % (err, pp))
    prev = err

# ---- Table descent
print("== Table 6 (tab:descent) ==")
pde = {4: ("5.28e-07", "3.090e-02", 351), 8: ("2.46e-06", "2.330e-03", 376)}
h = 0.01; u, tail = sums[h]; n = u.size
for t, (pe, pm, pc) in pde.items():
    errs = []; ex = []; rmax = 0
    for i in range(1, n - 1):
        jv = int(round(2 * t / h)) - i
        if jv < 1 or jv > n - 2:
            continue
        r = (u[jv] - u[i]) / 2
        if r < 0.25 - 1e-9 or t - r < 0.25 - 1e-9:
            continue
        dr = (tail[i, jv + 1] - tail[i, jv - 1]) / (2 * h) - (tail[i + 1, jv] - tail[i - 1, jv]) / (2 * h)
        val = -dr / (2 * np.pi * r)
        w = t * t - r * r
        e = -mu * j1(mu * math.sqrt(w)) / (4 * np.pi * math.sqrt(w))
        errs.append(abs(val - e)); ex.append(abs(e)); rmax = max(rmax, r)
    print("N-TABLE descent t=%d maxerr %.2e max|exact| %.3e points %d (r up to %.2f of cone radius %d)" % (t, max(errs), max(ex), len(errs), rmax, t))
    check(agree(max(errs), pe, "%.2e") and agree(max(ex), pm, "%.3e") and len(errs) == pc, "Table 6 row t=%d" % t,
          "err %.2e max %.3e pts %d (paper %s %s %d)" % (max(errs), max(ex), len(errs), pe, pm, pc))

# ---- bullets: pair composition, trace readout, NR table
print("== Sec 9 bullets and Table 7 (tab:NR) ==")
rng = np.random.default_rng(0)
ks = rng.uniform(-5, 5, 200); worst = 0
for k in ks:
    w = math.sqrt(k * k + 1)
    M = lambda t: np.array([[math.cos(w * t), math.sin(w * t) / w], [-w * math.sin(w * t), math.cos(w * t)]])
    worst = max(worst, np.linalg.norm(M(2.6) - M(1.9) @ M(0.7), 2))
print("N-TABLE pair composition max %.1e (k uniform in [-5,5], seed 0; paper 1.1e-14, distribution not stated)" % worst)
check(worst < 1e-13, "Sec 9 pair composition (Prop pair)", "max ||M(t1+t2)-M(t2)M(t1)|| = %.1e (paper 1.1e-14)" % worst)

N = 4096; Lb = 400.0
x = (np.arange(N) - N // 2) * (Lb / N); k = 2 * np.pi * np.fft.fftfreq(N, Lb / N)
psi = np.exp(-x ** 2 / 8 + 1j * x); ph = np.fft.fft(psi, norm='ortho'); ph /= np.linalg.norm(ph)
lim = 10 / 8 * np.linalg.norm(k ** 4 * ph)
print("N-TABLE NR limit %.3f" % lim)
check(agree(lim, "3.672", "%.3f"), "Sec 9 predicted limit (t/8)||k^4 psihat||", "%.4f (paper 3.672)" % lim)
pnr = {2: ("6.058e-01", "2.423"), 4: ("2.076e-01", "3.321"), 8: ("5.608e-02", "3.589"), 16: ("1.426e-02", "3.651"), 32: ("3.581e-03", "3.667")}
for c, (pe, px) in pnr.items():
    w = np.sqrt(c * c * k * k + c ** 4)
    a = np.exp(-1j * (w - c * c) * 10) * ph; s = np.exp(-1j * k * k * 10 / 2) * ph
    err = np.linalg.norm(a - s)
    xa = np.fft.ifft(a, norm='ortho')
    print("N-TABLE NR c=%d err %.3e errc2 %.3f drift %.1e" % (c, err, err * c * c, abs(np.linalg.norm(xa) - 1)))
    check(agree(err, pe, "%.3e") and agree(err * c * c, px, "%.3f"), "Table 7 row c=%d" % c, "%.3e, x c^2 = %.3f (paper %s, %s)" % (err, err * c * c, pe, px))
# trace readout: the paper does not say which c, which mixed solution; try the natural choices
ts = np.linspace(0, 10, 201)
for c in (1.0, 2.0):
    w = np.sqrt(c * c * k * k + c ** 4)
    for lab, am in (("equal mix a+ = a- = psihat", 1.0), ("a- = 0.5 a+", 0.5), ("J real part: a-(k) = conj a+(-k)", None)):
        nn = []; qq = []
        for t in ts:
            if am is None:
                J = np.real(np.fft.ifft(np.exp(-1j * w * t) * ph, norm='ortho')).astype(complex)
                Jt = np.real(np.fft.ifft(-1j * w * np.exp(-1j * w * t) * ph, norm='ortho')).astype(complex)
                nn.append(np.sum(np.abs(J) ** 2)); qq.append(0.0)
                continue
            jh = np.exp(-1j * w * t) * ph + am * np.exp(1j * w * t) * ph
            jt = -1j * w * np.exp(-1j * w * t) * ph + am * 1j * w * np.exp(1j * w * t) * ph
            nn.append(np.sum(np.abs(jh) ** 2)); qq.append(-np.sum(np.imag(np.conj(jh) * jt)) / c ** 2)
        nn = np.array(nn); qq = np.array(qq)
        print("N-TABLE trace mixed c=%g %-34s rel spread ||J||^2 %.2e ; KG charge spread %.1e (mean %.3f)"
              % (c, lab, (nn.max() - nn.min()) / nn.mean(), (qq.max() - qq.min()) / (abs(qq.mean()) + 1e-300) if am not in (None, 1.0) else (qq.max() - qq.min()), qq.mean()))
print("failed: %s" % (FAILS if FAILS else "none"))
sys.exit(1 if FAILS else 0)
