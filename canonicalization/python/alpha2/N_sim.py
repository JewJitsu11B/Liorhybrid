"""Agent N: direct simulation of the balanced alpha = 2 light-cone law (own code).

Written without opening the author's script.  numpy only.
Sections: lattice (KG comparison, mass, unitarity), near2, nr, nrlattice.
Usage: python N_sim.py [section ...]   (no argument = all)
Prints PASS|FAIL <label> <text>; exits non-zero if any check fails.
"""
import sys, time, math
import numpy as np

FAILS = []


def check(ok, label, text):
    print(("PASS " if ok else "FAIL ") + label + " " + text, flush=True)
    if not ok:
        FAILS.append(label)


def info(text):
    print("INFO " + text, flush=True)


# ----------------------------------------------------------------------
# Real-space lattice for 2J + K_2 J = S in 1+1.
# Def memory with d=1:  K_2 J(t,x) = tau_c^-2 int_0^inf tau [ (1/(2 c tau)) int_{|y|<=c tau} J(t-tau,x-y) dy ] dtau
#                                 = tau_c^-2/(2c) * (integral of J over the past light cone, dt dx)
# With u = ct-x, v = ct+x, du dv = 2c dt dx:  K_2 J = kappa * I,  kappa = tau_c^-2/(4c^2),
#   I(u,v) = int_{u'<=u} int_{v'<=v} J du' dv'.
# Composite 2-D trapezoid on a null lattice of cell h; slice n (t = n h/(2c)), site m (x = xmin + m h/2).
#   I[n,m] = I[n-1,m-1] + I[n-1,m+1] - I[n-2,m] + (h^2/4)(J[n,m] + J[n-1,m-1] + J[n-1,m+1] + J[n-2,m])
#   2 J[n,m] + kappa I[n,m] = S[n,m]      (linear in J[n,m], solved pointwise)
# ----------------------------------------------------------------------
def sh(a, k):
    out = np.zeros_like(a)
    if k > 0:
        out[..., k:] = a[..., :-k]
    else:
        out[..., :k] = a[..., -k:]
    return out


def run_lattice(c, tauc, h, xmin, xmax, tend, phis, tfac, t_rec, kprobe, snaps):
    dx = h / 2.0
    dt = h / (2.0 * c)
    M = int(round((xmax - xmin) / dx)) + 1
    x = xmin + dx * np.arange(M)
    P = np.array([p(x) for p in phis], dtype=complex)
    ns = P.shape[0]
    kappa = tauc ** -2 / (4.0 * c * c)
    q4 = h * h / 4.0
    den = 2.0 + kappa * q4
    mu = 1.0 / (math.sqrt(2.0) * tauc)
    Jm1 = np.zeros((ns, M), complex); Jm2 = np.zeros((ns, M), complex)
    Im1 = np.zeros((ns, M), complex); Im2 = np.zeros((ns, M), complex)
    E = np.exp(-1j * np.outer(kprobe, x)) * dx
    nsteps = int(round(tend / dt))
    snap_idx = {int(round(ts / dt)): ts for ts in snaps}
    out = dict(x=x, dt=dt, dx=dx, t=[], norm=[], gram=[], charge=[], grad=[], probe=[], snaps={})
    for n in range(nsteps + 1):
        t = n * dt
        S = tfac(t)[:, None] * P
        A = sh(Im1, 1) + sh(Im1, -1) - Im2 + q4 * (sh(Jm1, 1) + sh(Jm1, -1) + Jm2)
        J = (S - kappa * A) / den
        I = A + q4 * J
        if n in snap_idx:
            out['snaps'][snap_idx[n]] = J.copy()
        # diagnostics on slice n-1 (centred time derivative uses slices n and n-2)
        if n >= 2 and (t - dt) >= t_rec:
            Jc = Jm1
            dJ = (J - Jm2) / (2.0 * dt)
            out['t'].append(t - dt)
            out['norm'].append(np.sum(np.abs(Jc) ** 2, axis=1) * dx)
            out['gram'].append(np.sum(Jc[0] * np.conj(Jc[-1])) * dx)
            out['charge'].append(-np.sum(np.imag(np.conj(Jc) * dJ), axis=1) * dx / mu)
            gx = (sh(Jc, -2) - sh(Jc, 2)) / (2.0 * h)
            out['grad'].append(np.sum(np.abs(gx) ** 2, axis=1) * dx)
            out['probe'].append(Jc @ E.T)
        Jm2, Jm1 = Jm1, J
        Im2, Im1 = Im1, I
    for key in ('t', 'norm', 'gram', 'charge', 'grad', 'probe'):
        out[key] = np.array(out[key])
    return out


def fit_cos(a, q, dt):
    """least squares for a[n+q] + a[n-q] = 2 C a[n], C real -> omega."""
    mid = a[q:-q]
    C = np.real(np.sum(np.conj(mid) * (a[2 * q:] + a[:-2 * q]))) / (2.0 * np.sum(np.abs(mid) ** 2))
    return math.acos(max(-1.0, min(1.0, C))) / (q * dt)


def gauss_env(t, t0, sig):
    return math.exp(-(t - t0) ** 2 / (2 * sig * sig)) if 0.0 < t < 2 * t0 else 0.0


def kg_reference(x, phi, f, toff, t, c, mu, nq=20001, kmax=None):
    """Exact sourced solution (Thm KG(1)) per Fourier mode for S = f(t) phi(x), t >= toff:
    Jhat = -(mu^2/2) int sin(w (t-t'))/w f(t') dt' phihat  (f(t)=0 for t>=toff)."""
    M = x.size
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(M, dx)
    w = np.sqrt(c * c * k * k + mu * mu)
    Ph = np.fft.fft(phi)
    tq = np.linspace(0.0, toff, nq)
    fq = np.array([f(tt) for tt in tq])
    hq = tq[1] - tq[0]
    sel = np.abs(Ph) > 1e-13 * np.abs(Ph).max()
    A = np.zeros(M, complex)
    ws = w[sel]
    Fp = np.zeros(ws.size, complex); Fm = np.zeros(ws.size, complex)
    for i0 in range(0, ws.size, 512):
        ww = ws[i0:i0 + 512, None]
        Fp[i0:i0 + 512] = (np.exp(-1j * ww * tq[None, :]) * fq[None, :]).sum(1) * hq
        Fm[i0:i0 + 512] = (np.exp(+1j * ww * tq[None, :]) * fq[None, :]).sum(1) * hq
    integ = (np.exp(1j * ws * t) * Fp - np.exp(-1j * ws * t) * Fm) / (2j)
    A[sel] = -(mu * mu / 2.0) * integ / ws
    return np.fft.ifft(Ph * A)


def section_lattice():
    print("== lattice: balanced alpha=2 law in 1+1 (c=1, tau_c=1, mu=1/sqrt2) ==")
    c, tauc = 1.0, 1.0
    mu = 1.0 / (math.sqrt(2) * tauc)
    k0 = 1.0; w0 = math.sqrt(c * c * k0 * k0 + mu * mu)
    kb = 0.5; wb = math.sqrt(c * c * kb * kb + mu * mu)
    sig_t = 4.0; t0 = 6.5 * sig_t; toff = 2 * t0
    pa = lambda x: np.exp(-x ** 2 / 8.0 + 1j * k0 * x)
    pb = lambda x: np.exp(-(x - 3.0) ** 2 / 4.5 + 1j * kb * x)

    def tfac(t):
        g = gauss_env(t, t0, sig_t)
        return np.array([g * np.exp(-1j * w0 * t), g * np.exp(1j * w0 * t), g * (np.exp(-1j * w0 * t) + 0.5 * np.exp(1j * w0 * t)),
                         g * np.exp(-1j * wb * t)])
    tend = 83.0; t_rec = toff + 0.5
    snaps = [60.0, 70.0, 80.0]
    kprobe = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    names = ['positive', 'negative', 'mixed']
    res = {}
    errs = {}; mus = {}
    for h in (0.08, 0.04, 0.02):
        tic = time.time()
        r = run_lattice(c, tauc, h, -110.0, 115.0, tend, [pa, pa, pa, pb], tfac, t_rec, kprobe, snaps)
        res[h] = r
        x = r['x']
        # (a) compare with spectral Klein-Gordon evolution of the sourced problem
        e = []
        for ts in snaps:
            for si, sg in ((0, -1), (1, +1)):
                f = (lambda sg: (lambda tt: gauss_env(tt, t0, sig_t) * np.exp(sg * 1j * w0 * tt)))(sg)
                ref = kg_reference(x, pa(x), f, toff, ts, c, mu)
                num = r['snaps'][ts][si]
                e.append(np.linalg.norm(num - ref) / np.linalg.norm(ref))
        errs[h] = max(e)
        # (b) mass from dispersion, mixed (real-drive) run and positive run
        q = max(1, int(round(1.0 / r['dt'])))
        m2 = []
        for si in (2, 0):
            for ik, kk in enumerate(kprobe):
                w = fit_cos(r['probe'][:, si, ik], q, r['dt'])
                m2.append(w * w - c * c * kk * kk)
        m2 = np.array(m2)
        mus[h] = (math.sqrt(m2.mean()) * tauc, np.sqrt(m2.min()) * tauc, np.sqrt(m2.max()) * tauc)
        info("h=%.3f  max rel L2 error vs spectral KG (t=60,70,80; +/- runs) = %.3e ; mu*tau_c = %.6f (range %.6f..%.6f over 7 k, 2 runs)  [%.1fs]"
             % (h, errs[h], mus[h][0], mus[h][1], mus[h][2], time.time() - tic))
    hs = sorted(errs, reverse=True)
    rat = [errs[hs[i]] / errs[hs[i + 1]] for i in range(len(hs) - 1)]
    check(errs[hs[-1]] < 2e-3 and all(3.3 < r_ < 4.7 for r_ in rat), "Thm KG(1)-(2) lattice vs spectral KG",
          "rel L2 errors %s, ratios %s (second order => 4)" % (["%.3e" % errs[h] for h in hs], ["%.2f" % r_ for r_ in rat]))
    target = 1 / math.sqrt(2)
    d = [abs(mus[h][0] - target) for h in hs]
    check(d[-1] < 2e-4 and d[0] > d[-1], "Thm KG mass mu*tau_c=1/sqrt2",
          "measured %s vs 0.707107; |diff| %s" % (["%.6f" % mus[h][0] for h in hs], ["%.2e" % v for v in d]))
    # Richardson extrapolation of mu*tau_c
    ext = mus[hs[-1]][0] + (mus[hs[-1]][0] - mus[hs[-2]][0]) / 3.0
    check(abs(ext - target) < 2e-5, "Thm KG mass (Richardson h->0)", "mu*tau_c = %.7f vs %.7f" % (ext, target))

    # (c) unitarity and readout on the finest lattice
    r = res[hs[-1]]
    t = r['t']
    def spread(a):
        return (a.max() - a.min()) / abs(a.mean())
    for si, nm in enumerate(names):
        info("%-8s run: rel spread of ||J||^2 on t in [%.1f,%.1f] = %.3e ; of KG charge q = %.3e ; q/||J||^2 mean = %+.5f"
             % (nm, t[0], t[-1], spread(r['norm'][:, si]), spread(r['charge'][:, si]),
                (r['charge'][:, si] / r['norm'][:, si]).mean()))
    sp = [spread(res[h]['norm'][:, 0]) for h in hs]
    sn = [spread(res[h]['norm'][:, 1]) for h in hs]
    check(sp[-1] < 5e-4 and sn[-1] < 5e-4, "Thm unitary(1)-(2) trace on one sign of frequency",
          "rel spread of ||J||^2: positive %s, negative %s (h=%s)" % (["%.2e" % v for v in sp], ["%.2e" % v for v in sn], hs))
    sm = spread(r['norm'][:, 2])
    check(sm > 1e3 * sp[-1], "Thm unitary(3) mixed frequencies not conserved", "rel spread of ||J||^2 = %.3e (one-sign runs: %.1e); packet cross terms dephase after the source ends" % (sm, sp[-1]))
    dep = []
    for ik in range(kprobe.size):
        p = np.abs(r['probe'][:, 2, ik]) ** 2
        dep.append((p.max() - p.min()) / (p.max() + p.min()))
    info("mixed run, per-mode modulation depth of |Jhat_k|^2 at k=%s: %s" % (list(kprobe), ["%.3f" % v for v in dep]))
    check(min(dep) > 0.05, "Thm unitary(3) per-mode |Jhat_k|^2 oscillates", "modulation depth min %.3f max %.3f" % (min(dep), max(dep)))
    # per-mode oscillation of |Jhat(k)|^2 at 2 omega in the mixed run
    q = max(1, int(round(0.4 / r['dt'])))
    worst = 0.0
    for ik, kk in enumerate(kprobe):
        p = np.abs(r['probe'][:, 2, ik]) ** 2
        w2 = fit_cos(p - p.mean(), q, r['dt'])
        worst = max(worst, abs(w2 / (2 * math.sqrt(kk * kk + mu * mu)) - 1))
    check(worst < 2e-2, "Thm unitary(3) cross terms at 2 omega", "max rel deviation of fitted frequency of |Jhat_k|^2 from 2 omega(k) = %.2e" % worst)
    # charge: conserved on every solution, sign
    sq = [(r['charge'][:, si].max() - r['charge'][:, si].min()) / r['norm'][:, si].mean() for si in range(3)]
    check(max(sq) < 5e-4, "Thm unitary(4) KG charge conserved (+, -, mixed)", "spread of q / mean ||J||^2: %s" % ["%.2e" % v for v in sq])
    check(r['charge'][:, 0].min() > 0 and r['charge'][:, 1].max() < 0, "Thm unitary(4) sign of charge",
          "q(+) = %.5f, q(-) = %.5f, q(mixed) = %.3e (||J_mixed||^2 = %.5f)" % (r['charge'][:, 0].mean(), r['charge'][:, 1].mean(), r['charge'][:, 2].mean(), r['norm'][:, 2].mean()))
    # two norms: 0 <= q - ||J||^2 <= (c^2/(2 mu^2)) ||grad J||^2   (hbar^2/(2 m^2 c^2) = c^2/(2 mu^2))
    gap = r['charge'][:, 0] - r['norm'][:, 0]
    bound = c * c / (2 * mu * mu) * r['grad'][:, 0]
    check(gap.min() > 0 and (bound - gap).min() > 0, "Thm unitary(4) / Remark two norms inequality",
          "q-||J||^2 in [%.5f,%.5f], bound in [%.5f,%.5f], ||J||^2=%.5f" % (gap.min(), gap.max(), bound.min(), bound.max(), r['norm'][:, 0].mean()))
    # closure matrix Q for J = [[Ja, Jb],[0, Ja+Jb]] (both positive frequency)
    Gaa = r['norm'][:, 0]; Gbb = r['norm'][:, 3]; Gab = r['gram']
    Q11 = Gaa + Gbb; Q22 = Gaa + Gbb + 2 * Gab.real; Q12 = np.conj(Gab) + Gbb
    p1 = Q11 / (Q11 + Q22)
    sQ = max(spread(Q11), spread(Q22), (np.abs(Q12 - Q12.mean())).max() / abs(Q12.mean()))
    check(sQ < 1e-3 and (p1.max() - p1.min()) < 1e-3, "Thm unitary(2) closure Q and readout p_m constant",
          "max rel variation of Q entries %.2e; p_1 = %.6f +- %.1e; |<Ja,Jb>| = %.4f" % (sQ, p1.mean(), (p1.max() - p1.min()) / 2, abs(Gab.mean())))

    # (d) second units check: c = 2, tau_c = 0.5  (mu = sqrt2)
    c2, tc2 = 2.0, 0.5
    mu2 = 1 / (math.sqrt(2) * tc2)
    w02 = math.sqrt(c2 * c2 + mu2 * mu2)
    st = 1.5; t02 = 6.5 * st
    tf2 = lambda t: np.array([gauss_env(t, t02, st) * math.cos(w02 * t)])
    kp = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    mm = []
    for h in (0.04, 0.02):
        r2 = run_lattice(c2, tc2, h, -75.0, 75.0, 30.0, [pa], tf2, 2 * t02 + 0.5, kp, [])
        q = max(1, int(round(0.4 / r2['dt'])))
        m2 = np.array([fit_cos(r2['probe'][:, 0, i], q, r2['dt']) ** 2 - c2 * c2 * kp[i] ** 2 for i in range(kp.size)])
        mm.append(math.sqrt(m2.mean()) * tc2)
    ext = mm[1] + (mm[1] - mm[0]) / 3
    check(abs(ext - target) < 1e-4, "Thm KG mass, units check c=2 tau_c=0.5", "mu*tau_c = %.6f, %.6f -> extrap %.6f vs 0.707107" % (mm[0], mm[1], ext))


# ----------------------------------------------------------------------
# Near alpha = 2: order-alpha law per spatial Fourier mode (Def symbol: profile acts as sinc(c|k|tau))
#   alpha J(t) + (alpha-1) tau_c^-alpha int_0^t tau^(alpha-1)/Gamma(alpha) sinc(b tau) J(t-tau) dtau = S(t)
# ----------------------------------------------------------------------
def volterra(alpha, tauc, b, dt, nsteps, S):
    j = np.arange(nsteps + 1)
    tau = j * dt
    w = tau ** (alpha - 1) / math.gamma(alpha) * np.sinc(b * tau / np.pi)
    w = (alpha - 1) * tauc ** -alpha * dt * w
    J = np.zeros(nsteps + 1)
    den = alpha + 0.5 * w[0]
    for n in range(nsteps + 1):
        acc = 0.0
        if n >= 1:
            acc = np.dot(w[1:n + 1], J[n - 1::-1][:n])
            acc -= 0.5 * w[n] * J[0]
        J[n] = (S[n] - acc) / den
    return J


def symT(alpha, w, b, tauc):
    """symbol of T = alpha + (alpha-1) m_alpha at s = -i w (closed form of Prop symbol, principal branch)."""
    s = -1j * w
    return alpha + tauc ** -alpha * ((s - 1j * b) ** (1 - alpha) - (s + 1j * b) ** (1 - alpha)) / (2j * b)


def root_near(alpha, b, tauc, w0):
    w = complex(w0)
    for _ in range(200):
        f = symT(alpha, w, b, tauc)
        d = 1e-7
        fp = (symT(alpha, w + d, b, tauc) - symT(alpha, w - d, b, tauc)) / (2 * d)
        step = f / fp
        w -= step
        if abs(step) < 1e-14:
            break
    return w


def fit_damped(J, q, dt):
    mid = J[q:-q]; plus = J[2 * q:]; minus = J[:-2 * q]
    # plus + B minus = A mid
    Mx = np.stack([mid, -minus], axis=1)
    sol, *_ = np.linalg.lstsq(Mx, plus, rcond=None)
    A, B = sol
    gam = math.log(B) / (2 * q * dt)
    om = math.acos(A / (2 * math.sqrt(B))) / (q * dt)
    return gam, om


def section_near2():
    print("== near alpha=2: time-stepped order-alpha law, one mode (c=1, tau_c=1/sqrt2, k=1) ==")
    tauc = 1 / math.sqrt(2); b = 1.0
    Sfun = lambda t: np.where((t > 0) & (t < 12), np.exp(-(t - 6.0) ** 2 / 2.0), 0.0)
    T = 300.0
    rows = []
    for alpha in (1.90, 1.95, 2.00, 2.05, 2.10):
        rt = root_near(alpha, b, tauc, math.sqrt(2) + 0.0j)
        mu2 = rt.real ** 2 - b * b
        thm = (math.pi / 4) * (alpha - 2) * mu2 / abs(rt.real)
        fits = []
        for dt in (0.04, 0.02):
            n = int(round(T / dt))
            t = dt * np.arange(n + 1)
            J = volterra(alpha, tauc, b, dt, n, Sfun(t))
            i1, i2 = int((30 if alpha < 2 else 40) / dt), int((80 if alpha < 2 else 300) / dt)
            g, om = fit_damped(J[i1:i2], int(round(0.8 / dt)), dt)
            fits.append((g, om))
            if dt == 0.02:
                a50 = np.abs(J[int(45 / dt):int(55 / dt)]).max(); a200 = np.abs(J[int(195 / dt):int(205 / dt)]).max()
                Jkeep = J; tkeep = t
        g, om = fits[-1]
        gext = g + (g - fits[0][0]) / 3
        rows.append((alpha, rt, thm, g, gext, om))
        info("alpha=%.2f root(my Newton)=%.6f%+.4ei ; Thm near2 rate=%+.4e ; time-stepped rate dt=.04: %+.4e dt=.02: %+.4e extrap %+.4e ; omega fit %.5f ; env(t~50)=%.3e env(t~200)=%.3e"
             % (alpha, rt.real, rt.imag, thm, fits[0][0], g, gext, om, a50, a200))
        if alpha == 2.0:
            check(abs(gext) < 2e-5 and abs(om - math.sqrt(2)) < 2e-3, "Thm near2(1) conservative at alpha=2",
                  "fitted rate %+.2e (dt extrap %+.2e), omega %.5f vs %.5f; bounded: max|J| t in[50,300] = %.3e" % (g, gext, om, math.sqrt(2), np.abs(Jkeep[int(50 / 0.02):]).max()))
        else:
            check(abs(gext - rt.imag) < 0.02 * abs(rt.imag) and (gext < 0) == (alpha < 2), "Thm near2(2) alpha=%.2f rate vs exact root" % alpha,
                  "time-stepped %+.4e vs Im root %+.4e (rel diff %.2e)" % (gext, rt.imag, abs(gext / rt.imag - 1)))
            rel = abs(rt.imag / thm - 1)
            check(rel < 1.2 * abs(alpha - 2) + 1e-3, "Thm near2(2) alpha=%.2f first-order formula" % alpha,
                  "Im root %+.4e vs (pi/4)(alpha-2)mu^2/|omega| = %+.4e, rel diff %.3f (delta=%.2f; O(delta) allowed)" % (rt.imag, thm, rel, alpha - 2))
    # convergence of the first-order formula as delta -> 0
    for dl in (1e-2, 1e-3, 1e-4):
        for sgn in (-1, 1):
            a = 2 + sgn * dl
            rt = root_near(a, b, tauc, math.sqrt(2) + 0.0j)
            thm = (math.pi / 4) * (a - 2) * (rt.real ** 2 - 1) / rt.real
            info("alpha=2%+.0e: Im root %+.6e, formula %+.6e, (Im/formula - 1)/delta = %+.3f" % (sgn * dl, rt.imag, thm, (rt.imag / thm - 1) / (a - 2)))
    # both branches: -conj(root) is a root too (real kernel)
    r1 = root_near(2.05, b, tauc, math.sqrt(2)); r2 = root_near(2.05, b, tauc, -math.sqrt(2))
    check(abs(r2 + np.conj(r1)) < 1e-10, "Thm near2(2) both signs same rate", "roots %.6f%+.6fi and %.6f%+.6fi" % (r1.real, r1.imag, r2.real, r2.imag))
    # late-time behaviour below 2: exponential (pole) or something else?  Longer run, two steps.
    T2 = 700.0
    tails = {}
    for dt in (0.04, 0.02):
        n = int(round(T2 / dt)); t = dt * np.arange(n + 1)
        J = volterra(1.90, tauc, b, dt, n, Sfun(t))
        tt = [100, 200, 300, 400, 500, 600, 690]
        ev = [np.abs(J[int((a - 8) / dt):int((a + 8) / dt)]).max() for a in tt]
        pw = [-(math.log(ev[i + 1] / ev[i]) / math.log(tt[i + 1] / tt[i])) for i in range(len(tt) - 1)]
        seg = J[int(400 / dt):]
        tseg = t[int(400 / dt):]
        om_tail = fit_cos((seg * tseg ** 1.9).astype(complex), int(round(0.8 / dt)), dt)
        tails[dt] = (ev, pw, om_tail)
        info("alpha=1.90 dt=%.2f envelope max|J| near t=%s: %s" % (dt, tt, ["%.3e" % v for v in ev]))
        info("alpha=1.90 dt=%.2f local power-law exponents %s ; oscillation frequency for t>400: %.4f (c|k| = 1, pole Re omega = 1.3700)" % (dt, ["%.2f" % v for v in pw], om_tail))
    ev, pw, om_tail = tails[0.02]
    pole_at_600 = ev[0] * math.exp(-5.4354e-2 * 500)
    info("FINDING Thm near2 (not in paper): for alpha<2 the sourced solution does not decay exponentially at late time; "
         "pole part extrapolated to t=600 would be %.1e, measured %.3e; tail ~ t^-%.2f at frequency %.3f = c|k| (expected t^-alpha from the light-cone branch points omega=+-c|k|)"
         % (pole_at_600, ev[5], pw[-1], om_tail))
    check(abs(pw[-1] - 1.9) < 0.15 and abs(om_tail - 1.0) < 0.02 and abs(tails[0.04][0][5] / ev[5] - 1) < 0.05,
          "Thm near2(2) late-time tail below 2 (extra, not in paper)", "power law exponent %.2f (alpha=1.9), frequency %.4f (c|k|=1), dt-independent to %.1e"
          % (pw[-1], om_tail, abs(tails[0.04][0][5] / ev[5] - 1)))


# ----------------------------------------------------------------------
def section_nr():
    print("== nonrelativistic limit, spectral (hbar=m=1, packet exp(-x^2/8+ix), t=10) ==")
    N = 8192; L = 400.0
    x = (np.arange(N) - N // 2) * (L / N)
    k = 2 * np.pi * np.fft.fftfreq(N, L / N)
    psi = np.exp(-x ** 2 / 8 + 1j * x)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * (L / N))
    ph = np.fft.fft(psi, norm='ortho')
    ph /= np.sqrt(np.sum(np.abs(ph) ** 2))
    t = 10.0
    k4 = np.sqrt(np.sum(np.abs(k ** 4 * ph) ** 2)); k2 = np.sqrt(np.sum(np.abs(k ** 2 * ph) ** 2))
    lim = t / 8 * k4
    info("(t/8)||k^4 psihat|| = %.6f (paper: 3.672)" % lim)
    prev = None; ok = True; ok2 = True; last = None
    for c in (2, 4, 8, 16, 32, 64, 128):
        w = np.sqrt(c ** 2 * k ** 2 + c ** 4)
        a = np.exp(-1j * (w - c ** 2) * t) * ph
        s = np.exp(-1j * k ** 2 * t / 2) * ph
        err = np.sqrt(np.sum(np.abs(a - s) ** 2))
        drift = abs(np.sum(np.abs(a) ** 2) - 1)
        # Cor feynman multiplier
        f = (c ** 2 / w) * a
        errF = np.sqrt(np.sum(np.abs(f - s) ** 2)); bF = k2 / (2 * c * c) + lim / c ** 2
        ok &= err <= lim / c ** 2 * (1 + 1e-12); ok2 &= errF <= bF * (1 + 1e-12)
        info("c=%4d  ||KG-Schr|| = %.4e  err*c^2 = %.4f  bound*c^2 = %.4f  norm drift %.1e | Cor feynman err %.4e <= bound %.4e"
             % (c, err, err * c * c, lim, drift, errF, bF))
        last = err * c * c
    check(ok, "Thm NR bound", "error <= t/(8 c^2) ||Delta^2 psi|| for c=2..128")
    check(abs(last / lim - 1) < 1e-3, "Thm NR rate O(1/c^2) and coefficient", "err*c^2 -> %.5f at c=128 vs (t/8)||k^4 psihat|| = %.5f" % (last, lim))
    check(ok2, "Cor feynman bound", "error of (mc^2/(hbar omega)) multiplier <= stated bound for c=2..128")
    # composition law
    c = 4.0; w = np.sqrt(c ** 2 * k ** 2 + c ** 4)
    U = lambda tt: np.exp(-1j * (w - c * c) * tt)
    check(np.abs(U(0.7) * U(1.9) - U(2.6)).max() < 1e-12, "Thm unitary(1) group law", "max |U(t1)U(t2)-U(t1+t2)| = %.1e" % np.abs(U(0.7) * U(1.9) - U(2.6)).max())


def section_nrlattice():
    print("== nonrelativistic limit from the lattice law itself (hbar=m=1; mu=c^2, tau_c=1/(sqrt2 c^2)) ==")
    pa = lambda x: np.exp(-x ** 2 / 8.0 + 1j * x)
    for c, hs_ in ((2.0, (0.04, 0.02)), (4.0, (0.02, 0.01))):
        mu = c * c; tauc = 1 / (math.sqrt(2) * mu)
        w0 = math.sqrt(c * c + mu * mu)
        st = 1.0; t0 = 6.5 * st; t1 = 2 * t0 + 0.5; t2 = t1 + 10.0
        tf = lambda t: np.array([gauss_env(t, t0, st) * np.exp(-1j * w0 * t)])
        X = 22.0 + c * t2
        got = []
        for h in hs_:
            tic = time.time()
            r = run_lattice(c, tauc, h, -X, X, t2 + 0.01, [pa], tf, 1e9, np.array([1.0]), [t1, t2])
            x = r['x']; dx = r['dx']
            a = r['snaps'][t1][0] * np.exp(1j * mu * t1); bb = r['snaps'][t2][0] * np.exp(1j * mu * t2)
            k = 2 * np.pi * np.fft.fftfreq(x.size, dx)
            ah = np.fft.fft(a)
            sch = np.fft.ifft(ah * np.exp(-1j * k ** 2 * 10.0 / 2))
            w = np.sqrt(c * c * k * k + mu * mu)
            kgs = np.fft.ifft(ah * np.exp(-1j * (w - mu) * 10.0))
            nrm = np.linalg.norm(a)
            e_ls = np.linalg.norm(bb - sch) / nrm          # lattice law vs Schrodinger
            e_lk = np.linalg.norm(bb - kgs) / nrm          # lattice law vs spectral positive-frequency KG
            e_ks = np.linalg.norm(kgs - sch) / nrm         # spectral KG vs Schrodinger on the same state
            bound = 10.0 / (8 * c * c) * np.linalg.norm(k ** 4 * ah) / np.linalg.norm(ah)
            got.append((h, e_ls, e_lk, e_ks, bound))
            info("c=%g h=%.3f: lattice vs Schr %.4e ; lattice vs spectral KG+ %.3e ; spectral KG+ vs Schr %.4e ; Thm NR bound %.4e  [%.0fs]"
                 % (c, h, e_ls, e_lk, e_ks, bound, time.time() - tic))
        h, e_ls, e_lk, e_ks, bound = got[-1]
        check(e_lk < 0.1 * e_ks and e_ks <= bound, "Thm NR on lattice-evolved state c=%g" % c,
              "lattice-vs-Schr %.4e (x c^2 = %.3f), bound %.4e, lattice discretisation error %.1e" % (e_ls, e_ls * c * c, bound, e_lk))


if __name__ == '__main__':
    secs = sys.argv[1:] or ['lattice', 'near2', 'nr', 'nrlattice']
    t0_ = time.time()
    for s_ in secs:
        globals()['section_' + s_]()
    print("elapsed %.0fs; failed checks: %s" % (time.time() - t0_, FAILS if FAILS else "none"))
    sys.exit(1 if FAILS else 0)
