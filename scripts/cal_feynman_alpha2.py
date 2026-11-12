#!/usr/bin/env python3
"""Run the CAL at closure order alpha = 2 with a light-cone kernel: the Feynman path sum.

Companion to docs/paper/cal_feynman_alpha2.tex.  Requires numpy only.
Units: hbar = c = 1 unless stated, and mu = m c^2 / hbar.

Sections
  1  Light-cone memory of order alpha: closed-form symbol against direct integration,
     and Lorentz invariance only at alpha = 2 (and trivially alpha = 0).
  2  The alpha = 2 law gives Klein-Gordon: dispersion from the retarded symbol.
     The channel-phase alternative gives a tachyon.
  3  Path sum in 1+1: iterated accumulation along the two null legs (one mass insertion
     per step) against the massive propagator (1/2) J0(mu sqrt(uv)); grid refinement.
  3b Feynman's checkerboard on a lattice (amplitude i eps mu per corner): the corner-odd
     amplitude divided by i eps mu converges to J0, i.e. to 2 c G_mu.  The CAL term with
     n insertions is the (2n+1)-corner amplitude over 2 i c mu, since (-mu^2)^n = (i mu)^(2n).
  4  Path sum in 3+1 by exact descent: the tail inside the cone against
     -mu J1(mu sqrt(w)) / (4 pi sqrt(w)).
  5  Positive frequency: composition on the pair (J, dJ/dt), unitary evolution, the
     nonrelativistic limit to Feynman's free kernel, and why the positive-frequency
     restriction is needed for the trace readout.
"""
import math
import numpy as np

PI = math.pi


# --------------------------------------------------------------------------- tools
def m_closed(a, s, kc):
    """Order-a light-cone memory symbol, R^-a sin((a-1) phi) / ((a-1) sin phi)."""
    R = math.hypot(s, kc)
    ph = math.atan2(kc, s)
    return R ** (-a) * math.sin((a - 1) * ph) / ((a - 1) * math.sin(ph))


def m_numeric(a, s, kc):
    """(1/Gamma(a)) int_0^inf tau^(a-1) e^{-s tau} sinc(kc tau) dtau, on a log grid."""
    u = np.linspace(-14, 8, 400001)
    t = np.exp(u)
    f = t ** (a - 1) * np.exp(-s * t) * np.sinc(kc * t / PI) * t   # np.sinc(x) = sin(pi x)/(pi x)
    return np.trapezoid(f, u) / math.gamma(a)


def bessel_series(z2, order, terms=160):
    """J0 or J1 via the power series; z2 = (z/2)^2 (array).  Returns J0(z) or J1(z)/(z/2)."""
    out = np.zeros_like(z2)
    t = np.ones_like(z2) / (1.0 if order == 0 else 1.0)
    for n in range(terms):
        if n > 0:
            t = t * (-z2) / (n * (n + order))
        out += t
    return out


def null_leg_path_sum(mu, L, N, n_max):
    """1+1 path sum on the light-cone grid u, v in [0, L].

    G0 = (1/2) theta(u) theta(v); one convolution with G0 is (1/4) times the cumulative
    integral along u and then along v: one order-one accumulation per null leg.
    Returns u grid, partial sums after n_max insertions, and the record of max errors.
    """
    u = np.linspace(0, L, N)
    du = u[1] - u[0]

    def cum(f, axis):
        g = np.zeros_like(f)
        if axis == 0:
            g[1:, :] = np.cumsum(0.5 * (f[1:, :] + f[:-1, :]) * du, axis=0)
        else:
            g[:, 1:] = np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * du, axis=1)
        return g

    UU, VV = np.meshgrid(u, u, indexing="ij")
    target = 0.5 * bessel_series(mu * mu * UU * VV / 4, 0)
    term = 0.5 * np.ones((N, N))
    total = term.copy()
    record = {}
    for n in range(1, n_max + 1):
        term = 0.25 * cum(cum(term, 0), 1)
        total = total + (-mu * mu) ** n * term
        record[n] = np.max(np.abs(total - target))
    return u, total, target, record


# --------------------------------------------------------------------------- section 1
def section_1():
    print("=" * 78)
    print("1. Light-cone memory of order alpha: m_a(s,k) = (1/G(a)) int tau^(a-1) e^{-s tau} sinc(kc tau)")
    print("   closed form R^-a sin((a-1)phi)/((a-1) sin phi), with s = R cos phi, kc = R sin phi")
    print(f"   {'alpha':>6} {'numeric':>10} {'closed':>10}   R^a m_a at phi = 0.2, 0.6, 1.0, 1.4")
    for a in (1.5, 2.0, 2.5, 3.0):
        vals = [m_closed(a, math.cos(p), math.sin(p)) for p in (0.2, 0.6, 1.0, 1.4)]
        print(f"   {a:6.1f} {m_numeric(a, 0.8, 0.6):10.6f} {m_closed(a, 0.8, 0.6):10.6f}   "
              + "  ".join(f"{v:.4f}" for v in vals))
    print("   Lorentz-invariance defect max_phi |R^a m_a(phi) - R^a m_a(0+)| over phi in (0, pi/2]:")
    phis = np.linspace(1e-4, PI / 2, 2001)
    row = []
    for a in (0.0, 0.5, 1.5, 1.9, 2.0, 2.1, 3.0):
        if abs(a - 1) < 1e-12:
            continue
        h = np.sin((a - 1) * phis) / ((a - 1) * np.sin(phis))
        row.append(f"a={a:g}: {np.max(np.abs(h - h[0])):.1e}")
    print("   " + ",  ".join(row))
    print("   -> only alpha = 2 (and trivially alpha = 0) depends on s^2 + c^2 k^2 alone.")
    print(f"   alpha = 2: m_2(0.8, 0.6) = {m_closed(2, 0.8, 0.6):.6f} = 1/(s^2 + c^2 k^2) = {1 / (0.8 ** 2 + 0.6 ** 2):.6f}")


# --------------------------------------------------------------------------- section 2
def section_2():
    print("=" * 78)
    print("2. The alpha = 2 law  T = 2J + (r1/tau_c^2) G * J = 0  on plane waves (retarded symbol)")
    r1, tau_c = 2.0, 1.0
    mu2 = r1 / (2 * tau_c ** 2)
    print(f"   r1 = {r1}, tau_c = {tau_c}: mu^2 = r1/(2 tau_c^2) = {mu2}")
    print(f"   {'k':>5} {'root E (additive form)':>23} {'sqrt(k^2+mu^2)':>15} {'channel phase: E^2':>19}")
    eta = 1e-12
    for k in (0.0, 0.5, 1.0, 2.0, 4.0):
        def F(E, sign):
            s = eta - 1j * E
            return 2 + sign * (r1 / tau_c ** 2) / (s * s + k * k)
        lo, hi = math.sqrt(k * k + 1e-9) + 1e-9, 50.0
        f = lambda E: F(E, +1).real
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        E2_phase = k * k - mu2                        # 2 - kappa/(k^2 - E^2) = 0
        print(f"   {k:5.1f} {0.5 * (lo + hi):23.10f} {math.sqrt(k * k + mu2):15.10f} {E2_phase:19.4f}")
    print("   -> Klein-Gordon dispersion E^2 = k^2 + mu^2 with mu^2 = r1/(2 tau_c^2): real mass for r1 > 0.")
    print("      With the edge paper's channel phase on the memory, E^2 = k^2 - mu^2: a tachyon for k < mu.")


# --------------------------------------------------------------------------- section 3
def section_3():
    print("=" * 78)
    print("3. Path sum in 1+1: sum_n (-mu^2)^n G0^{*(n+1)} by iterated null-leg accumulation")
    print("   against the massive retarded propagator (1/2) J0(mu sqrt(uv)),  mu = 1,  u, v in [0, 12]")
    for N in (601, 1201):
        u, total, target, rec = null_leg_path_sum(1.0, 12.0, N, 60)
        line = "  ".join(f"n={n}: {rec[n]:.2e}" for n in (2, 5, 10, 20, 40, 60))
        print(f"   grid h = {u[1] - u[0]:.3f}:  {line}")
    print("   -> partial sums swing (alternating series) and then converge; the residual error")
    print("      falls by 4 when h halves (trapezoid), so it is discretization, not the series.")

# --------------------------------------------------------------------------- section 3b
def checkerboard(mu, T, eps):
    """Feynman's checkerboard in 1+1 (c = 1): steps of +-eps in x per time step eps.

    The amplitude is 1 for keeping direction and i eps mu for a corner.  The first leg
    moves right.  Returns x, and the amplitudes of arriving moving right (even corners)
    and moving left (odd corners) at time T.
    """
    N = int(round(T / eps))
    off = N + 1
    R = np.zeros(2 * N + 3, complex)
    L = np.zeros(2 * N + 3, complex)
    R[off + 1] = 1.0
    a = 1j * eps * mu
    for _ in range(N - 1):
        Rn = np.zeros_like(R)
        Ln = np.zeros_like(L)
        Rn[1:] = R[:-1] + a * L[:-1]
        Ln[:-1] = L[1:] + a * R[1:]
        R, L = Rn, Ln
    j = np.arange(2 * N + 3) - off
    keep = (np.abs(j) < N) & ((j - N) % 2 == 0)
    return j[keep] * eps, R[keep], L[keep]


def section_3b():
    print("=" * 78)
    print("3b. Feynman's checkerboard: corner-odd amplitude / (i eps mu) vs J0(mu sqrt(t^2 - x^2))")
    print("    (the CAL term (-mu^2)^n G^{*(n+1)} equals the (2n+1)-corner amplitude / (2 i c mu))")
    mu, T = 1.0, 8.0
    prev = None
    for eps in (0.02, 0.01, 0.005, 0.0025):
        x, R, L = checkerboard(mu, T, eps)
        inside = np.abs(x) <= T - 0.5
        ref = bessel_series(mu * mu * (T * T - x[inside] ** 2) / 4, 0)
        err = np.max(np.abs(L[inside] / (1j * eps * mu) - ref))
        ratio = "" if prev is None else f"   ratio {prev / err:.2f}"
        print(f"    eps = {eps:<7g} max |L/(i eps mu) - J0| on |x| <= {T - 0.5:g}: {err:.3e}{ratio}")
        prev = err
    print("    -> converges as eps: the alpha = 2 path sum is the checkerboard's corner-odd sum;")
    print("       n real insertions weigh as 2n corners, (-mu^2)^n = (i mu)^(2n).")


# --------------------------------------------------------------------------- section 4
def section_4():
    print("=" * 78)
    print("4. Path sum in 3+1 by exact descent  G_3(t,r) = -(1/(2 pi r)) d/dr G_1(t,r)")
    print("   inside the cone: tail of the summed series vs -mu J1(mu sqrt(w))/(4 pi sqrt(w)), w = t^2 - r^2")
    mu, L, N = 1.0, 12.0, 1201
    u, total, target, rec = null_leg_path_sum(mu, L, N, 60)
    du = u[1] - u[0]
    tail1 = total - 0.5                                   # remove the n = 0 term (the shell)
    dU = np.gradient(tail1, du, axis=0)
    dV = np.gradient(tail1, du, axis=1)
    for t_target in (4.0, 8.0):
        s_idx = int(round(2 * t_target / du))             # i + j = 2t/du  (t = (u+v)/2)
        errs, refs = [], []
        for i in range(1, s_idx):
            j = s_idx - i
            if j >= N or i >= j - 2:                      # keep r = (v-u)/2 > 0, inside the grid
                continue
            uu, vv = i * du, j * du
            r = 0.5 * (vv - uu)
            if r < 0.25 or uu < 0.25:
                continue
            dr = dV[i, j] - dU[i, j]                       # d/dr = d/dv - d/du
            g3 = -dr / (2 * PI * r)
            w = uu * vv
            z2 = mu * mu * w / 4
            J1 = (mu * math.sqrt(w) / 2) * bessel_series(np.array([z2]), 1)[0]
            ref = -mu * J1 / (4 * PI * math.sqrt(w))
            errs.append(abs(g3 - ref))
            refs.append(abs(ref))
        print(f"   t = {t_target:3.0f}: max |G3 tail - exact| = {max(errs):.2e}   (max |exact| = {max(refs):.3e},"
              f" {len(errs)} points with 0.25 <= r, t - r)")
    print("   -> the shells on the cone, summed with mass insertions, build the J1 tail inside the cone.")


# --------------------------------------------------------------------------- section 5
def section_5():
    print("=" * 78)
    print("5. Positive frequency, composition on the pair, and the nonrelativistic limit")
    rng = np.random.default_rng(5)
    # 5a: composition on the pair (J, dJ/dt) for Klein-Gordon modes
    ks = rng.uniform(0, 5, 200)
    mu = 1.0
    w = np.sqrt(ks ** 2 + mu ** 2)
    def M(t):
        c, s = np.cos(w * t), np.sin(w * t)
        return np.array([[c, s / w], [-w * s, c]])        # (J, dJ/dt) -> (J, dJ/dt)
    t1, t2 = 0.7, 1.9
    A = M(t1 + t2)
    B = np.einsum("ijk,jlk->ilk", M(t2), M(t1))
    print(f"   5a pair propagator: max |M(t1+t2) - M(t2) M(t1)| over 200 modes = {np.max(np.abs(A - B)):.1e}")
    # 5b: nonrelativistic limit of the positive-frequency evolution (hbar = m = 1, mu = c^2)
    N, Lb = 4096, 400.0
    x = (np.arange(N) - N // 2) * (Lb / N)
    k = 2 * PI * np.fft.fftfreq(N, d=Lb / N)
    psi = np.exp(-x ** 2 / 8 + 1j * x)
    psi /= np.linalg.norm(psi)
    phik = np.fft.fft(psi) / math.sqrt(N)
    T = 10.0
    schr = np.exp(-1j * k ** 2 / 2 * T) * phik
    print(f"   5b packet k0 = 1, t = {T:g}: positive-frequency KG (rest phase removed) vs Schrodinger")
    print(f"      {'c':>5} {'||KG - Schrodinger||':>21} {'x c^2':>8} {'norm drift':>11}")
    for c in (2.0, 4.0, 8.0, 16.0, 32.0):
        om = np.sqrt(c ** 4 + (c * k) ** 2) - c ** 2      # hbar omega(k) - m c^2
        kg = np.exp(-1j * om * T) * phik
        err = np.linalg.norm(kg - schr)
        drift = abs(np.linalg.norm(kg) - np.linalg.norm(phik))
        print(f"      {c:5.0f} {err:21.3e} {err * c * c:8.3f} {drift:11.1e}")
    k4 = np.linalg.norm(k ** 4 * phik)
    print(f"      predicted limit of err x c^2:  (t/8) ||k^4 psi_hat|| = {T / 8 * k4:.3f}")
    print("   -> unitary at every c; converges to Feynman's free kernel as 1/c^2.")
    # 5c: the trace readout needs one sign of frequency
    c = 1.0
    om = np.sqrt(1 + k ** 2)
    ap = np.exp(-2 * (k - 1) ** 2).astype(complex)
    am = 0.5 * np.exp(-2 * (k + 0.5) ** 2).astype(complex)
    l2p, l2m, kg = [], [], []
    for t in np.linspace(0, 10, 41):
        Jp = ap * np.exp(-1j * om * t)
        Jm = Jp + am * np.exp(1j * om * t)
        dJm = -1j * om * ap * np.exp(-1j * om * t) + 1j * om * am * np.exp(1j * om * t)
        l2p.append(np.sum(np.abs(Jp) ** 2))
        l2m.append(np.sum(np.abs(Jm) ** 2))
        kg.append((0.5j * np.sum(np.conj(Jm) * dJm - np.conj(dJm) * Jm)).real)
    l2p, l2m, kg = map(np.array, (l2p, l2m, kg))
    print("   5c trace sum_k |J|^2 over t in [0, 10] (each sign of frequency has its own conserved norm):")
    print(f"      positive frequency only : relative spread {np.ptp(l2p) / l2p.mean():.1e}")
    print(f"      mixed (+) and (-)       : relative spread {np.ptp(l2m) / l2m.mean():.1e}")
    print(f"      mixed, Klein-Gordon charge: relative spread {np.ptp(kg) / abs(kg.mean()):.1e}"
          f"  (value {kg.mean():.3f}; the (-) part counts negatively)")


if __name__ == "__main__":
    section_1()
    section_2()
    section_3()
    section_3b()
    section_4()
    section_5()
