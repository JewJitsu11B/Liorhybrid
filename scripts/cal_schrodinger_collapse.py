#!/usr/bin/env python3
"""Run the CAL field equation near alpha -> 1^- and check that it collapses to Schrodinger.

Field equation (docs/SCHRODINGER_ALPHA_TO_ONE_PLAN.md):

    T = 0:   alpha J = eps (sigma i tau_c)^(-alpha) R W^alpha J,     eps = 1 - alpha,

with W^alpha the causal (left-sided) Weyl integral, a power-law kernel, and the
channel orientation sigma = +-1 from Axiom 3.  Applying the Weyl derivative gives

    alpha (sigma i tau_c)^alpha D^alpha J = Lambda J,                Lambda = eps R.

The limit alpha -> 1^- is taken with Lambda held fixed (the double scaling).

Units: hbar = tau_c = 1, so E_c = 1.  Requires numpy only.

Sections
  A  Factorized kernel, eternal (Weyl) solutions: exact dispersion and wave-packet
     error against Schrodinger as eps -> 0.                          (Theorems 1, 2)
  B  Factorized kernel, switched-on solutions: Mittag-Leffler response, cross-checked
     by time stepping the memory equation (BDF2 convolution quadrature); norm,
     decay above the cap, runaway from above.                         (Theorems 2, 3)
  C  Cone-supported kernel (memory inside the past light cone): exact dispersion,
     effective mass, and the stationary sector.                       (plan section 5)
"""
import math
import numpy as np

PI = math.pi


# --------------------------------------------------------------------------- tools
def ml_response(alpha, beta, z, t):
    """Inverse Laplace transform of s^(alpha-beta) / (s^alpha - z) at times t > 0.

    Equals t^(beta-1) E_{alpha,beta}(z t^alpha).  Computed as the sum of residues at
    the poles on the principal sheet plus the branch-cut (Hankel) integral along the
    negative real axis.  Valid for 0 < alpha < 2 with beta = 1, and alpha > 1 with
    beta = 2 (the cases used here).
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros(t.shape, dtype=complex)
    # poles s = z^(1/alpha) e^(2 pi i j / alpha) with principal argument in (-pi, pi)
    for j in range(-3, 4):
        arg = (np.angle(z) + 2 * PI * j) / alpha
        if -PI < arg < PI:
            s = abs(z) ** (1 / alpha) * np.exp(1j * arg)
            out += s ** (1 - beta) / alpha * np.exp(s * t)
    # cut: (1/2 pi i) int_0^inf e^{-rt} [F(r e^{-i pi}) - F(r e^{+i pi})] dr
    x = np.linspace(math.log(1e-12), math.log(60.0 / t.min()), 8000)
    r = np.exp(x)
    def F(sign):
        num = r ** (alpha - beta) * np.exp(sign * 1j * PI * (alpha - beta))
        den = r ** alpha * np.exp(sign * 1j * PI * alpha) - z
        return num / den
    kern = (F(-1) - F(+1)) / (2j * PI) * r            # dr = r dx
    w = np.exp(-np.outer(t, r)) * kern
    out += np.trapezoid(w, x, axis=1)
    return out


def ml_series(alpha, beta, w, terms=200):
    """Power series of E_{alpha,beta}(w) for moderate |w| (validation only)."""
    acc, term_w = 0j, 1 + 0j
    for k in range(terms):
        if alpha * k + beta > 170:              # gamma overflows; terms are negligible
            break
        acc += term_w / math.gamma(alpha * k + beta)
        term_w *= w
    return acc


def bdf2_cq_weights(alpha, n):
    """Convolution-quadrature weights of (delta(zeta))^alpha, delta = (1-z)+(1-z)^2/2."""
    a = np.empty(n); b = np.empty(n)
    a[0] = b[0] = 1.0
    for j in range(1, n):
        a[j] = a[j - 1] * (j - 1 - alpha) / j
        b[j] = b[j - 1] * (j - 1 - alpha) / j / 3.0
    return (1.5 ** alpha) * np.convolve(a, b)[:n]


def time_step_mode(alpha, sigma, lam, T, h, J0, J1=0.0):
    """Time-step alpha (sigma i)^alpha  ^C D^alpha J = lam J  with full memory.

    Caputo form (switched on at t = 0): u = J - J0 - t J1 has u(0) = 0, and the
    Riemann-Liouville derivative of u is approximated by BDF2 convolution quadrature.
    """
    n = int(round(T / h))
    w = bdf2_cq_weights(alpha, n + 1)
    c = alpha * (sigma * 1j) ** alpha * h ** (-alpha)
    u = np.zeros(n + 1, dtype=complex)
    t = h * np.arange(n + 1)
    P = np.full(n + 1, J0, dtype=complex) + (t * J1 if alpha > 1 else 0.0)
    denom = c * w[0] - lam
    for k in range(1, n + 1):
        hist = np.dot(w[1:k + 1], u[k - 1::-1])
        u[k] = (lam * P[k] - c * hist) / denom
    return t, u + P


def gauss_packet(N=2048, L=400.0, x_width=2.0, k0=1.0):
    x = (np.arange(N) - N // 2) * (L / N)
    k = 2 * PI * np.fft.fftfreq(N, d=L / N)
    psi = np.exp(-x ** 2 / (4 * x_width ** 2) + 1j * k0 * x)
    psi /= np.linalg.norm(psi)
    return x, k, np.fft.fft(psi) / math.sqrt(N)        # unitary FFT normalization


# --------------------------------------------------------------------------- section A
def section_A():
    print("=" * 78)
    print("A. Factorized kernel, eternal (Weyl) solutions  [sigma = -1, m = 1, E_cap = 4]")
    print("   Lambda(k) = (E_cap - K)/E_c, K = k^2/2m.  Stationary energy on Lambda >= 0:")
    print("   E_alpha = -E_c (Lambda/alpha)^(1/alpha);  Schrodinger: E_1 = K - E_cap.")
    m, Ecap = 1.0, 4.0
    x, k, phik = gauss_packet()
    K = k ** 2 / (2 * m)
    lam = Ecap - K
    above = np.sum(np.abs(phik[lam < 0]) ** 2)
    print(f"   packet: k0 = 1, width 2; probability above the cap = {above:.1e} (dropped)")
    phik = np.where(lam >= 0, phik, 0)
    times = np.linspace(0, 40, 81)
    E1 = -lam
    print(f"   {'eps':>8} {'max|E_a-E_1|':>13} {'eps*max L|1+lnL|':>17} {'ratio':>6}"
          f" {'sup_t err':>10} {'phase-aligned':>14} {'/eps':>7} {'norm drift':>11}")
    for eps in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
        a = 1 - eps
        L = np.clip(lam, 0, None)
        Ea = -(L / a) ** (1 / a)
        Lin = np.linspace(1e-9, Ecap, 4000)
        dmax = np.max(np.abs(-(Lin / a) ** (1 / a) + Lin))
        bound = eps * np.max(Lin * np.abs(1 + np.log(Lin)))
        errs, perrs, drift = [], [], 0.0
        for t in times:
            Ja = np.exp(-1j * Ea * t) * phik
            J1 = np.exp(-1j * E1 * t) * phik
            errs.append(np.linalg.norm(Ja - J1))
            ov = np.vdot(J1, Ja)                       # remove the best global phase
            perrs.append(np.linalg.norm(Ja * np.exp(-1j * np.angle(ov)) - J1))
            drift = max(drift, abs(np.linalg.norm(Ja) - np.linalg.norm(phik)))
        e, pe = max(errs), max(perrs)
        print(f"   {eps:8.0e} {dmax:13.3e} {bound:17.3e} {dmax / bound:6.3f} {e:10.3e} {pe:14.3e}"
              f" {pe / eps:7.2f} {drift:11.1e}")
    print("   -> energy error -> eps * max L|1+ln L| (the first-order term; ratio -> 1).")
    print("      The raw packet error is mostly a global phase drift (the constant term);")
    print("      with that removed, the observable error is linear in eps. Norm is exact.")


# --------------------------------------------------------------------------- section B
def section_B():
    print("=" * 78)
    print("B. Factorized kernel, switched on at t = 0 (Caputo form), sigma = -1")
    sigma = -1
    # B1: validate the Mittag-Leffler evaluator against the power series
    worst = 0.0
    for a in (0.9, 0.99, 1.3):
        for zz in (0.7 * np.exp(1j * 0.3), 1.1 * np.exp(1j * 2.6), 0.5j):
            for t in (0.5, 1.5):
                ref = ml_series(a, 1, zz * t ** a)
                got = ml_response(a, 1, zz, t)[0]
                worst = max(worst, abs(ref - got))
    print(f"   B1 Mittag-Leffler evaluator vs power series: max |diff| = {worst:.1e}")

    # B2: time-stepping the memory equation reproduces the exact response
    print("   B2 time stepping (BDF2 convolution quadrature, full history) vs exact:")
    print(f"      {'alpha':>6} {'Lambda':>7} {'h':>7} {'max|J_step - J_exact| (t<=20)':>31}")
    for a, lam in ((0.9, 1.0), (0.99, 1.0), (0.99, -1.0)):
        z = lam / (a * (sigma * 1j) ** a)
        for h in (0.02, 0.01, 0.005):
            t, J = time_step_mode(a, sigma, lam, 20.0, h, 1.0 + 0j)
            sel = slice(int(1.0 / h), None, int(0.5 / h))
            ex = ml_response(a, 1, z, t[sel])
            print(f"      {a:6.2f} {lam:7.1f} {h:7.3f} {np.max(np.abs(J[sel] - ex)):31.2e}")
    print("      -> error halves with h (first order: the t^alpha start-up singularity of a")
    print("         switched-on solution, no correction weights). The stepped field equation")
    print("         converges to the closed form.")

    # B3: wave packet, switched on, error vs Schrodinger and norm
    m, Ecap = 1.0, 4.0
    x, k, phik = gauss_packet()
    lam = Ecap - k ** 2 / (2 * m)
    keep = np.abs(phik) > 1e-10 * np.abs(phik).max()
    times = np.linspace(0.25, 40, 160)
    print("   B3 packet (k0 = 1, below the cap), switched on at t = 0:")
    print(f"      {'eps':>8} {'sup_t ||J_a - J_1||':>20} {'phase-aligned':>14} {'/eps':>7}"
          f" {'norm at t=40':>13} {'1/alpha':>9}")
    for eps in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
        a = 1 - eps
        Ja = np.zeros((len(times), keep.sum()), dtype=complex)
        for i, (lm, ph) in enumerate(zip(lam[keep], phik[keep])):
            z = lm / (a * (sigma * 1j) ** a)
            Ja[:, i] = ml_response(a, 1, z, times) * ph
        J1 = np.exp(1j * np.outer(times, lam[keep])) * phik[keep]
        err = np.linalg.norm(Ja - J1, axis=1).max()
        ph = np.exp(-1j * np.angle(np.sum(np.conj(J1) * Ja, axis=1)))[:, None]
        perr = np.linalg.norm(Ja * ph - J1, axis=1).max()
        nrm = np.linalg.norm(Ja[-1])
        print(f"      {eps:8.0e} {err:20.3e} {perr:14.3e} {perr / eps:7.2f} {nrm:13.6f} {1 / a:9.6f}")
    print("      -> converges to Schrodinger linearly in eps. Switched-on norm tends to")
    print("         1/alpha (stationary amplitude), i.e. 1 + eps: a switch-on effect,")
    print("         absent for eternal solutions (section A).")

    # B4: above the cap: decay from below, runaway from above
    print("   B4 single mode above the cap (Lambda = -1, |E| = 1):")
    print(f"      {'alpha':>7} {'pole Re s':>11} {'pi*|eps|*|E|':>13} {'|J(10)|':>9}"
          f" {'|J(40)|':>9} {'|J(80)|':>9}")
    lam = -1.0
    for a in (0.99, 0.999, 1.0, 1.001, 1.01):
        eps = 1 - a
        z = lam / (a * (sigma * 1j) ** a)
        poles = [abs(z) ** (1 / a) * np.exp(1j * (np.angle(z) + 2 * PI * j) / a)
                 for j in range(-3, 4) if -PI < (np.angle(z) + 2 * PI * j) / a < PI]
        rs = max(p.real for p in poles)
        tt = np.array([10.0, 40.0, 80.0])
        if a == 1.0:
            J = np.exp(z * tt)
        elif a < 1:
            J = ml_response(a, 1, z, tt)
        else:                                   # Caputo order in (1,2): J(0)=1, J'(0)=i Lambda
            J = ml_response(a, 1, z, tt) + 1j * lam * ml_response(a, 2, z, tt)
        print(f"      {a:7.3f} {rs:11.2e} {PI * abs(eps):13.2e} {abs(J[0]):9.4f}"
              f" {abs(J[1]):9.4f} {abs(J[2]):9.4f}")
    print("      -> from below the mode decays at pi*eps*|E|; at alpha = 1 it is unitary;")
    print("         from above it grows at the same rate (runaway).")


# --------------------------------------------------------------------------- section C
def cone_symbol(alpha, sigma, E, kv):
    """(sigma i)^(-alpha) * M(s = -iE, k) for a thin shell at radius v*tau (tau_c = 1)."""
    s = -1j * E
    if kv == 0:
        M = s ** (-alpha)
    elif alpha == 1.0:
        M = np.log((s + 1j * kv) / (s - 1j * kv)) / (2j * kv)
    else:
        M = ((s - 1j * kv) ** (1 - alpha) - (s + 1j * kv) ** (1 - alpha)) / (2j * kv * (alpha - 1))
    return (sigma * 1j) ** (-alpha) * M


def cone_energy(alpha, sigma, E0, kv):
    """Real root E > kv of alpha = E0 * symbol (stationary mode), by bisection."""
    g = lambda E: alpha - E0 * cone_symbol(alpha, sigma, E, kv).real
    lo, hi = kv * (1 + 1e-12) + 1e-14, 1e6
    if g(lo) > 0 or g(hi) < 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def section_C():
    print("=" * 78)
    print("C. Cone-supported kernel: spatial reach v*tau at lag tau (v <= c), sigma = +1")
    print("   Double scaling holds E0 = eps E_c r1 fixed.  At alpha = 1 the exact dispersion is")
    print("   E(k) = hbar k v coth(hbar k v / E0); small k: E0 + hbar^2 k^2 / 2m, m = 3 E0/(2 v^2).")
    E0, v = 1.0, 1.0
    # phase cancellation on E > kv: imaginary part of the symbol vanishes
    worst = 0.0
    for a in (0.7, 0.9, 0.99, 1.0):
        for kv in (0.1, 1.0, 3.0):
            for E in (kv + 0.05, kv + 1.0, kv + 5.0):
                sym = cone_symbol(a, +1, E, kv)
                worst = max(worst, abs(sym.imag) / abs(sym))
    print(f"   C1 |Im symbol|/|symbol| on E > kv, alpha in {{0.7,0.9,0.99,1}}: {worst:.1e}"
          "  (stationary => unitary)")
    print("   C2 alpha = 1: root vs closed form, and vs Schrodinger (m = 3E0/2v^2):")
    print(f"      {'k':>6} {'E root':>10} {'kv coth(kv/E0)':>15} {'E0+k^2/2m':>11}"
          f" {'rel. kinetic dev':>17} {'K/E0':>7} {'v_group/v':>10}")
    m = 3 * E0 / (2 * v ** 2)
    for kk in (0.01, 0.05, 0.1, 0.3, 1.0, 3.0):
        kv = kk * v
        Er = cone_energy(1.0, +1, E0, kv)
        Ec = kv / math.tanh(kv / E0)
        Es = E0 + kk ** 2 / (2 * m)
        dev = ((Er - E0) - kk ** 2 / (2 * m)) / (kk ** 2 / (2 * m))
        y = kv / E0
        vg = 1 / math.tanh(y) - y / math.sinh(y) ** 2
        print(f"      {kk:6.2f} {Er:10.6f} {Ec:15.6f} {Es:11.6f} {dev:17.2e} {(Es - E0) / E0:7.3f}"
              f" {vg:10.4f}")
    print("      -> Schrodinger is the K << E0 limit; corrections ~ K/E0 (nonrelativistic);")
    print("         group velocity stays below v <= c for all k (causal).")
    print("   C3 alpha -> 1^- with the cone kernel (stationary roots, E0 = 1):")
    print(f"      {'eps':>8} {'E(0)':>10} {'(E0/a)^(1/a)':>13} {'m_eff fit':>10}"
          f" {'3E*/((1+a)v^2)':>15} {'max|E_a-E_1| k<=1':>18} {'UV cap kv':>10}")
    ks = np.linspace(0.0, 1.0, 41)
    E1 = np.array([cone_energy(1.0, +1, E0, kk * v) for kk in ks])
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        a = 1 - eps
        Ea = np.array([cone_energy(a, +1, E0, kk * v) for kk in ks])
        Estar = (E0 / a) ** (1 / a)
        kf = 0.02
        Ek = cone_energy(a, +1, E0, kf * v)
        mfit = kf ** 2 / (2 * (Ek - Ea[0]))
        cap = 0.5 * (E0 / (a * eps)) ** (1 / a)
        print(f"      {eps:8.0e} {Ea[0]:10.6f} {Estar:13.6f} {mfit:10.5f}"
              f" {3 * Estar / ((1 + a) * v ** 2):15.5f} {np.nanmax(np.abs(Ea - E1)):18.3e} {cap:10.2e}")
    print("      -> unitary dispersion for every alpha < 1 below a UV cap kv ~ E0/(2 eps)")
    print("         that recedes to infinity; converges to the alpha = 1 law linearly in eps.")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    section_A()
    section_B()
    section_C()
