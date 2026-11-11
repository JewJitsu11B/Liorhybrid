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
  B  Factorized kernel, source-driven solutions: the integral field equation T = S is
     time-stepped with Weyl integration and a zero past (no Caputo data); persistent
     response, decay above the cap, runaway or preacceleration from above. (Thm 2, 3)
  C  The two Markovian regimes: memory rate with a fixed time unit (the past
     disconnects even from below) versus a shrinking one (flat accumulator), and a
     normalizable (narrow, delta-like) kernel, which has no stationary band.  (Sec. 6)
  D  Readout: a single carrier as two tensor factors (JJ^dag is the reduced state,
     concurrence = 1/gamma), the trace readout at any N, and tensor composition.
  Appendix C  Light cone imposed by hand. The framework assigns the Lorentz signature
     to the alpha = 2 closure, so this is NOT the alpha -> 1^- edge; kept for record.
"""
import math
import numpy as np

PI = math.pi


# --------------------------------------------------------------------------- tools
def weyl_int_weights(alpha, n):
    """BDF2 convolution-quadrature weights of the integral of order alpha.

    Generating function (delta(zeta))^(-alpha), delta = (1-z)+(1-z)^2/2, so that
    W^alpha J(t_n) ~ h^alpha * sum_j q_j J_{n-j} for J with zero past.
    """
    a = np.empty(n); b = np.empty(n)
    a[0] = b[0] = 1.0
    for j in range(1, n):
        a[j] = a[j - 1] * (j - 1 + alpha) / j
        b[j] = b[j - 1] * (j - 1 + alpha) / j / 3.0
    return (1.5 ** (-alpha)) * np.convolve(a, b)[:n]


def source(t, Omega, t0=10.0, w=2.0):
    """Gaussian pulse on carrier e^{-i Omega t}; negligible before t = 0."""
    return np.exp(-(t - t0) ** 2 / (2 * w * w)) * np.exp(-1j * Omega * t)


def source_hat(E, Omega, t0=10.0, w=2.0):
    """S^(E) = int S(t) e^{iEt} dt."""
    return math.sqrt(2 * PI) * w * np.exp(-w * w * (E - Omega) ** 2 / 2) * np.exp(1j * (E - Omega) * t0)


def step_field_equation(alpha, sigma, lam, T, h, Omega):
    """Time-step the integral field equation  alpha J - lam e^{-i sigma alpha pi/2} W^alpha J = S.

    Weyl integration with a zero past (J = S = 0 before t = 0), full history kept.
    This is T = S for one spatial mode; no initial-value (Caputo) data enter.
    """
    n = int(round(T / h))
    q = weyl_int_weights(alpha, n + 1)
    t = h * np.arange(n + 1)
    S = source(t, Omega)
    kap = lam * np.exp(-1j * sigma * alpha * PI / 2) * h ** alpha
    J = np.zeros(n + 1, dtype=complex)
    den = alpha - kap * q[0]
    for k in range(n + 1):
        hist = np.dot(q[1:k + 1], J[k - 1::-1]) if k else 0.0
        J[k] = (S[k] + kap * hist) / den
    return t, J


def symbol(E, alpha, sigma):
    """g_alpha(E) = alpha (sigma i)^alpha (-i E)^alpha  (E_c = 1)."""
    return alpha * (sigma * 1j) ** alpha * (-1j * E) ** alpha


def green_pole(lam, alpha, sigma, E0):
    """Complex root of g_alpha(E) = lam near E0 (pole of the response function)."""
    E = complex(E0)
    for _ in range(100):
        f = symbol(E, alpha, sigma) - lam
        d = (symbol(E + 1e-7, alpha, sigma) - symbol(E - 1e-7, alpha, sigma)) / 2e-7
        E -= f / d
    return E


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
    print("B. Factorized kernel, source-driven (Weyl integration, zero past), sigma = -1")
    print("   One spatial mode: alpha J - lam e^{-i sigma alpha pi/2} W^alpha J = S, pulse at t0 = 10.")
    print("   Response function G(E) = (sigma i)^a (-iE)^a / (g_a(E) - lam) is analytic in Im E > 0")
    print("   except at roots of g_a(E) = lam.")
    sigma = -1
    # B1: persistent response below the cap = residue at the real pole
    print("   B1 below the cap (lam = +1): late amplitude vs residue at the real pole E_lam")
    print(f"      {'alpha':>6} {'h':>6} {'mean|J|, t in [80,120]':>23} {'residue prediction':>19}"
          f" {'ratio':>8}")
    for a in (0.9, 0.99, 0.999, 1.0, 1.01):
        lam = 1.0
        El = sigma * (lam / a) ** (1 / a)
        gp = abs((symbol(El + 1e-6, a, sigma) - symbol(El - 1e-6, a, sigma)) / 2e-6)
        pred = abs((sigma * 1j) ** a * (-1j * El) ** a) * abs(source_hat(El, El)) / gp
        for h in ((0.02, 0.01) if a in (0.9, 1.0) else (0.01,)):
            t, J = step_field_equation(a, sigma, lam, 120.0, h, El)
            late = np.abs(J[t > 80]).mean()
            print(f"      {a:6.3f} {h:6.3f} {late:23.5f} {pred:19.5f} {late / pred:8.5f}")
    print("      -> after the pulse the mode oscillates forever at constant amplitude: the")
    print("         stationary (unitary) solution of Theorem 2, excited causally.")

    # B2: above the cap: pole off the real axis
    print("   B2 above the cap (lam = -1): pole of G and measured late-time rate")
    print(f"      {'alpha':>6} {'pole E':>22} {'-Im E (decay)':>14} {'pi*eps*|E|':>11}"
          f" {'measured rate':>14}")
    for a in (0.97, 0.99, 0.999, 1.0, 1.001, 1.01, 1.03):
        lam = -1.0
        Ep = green_pole(lam, a, sigma, -sigma * 1.0)
        t, J = step_field_equation(a, sigma, lam, 120.0, 0.01, Ep.real)
        i1, i2 = np.searchsorted(t, 40.0), np.searchsorted(t, 80.0)
        rate = (math.log(abs(J[i1])) - math.log(abs(J[i2]))) / 40.0
        print(f"      {a:6.3f} {Ep.real:+11.6f}{Ep.imag:+10.6f}i {-Ep.imag:+14.4e}"
              f" {PI * (1 - a) * abs(Ep):+11.4e} {rate:+14.4e}")
    print("      -> from below the pole is in Im E < 0: causal decay at pi*eps*|E|.")
    print("         At alpha = 1 it sits on the real axis. From above it moves to Im E > 0:")
    print("         the causal response runs away.")

    # B3: from above, the bounded alternative is acausal (preacceleration)
    print("   B3 real-axis (bounded) response: amplitude BEFORE the pulse arrives")
    print(f"      {'alpha':>6} {'|J(t0-10)|':>12} {'|J(t0-20)|':>12} {'|J(t0+40)|':>12}")
    E = np.linspace(-40, 40, 160001)
    dE = E[1] - E[0]
    for a in (0.99, 1.01):
        lam = -1.0
        Ep = green_pole(lam, a, sigma, -sigma * 1.0)
        G = (sigma * 1j) ** a * (-1j * E) ** a / (symbol(E, a, sigma) - lam)
        Jhat = G * source_hat(E, Ep.real)
        vals = []
        for tt in (0.0, -10.0, 50.0):                     # t0 - 10, t0 - 20, t0 + 40
            vals.append(abs(np.sum(Jhat * np.exp(-1j * E * tt)) * dE / (2 * PI)))
        print(f"      {a:6.3f} {vals[0]:12.3e} {vals[1]:12.3e} {vals[2]:12.3e}")
    print("      -> from below the bounded solution is causal (nothing before the pulse, only")
    print("         the Gaussian's own tail). From above, staying bounded forces a response")
    print("         that precedes its source: runaway or preacceleration, the ALD dichotomy.")


# --------------------------------------------------------------------------- section C
def section_C():
    print("=" * 78)
    print("C. The two Markovian regimes (Sec. 6)")
    print("   C1 memory rate (1-alpha) tau_c^(-alpha) tau^(alpha-1)/Gamma(alpha) at lag tau")
    print(f"      {'eps':>8} {'fixed tau_c=1: tau=1':>21} {'tau=100':>9}   "
          f"{'tau_c=eps: tau=0.01':>20} {'tau=1':>7} {'tau=100':>8}")
    def rate(eps, tc, tau):
        a = 1 - eps
        return eps * tc ** (-a) * tau ** (a - 1) / math.gamma(a)
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        print(f"      {eps:8.0e} {rate(eps, 1.0, 1.0):21.3e} {rate(eps, 1.0, 100.0):9.3e}   "
              f"{rate(eps, eps, 0.01):20.4f} {rate(eps, eps, 1.0):7.4f} {rate(eps, eps, 100.0):8.4f}")
    print("      -> with tau_c fixed the past disconnects from below (rate ~ eps); with")
    print("         tau_c ~ eps the rate stays ~1 at every lag: a flat accumulator, not a delta.")
    print("   C2 normalizable narrow kernel w = e^{-tau/tau_w}/tau_w, generator kappa = 1 fixed")
    print("      exact root E = -(i/tau_w) + (kappa/alpha)(sigma sin th + i cos th), th = alpha pi/2")
    print(f"      {'eps':>8} {'tau_w':>8} {'Re E':>9} {'Im E':>12} {'-1/tau_w':>10}")
    sigma, kappa = +1, 1.0
    for eps in (1e-2, 1e-3):
        a = 1 - eps
        th = a * PI / 2
        for tw in (1.0, 0.1, 0.01, 0.001):
            lam = kappa * tw / eps                      # eps*lam/tau_w = kappa
            E = -1j / tw * (1 - (eps * lam / a) * np.exp(-1j * sigma * th))
            print(f"      {eps:8.0e} {tw:8.3f} {E.real:+9.4f} {E.imag:+12.4e} {-1 / tw:+10.1e}")
    print("      -> the real part is the Schrodinger generator, but every mode decays at")
    print("         ~1/tau_w; a delta (tau_w -> 0) kills it instantly. Power law: real roots.")
    a = 1 - 1e-3
    print(f"      power law, eps=1e-3: E = {(1 / a) ** (1 / a):.4f} + 0i")


# --------------------------------------------------------------------------- section D
def section_D():
    print("=" * 78)
    print("D. Readout: one carrier as two factors, and composition (Sec. 6.5)")
    rng = np.random.default_rng(3)
    rnd = lambda n: rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    def basis(n):
        U, _ = np.linalg.qr(rnd(n))
        return [np.outer(U[:, k], U[:, k].conj()) for k in range(n)]
    prob = lambda Q, P: np.trace(P @ Q).real / np.trace(Q).real
    sig = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    # D1 one carrier: JJ^dag = Tr_right |J>><<J|, concurrence = 1/gamma
    e1 = e2 = 0.0
    for _ in range(200):
        J = rnd(2)
        rho = np.outer(J.reshape(4), J.reshape(4).conj()).reshape(2, 2, 2, 2)
        e1 = max(e1, np.max(np.abs(np.einsum("ijkj->ik", rho) - J @ J.conj().T)))
        Q = J @ J.conj().T
        x0 = np.trace(Q).real / 2
        xv = np.array([np.trace(Q @ s_).real / 2 for s_ in sig])
        C = 2 * abs(np.linalg.det(J)) / np.trace(Q).real
        e2 = max(e2, abs(C - math.sqrt(1 - (np.linalg.norm(xv) / x0) ** 2)))
    print(f"   D1 one carrier (200 random J): max|Tr_right|J>><<J| - JJ^dag| = {e1:.1e},"
          f"  max|C - 1/gamma| = {e2:.1e}")
    # D2 trace readout = Born for rank-one carriers, any N
    worst = 0.0
    for N in (2, 4, 16, 1024):
        Psi = rng.normal(size=N) + 1j * rng.normal(size=N)
        eta = rng.normal(size=N) + 1j * rng.normal(size=N)
        eta /= np.linalg.norm(eta)
        J = np.outer(Psi, eta.conj())
        Q = J @ J.conj().T
        U, _ = np.linalg.qr(rnd(N))
        pm = np.real(np.einsum("im,ij,jm->m", U.conj(), Q, U)) / np.trace(Q).real
        born = np.abs(U.conj().T @ Psi) ** 2 / np.vdot(Psi, Psi).real
        worst = max(worst, np.max(np.abs(pm - born)))
    print(f"   D2 Tr(P_m JJ^dag)/Tr(JJ^dag) vs |<Phi_m|Psi>|^2/|Psi|^2, N in {{2,4,16,1024}}: max diff = {worst:.1e}")
    # D3 composition: product statistics and marginal consistency
    JA, JB = rnd(2), rnd(2)
    Q = np.kron(JA, JB) @ np.kron(JA, JB).conj().T
    PA, PB = basis(2), basis(2)
    e3 = max(abs(prob(Q, np.kron(a, b)) - prob(JA @ JA.conj().T, a) * prob(JB @ JB.conj().T, b))
             for a in PA for b in PB)
    J = rnd(4)
    Q = J @ J.conj().T
    QA = np.einsum("ijkj->ik", Q.reshape(2, 2, 2, 2))
    e4 = max(abs(prob(Q, np.kron(a, np.eye(2))) - prob(QA, a)) for a in PA)
    print(f"   D3 product J_A(x)J_B: max|p(m,n) - pA pB| = {e3:.1e};  rank-{np.linalg.matrix_rank(J)} J_AB:"
          f" marginal mismatch = {e4:.1e}")
    # D4 rank of J_AB versus A-B entanglement
    phi = np.array([1, 0, 0, 1]) / math.sqrt(2)
    J = np.outer(phi, np.array([1, 0, 0, 0]))
    Q = J @ J.conj().T
    QA = np.einsum("ijkj->ik", Q.reshape(2, 2, 2, 2))
    print(f"   D4 J_AB = Phi+ eta^dag: rank {np.linalg.matrix_rank(J)}, Tr_B Q = diag"
          f"({QA[0, 0].real:.2f}, {QA[1, 1].real:.2f})  (pure, maximally A-B entangled)")
    # D5 intrinsic normalization: det = 0 is not enough above M_2
    import itertools
    def max_minor(J):
        n = J.shape[0]
        return max(abs(np.linalg.det(J[np.ix_(r, c)])) for r in itertools.combinations(range(n), 2)
                   for c in itertools.combinations(range(n), 2))
    for name, J in (("2|Phi+><Phi+|", 2 * np.outer(phi, phi)), ("diag(2,2,0,0)", np.diag([2.0, 2, 0, 0])),
                    ("diag(2,2,2,0)", np.diag([2.0, 2, 2, 0]))):
        ok = np.allclose(2 * J, J @ J.conj().T)
        print(f"   D5 {name:14s} 2J=JJ^dag: {ok}, det = {abs(np.linalg.det(J)):.0f},"
              f" max 2x2 minor = {max_minor(J):.0f}, tr P = {np.trace(J / 2).real:.0f}")


# --------------------------------------------------------------------------- appendix C
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


def section_cone():
    print("=" * 78)
    print("Appendix C (alpha = 2 structure, NOT the edge): light cone imposed by hand, sigma = +1")
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
    section_D()
    section_cone()
