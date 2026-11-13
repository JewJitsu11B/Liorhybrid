#!/usr/bin/env python3
"""The bare CAL at alpha -> 1-, iterated as a transducer: each output state becomes the next current.

    J_{n+1} = T_n = alpha J_n - (1 - alpha) ghat(k) sum_{j>=1} w_j J_{n-j},
    w_j = (j^alpha - (j-1)^alpha) / Gamma(alpha + 1)          (order-alpha power-law memory)

No prefactor, no coupling constant r1, no time unit: n counts updates, and times are quoted in
radians of the carrier's own rest rotation.  ghat(k) = exp(-<r^2> k^2 / 6) is the kernel's spatial
profile.  numpy only.

Sections
  A  The exact memory sum W(z) = sum_j w_j z^{-j} (Hurwitz-zeta form) against direct summation.
  B  Modes: roots of z - alpha + (1 - alpha) ghat W(z) = 0.  Rest rotation per update -> sqrt(1-alpha),
     decay per radian -> (pi/4)(1 - alpha): unitary in the limit.  Cross-check against direct iteration.
  C  Dispersion and mass: (phi(0) - phi(k)) / phi(0) -> <r^2> k^2 / 12, i.e. hbar/2m = <r^2>/12 in
     rest-radian units.
  D  Wavepacket evolved by the iterated CAL against free Schrodinger evolution.
  E  The closure Q = sum_k J J^dagger and its Born readout: conserved on one rotation direction,
     oscillating when both directions are present.
"""
import cmath
import math
import time
import numpy as np

PI = math.pi
B2K = [1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6]


# --------------------------------------------------------------------------- memory sum
def hurwitz(s, q, N=30):
    """Hurwitz zeta(s, q) for complex q with Re q > 0, by Euler-Maclaurin."""
    tot = sum((n + q) ** (-s) for n in range(N))
    x = N + q
    tot += x ** (1 - s) / (s - 1) + 0.5 * x ** (-s)
    rising = s
    for k, b in enumerate(B2K, start=1):
        if k > 1:
            rising *= (s + 2 * k - 3) * (s + 2 * k - 2)
        tot += b / math.factorial(2 * k) * rising * x ** (-s - 2 * k + 1)
    return tot


def W(phi, a):
    """sum_{j>=1} w_j e^{-i j phi}, continued analytically to Im phi > 0 (Jonquiere's formula)."""
    s = a + 1
    k0 = (1j * phi) ** (-s)
    rest = (2 * PI) ** (-s) * (cmath.exp(-1j * PI * s / 2) * hurwitz(s, 1 + phi / (2 * PI))
                               + cmath.exp(1j * PI * s / 2) * hurwitz(s, 1 - phi / (2 * PI)))
    return (1 - cmath.exp(-1j * phi)) * (k0 + rest)


def W_series(phi, a, N=20000):
    j = np.arange(1, N + 1, dtype=float)
    w = (j ** a - (j - 1) ** a) / math.gamma(a + 1)
    return np.sum(w * np.exp(-1j * phi * j))


def D(phi, a, g):
    """Characteristic function of the iterated CAL on the mode z = e^{i phi}."""
    return cmath.exp(1j * phi) - a + (1 - a) * g * W(phi, a)


def root(a, g, guess=None):
    """Co-rotating root phi = rotation + i decay (per update)."""
    eps = 1 - a
    z = guess if guess is not None else math.sqrt(eps * g) + 1j * (PI / 4) * eps ** 1.5 * math.sqrt(g)
    for _ in range(100):
        h = 1e-7 * max(abs(z), 1e-6)
        d = (D(z + h, a, g) - D(z - h, a, g)) / (2 * h)
        dz = D(z, a, g) / d
        z -= dz
        if abs(dz) < 1e-15 * max(abs(z), 1e-12):
            break
    return z


def weights(a, n):
    j = np.arange(1, n + 1, dtype=float)
    return (j ** a - (j - 1) ** a) / math.gamma(a + 1)


def iterate(a, gk, J0, N, past=None):
    """Run the CAL transducer for N updates from J_0 (one value per mode).

    past=None: zero past.  past=(c, phi): the carrier has rotated as c e^{i phi n} for every n < 0,
    over the whole infinite past; its memory contribution is added exactly as
    c e^{i phi n} (W(phi) - sum_{j<=n} w_j e^{-i phi j}).
    """
    M = len(J0)
    J = np.zeros((M, N + 1), complex)
    J[:, 0] = J0
    w = weights(a, N + 1)
    eps = 1 - a
    if past is not None:
        c, ph = past
        Wk = np.array([W(x, a) for x in ph])
        j = np.arange(1, N + 1)
        S = np.cumsum(w[None, :N] * np.exp(-1j * ph[:, None] * j[None, :]), axis=1)
    for n in range(N):
        conv = J[:, :n] @ w[n - 1::-1] if n > 0 else np.zeros(M, complex)
        if past is not None:
            Sn = S[:, n - 1] if n > 0 else 0.0
            conv = conv + c * np.exp(1j * ph * n) * (Wk - Sn)
        J[:, n + 1] = a * J[:, n] - eps * gk * conv
    return J


# --------------------------------------------------------------------------- sections
def section_A():
    print("=" * 84)
    print("A. Exact memory sum (Hurwitz-zeta form) against direct summation, |z| > 1 where the sum converges")
    for a in (0.9, 0.99, 1 - 1e-6):
        for phi in (0.3 - 0.02j, 0.05 - 0.005j):
            f, sr = W(phi, a), W_series(phi, a)
            print(f"   alpha={a:.6f}  phi={phi}:  |formula - series| = {abs(f - sr):.1e}   (|W| = {abs(f):.3f})")


def section_B():
    print("=" * 84)
    print("B. Modes of the iterated CAL at k = 0.  Unitary in the limit: decay per radian / (1-alpha) -> pi/4 = 0.7854")
    print(f"   {'1-alpha':>8} {'rotation/update':>16} {'/sqrt(1-alpha)':>15} {'decay/update':>13} {'decay per radian':>17} {'/(1-alpha)':>11}")
    for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        z = root(1 - eps, 1.0)
        print(f"   {eps:8.0e} {z.real:16.8f} {z.real / math.sqrt(eps):15.6f} {z.imag:13.3e} {z.imag / z.real:17.3e} {z.imag / z.real / eps:11.5f}")
    print("   Flat weights (w_j = 1) give z^2 - (1+alpha) z + 1 = 0: both roots exactly on |z| = 1 at k = 0.")
    print("   Cross-check by direct iteration (zero past, J_0 = 1), alpha = 0.999, 24000 updates:")
    a = 0.999
    J = iterate(a, np.array([1.0]), np.array([1.0 + 0j]), 24000)[0].real
    x = J[200:]
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x))
    an = np.fft.ifft(np.where(f > 0, 2 * X, 0))
    n = np.arange(len(x))
    m = slice(len(x) // 10, 9 * len(x) // 10)
    rot = np.polyfit(n[m], np.unwrap(np.angle(an))[m], 1)[0]
    dec = -np.polyfit(n[m], np.log(np.abs(an[m])), 1)[0]
    z = root(a, 1.0)
    print(f"     iteration: rotation {rot:.6f}, decay {dec:.3e};   root: rotation {z.real:.6f}, decay {z.imag:.3e}")


def section_C():
    print("=" * 84)
    print("C. Dispersion and mass at 1-alpha = 1e-5:  (phi(0) - phi(k)) / (phi(0) k^2)  ->  <r^2>/12  (hbar/2m in rest radians)")
    a = 1 - 1e-5
    z0 = root(a, 1.0)
    print(f"   {'<r^2>':>6} {'k':>5} {'measured':>12} {'<r^2>/12':>10} {'ratio':>8}")
    for r2 in (0.06, 0.6, 6.0):
        for k in (0.05, 0.1, 0.2):
            g = math.exp(-r2 * k * k / 6)
            zk = root(a, g, z0)
            c = (z0.real - zk.real) / (z0.real * k * k)
            print(f"   {r2:6.2f} {k:5.2f} {c:12.6f} {r2 / 12:10.6f} {c / (r2 / 12):8.4f}")
    print("   Diffusive admixture: extra decay of mode k over its kinetic rotation, (Im phi(k) - Im phi(0)) / (phi(0) - phi(k))")
    print(f"   {'1-alpha':>8} {'k=0.1':>9} {'k=0.3':>9} {'k=1':>9} {'sqrt(1-alpha)':>14}")
    for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        a = 1 - eps
        z0 = root(a, 1.0)
        row = []
        for k in (0.1, 0.3, 1.0):
            zk = root(a, math.exp(-0.06 * k * k / 6), z0)
            row.append((zk.imag - z0.imag) / (z0.real - zk.real))
        print(f"   {eps:8.0e} " + " ".join(f"{v:9.5f}" for v in row) + f" {math.sqrt(eps):14.5f}")
    print("   -> the edge evolves as exp(-(i + sqrt(1-alpha)) (hbar/2m) k^2 t): Schrodinger plus a diffusion term")
    print("      of relative size sqrt(1-alpha), on top of the uniform (pi/4)(1-alpha) per radian; both vanish at 1-.")


def packet_run(eps, r2, tau_s, psi0_hat, k, keep):
    a = 1 - eps
    gk = np.exp(-r2 * k[keep] ** 2 / 6)
    z0 = root(a, 1.0)
    phiR = z0.real
    zk = np.array([root(a, g, z0) for g in gk])
    N = int(round(12 * tau_s / r2 / phiR))
    c = psi0_hat[keep]
    J = iterate(a, gk, c.astype(complex), N, past=(c, zk.real))   # one rotation direction, whole past
    JN = J[:, -1] * np.exp(-1j * phiR * N)
    tail = J[:, -int(round(2 * 2 * PI / phiR)):]
    nn = np.arange(tail.shape[1])
    cp = cm = 0.0
    for i in range(len(zk)):
        A = np.stack([np.exp(1j * zk[i].real * nn), np.exp(-1j * zk[i].real * nn)], axis=1)
        cc, *_ = np.linalg.lstsq(A, tail[i], rcond=None)
        cp += abs(cc[0]) ** 2
        cm += abs(cc[1]) ** 2
    return JN, zk, phiR, N, cm / cp


def section_D():
    print("=" * 84)
    print("D. Wavepacket evolved by the iterated CAL vs free Schrodinger, Schrodinger time (hbar/2m) t = 0.5")
    print("   packet exp(-x^2/2 + i x) on a box of 40 (128 points); past: the carrier has always rotated one way.")
    Lx, Nx = 40.0, 128
    x = (np.arange(Nx) - Nx // 2) * (Lx / Nx)
    k = 2 * PI * np.fft.fftfreq(Nx, d=Lx / Nx)
    psi0 = np.exp(-x ** 2 / 2 + 1j * x)
    psi0 /= np.linalg.norm(psi0)
    psi0_hat = np.fft.fft(psi0) / math.sqrt(Nx)
    keep = np.abs(psi0_hat) > 1e-9 * np.abs(psi0_hat).max()
    tau_s = 0.5
    schr = np.zeros(Nx, complex)
    schr[keep] = psi0_hat[keep] * np.exp(-1j * tau_s * k[keep] ** 2)
    print(f"   {'1-alpha':>8} {'<r^2>':>6} {'updates':>8} {'|CAL - Schr|':>13} {'|eigen - Schr|':>15} {'norm drift':>11} "
          f"{'pred drift':>11} {'counter-rot':>12}")
    for eps in (1e-2, 1e-3, 1e-4):
        for r2 in (0.24, 0.12, 0.06):
            JN, zk, phiR, N, frac = packet_run(eps, r2, tau_s, psi0_hat, k, keep)
            cal = np.zeros(Nx, complex)
            cal[keep] = JN
            eig = np.zeros(Nx, complex)
            eig[keep] = psi0_hat[keep] * np.exp(1j * (zk.real - phiR) * N)       # exact modes, decay removed
            drift = np.linalg.norm(cal) - 1
            pred = math.exp(-root(1 - eps, 1.0).imag * N) - 1
            def aligned(u, v):
                u = u / np.linalg.norm(u)
                ph = np.vdot(u, v)
                return np.linalg.norm(u * ph / abs(ph) - v)
            print(f"   {eps:8.0e} {r2:6.2f} {N:8d} {aligned(cal, schr):13.3e} {aligned(eig, schr):15.3e} {drift:+11.2e} "
                  f"{pred:+11.2e} {frac:12.1e}")
    print("   |eigen - Schr| is the long-wavelength floor (k^4 terms), falling with the kernel width <r^2>;")
    print("   the norm drift is the predicted (pi/4)(1-alpha) per radian, vanishing as alpha -> 1.")


def section_E():
    print("=" * 84)
    print("E. Closure Q = sum_k J J^dagger for an M2(C) carrier J = psi_a U + psi_b V; Born readout tr(P Q)/tr Q")
    rng = np.random.default_rng(3)
    U = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    V = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    P = np.outer(q[:, 0], q[:, 0].conj())
    Lx, Nx = 40.0, 128
    x = (np.arange(Nx) - Nx // 2) * (Lx / Nx)
    k = 2 * PI * np.fft.fftfreq(Nx, d=Lx / Nx)
    pa = np.fft.fft(np.exp(-x ** 2 / 2 + 1j * x)) / math.sqrt(Nx)
    pb = np.fft.fft(np.exp(-(x - 3) ** 2 / 3 - 1.5j * x)) / math.sqrt(Nx)
    keep = (np.abs(pa) > 1e-9 * np.abs(pa).max()) | (np.abs(pb) > 1e-9 * np.abs(pb).max())
    r2 = 0.12
    print(f"   {'1-alpha':>8} {'start':>22} {'radians':>8} {'tr Q drift per radian':>22} {'tr Q spread':>12} {'max |dp|':>10}")
    for eps in (1e-2, 1e-3, 1e-4):
        a = 1 - eps
        gk = np.exp(-r2 * k[keep] ** 2 / 6)
        z0 = root(a, 1.0)
        zk = np.array([root(a, g, z0) for g in gk])
        N = int(round(60 / z0.real))
        for label in ("one direction (oriented)", "zero past (both dirs)"):
            amps = []
            for ph in (pa, pb):
                c = ph[keep].astype(complex)
                past = (c, zk.real) if label.startswith("one") else None
                amps.append(iterate(a, gk, c, N, past=past))
            A, B = amps
            trQ, pm = [], []
            for n in range(0, N + 1, max(1, N // 400)):
                Jn = A[:, n][:, None, None] * U[None] + B[:, n][:, None, None] * V[None]
                Q = np.einsum("kij,klj->il", Jn, Jn.conj())
                trQ.append(np.trace(Q).real)
                pm.append(np.trace(P @ Q).real / np.trace(Q).real)
            trQ, pm = np.array(trQ), np.array(pm)
            rad = N * z0.real
            slope = (np.log(trQ[-1]) - np.log(trQ[0])) / rad
            print(f"   {eps:8.0e} {label:>22} {rad:8.1f} {slope:22.3e} {np.ptp(trQ) / trQ.mean():12.2e} {np.max(np.abs(pm - pm[0])):10.2e}")
    print("   oriented: tr Q decays at 2 x (pi/4)(1-alpha) per radian -> conserved in the limit, Born readout constant;")
    print("   both directions: the trace swings (cross terms at twice the rotation), as for mixed frequencies at alpha = 2.")


if __name__ == "__main__":
    t0 = time.time()
    section_A()
    section_B()
    section_C()
    section_D()
    section_E()
    print(f"[{time.time() - t0:.1f} s]")
