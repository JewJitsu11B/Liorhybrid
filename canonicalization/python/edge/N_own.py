"""Agent N: independent implementation (written before opening the author's script).
Usage: python N_own.py <section>   sections: modes orient packet closure readout all
"""
import sys, time
import numpy as np
import mpmath as mp
from math import gamma, pi

mp.mp.dps = 30


# ---------------------------------------------------------------- weights, W, F, roots
def weights(alpha, N):
    """w_1..w_N (index 0 holds w_1)."""
    j = np.arange(1, N + 1, dtype=float)
    out = np.empty(N)
    out[0] = 1.0
    jj = j[1:]
    out[1:] = -jj**alpha * np.expm1(alpha * np.log1p(-1.0 / jj))
    return out / gamma(alpha + 1)


def W_jonq(phi, alpha):
    """eq. jonquiere, k!=0 terms through Hurwitz zeta (mpmath), principal powers."""
    phi = mp.mpmathify(phi)
    s = mp.mpf(alpha) + 1
    a = phi / (2 * mp.pi)
    k0 = (1j * phi) ** (-s)
    pos = (2 * mp.pi) ** (-s) * mp.exp(-1j * mp.pi * s / 2) * mp.zeta(s, 1 + a)
    neg = (2 * mp.pi) ** (-s) * mp.exp(+1j * mp.pi * s / 2) * mp.zeta(s, 1 - a)
    return (1 - mp.exp(-1j * phi)) * (k0 + pos + neg)


def F_phi(phi, alpha, g):
    return mp.exp(1j * phi) - alpha + (1 - alpha) * g * W_jonq(phi, alpha)


def root(alpha, g):
    eps = 1 - alpha
    guess = mp.sqrt(eps * g) + 1j * (0.5 * eps * (1 - g) + mp.pi / 4 * eps**1.5 * mp.sqrt(g))
    r = mp.findroot(lambda p: F_phi(p, alpha, g), guess, tol=1e-24)
    return complex(r)


# ---------------------------------------------------------------- iteration
def iterate(alpha, g, J0, nsteps, past=None, w=None, hist=None):
    """Iterate eq. update for M independent modes.
    g: (M,) spatial factors. J0: (M,) values at n=0.
    past=None: zero past. past=(phi_h (M,), W_h (M,)): one-way past J_n=J0 e^{i phi_h n}, n<0,
      added exactly through Lemma modes (4).
    hist: (L, M) explicit values for n=-L..-1 (zero before that).
    Returns J[n], n=0..nsteps, shape (nsteps+1, M)."""
    g = np.atleast_1d(np.asarray(g, dtype=float))
    J0 = np.atleast_1d(np.asarray(J0, dtype=complex))
    M = J0.size
    L = 0 if hist is None else hist.shape[0]
    tot = L + nsteps + 1
    if w is None:
        w = weights(alpha, tot)
    wr = w[:tot][::-1].copy()          # wr[tot - j] = w_j
    H = np.zeros((tot, M), dtype=complex)
    if L:
        H[:L] = hist
    H[L] = J0
    if past is not None:
        phi_h, W_h = past
        e = np.exp(-1j * phi_h)
        S = np.zeros(M, dtype=complex)
        epow = np.ones(M, dtype=complex)   # e^{-i phi_h n}
    for n in range(nsteps):
        m = L + n                       # row index of J_n
        mem = wr[tot - m:] @ H[:m] if m > 0 else np.zeros(M, dtype=complex)
        if past is not None:
            # S = sum_{j<=n} w_j e^{-i phi_h j}; tail = J0 e^{i phi_h n} (W_h - S)
            mem = mem + J0 / epow * (W_h - S)
            epow = epow * e
            S = S + w[n] * epow         # adds w_{n+1} e^{-i phi_h (n+1)}
        H[m + 1] = alpha * H[m] - (1 - alpha) * g * mem
    return H[L:]


def fit_two(J, phi, n0, n1):
    """LS fit J_n ~ A z^n + B conj(z)^n over n0<=n<n1, z=e^{i phi}. Amplitudes referred to n1-1."""
    n = np.arange(n0, n1) - (n1 - 1)
    zp = np.exp(1j * phi * n)
    zm = np.exp(-1j * np.conj(phi) * n)
    A = np.stack([zp, zm], axis=1)
    c, *_ = np.linalg.lstsq(A, J[n0:n1], rcond=None)
    return c[0], c[1]


def prony2(J):
    """order-2 linear prediction: J_{n+1} = a J_n + b J_{n-1}; return the two roots."""
    A = np.stack([J[1:-1], J[:-2]], axis=1)
    c, *_ = np.linalg.lstsq(A, J[2:], rcond=None)
    return np.roots([1, -c[0], -c[1]])


# ---------------------------------------------------------------- sections
def sec_modes():
    print("=== Lemma W sanity: Jonquiere vs direct sum at |z|>1 ===")
    for alpha in (0.5, 0.999):
        phi = 0.3 - 0.05j              # |z| = e^{0.05} > 1
        z = np.exp(1j * phi)
        N = 4000
        direct = np.sum(weights(alpha, N) * z ** (-np.arange(1, N + 1.0)))
        print(f" alpha={alpha} |direct - jonq| = {abs(direct - complex(W_jonq(phi, alpha))):.2e}")

    print("=== Table modes (roots of F through eq. jonquiere, mpmath) ===")
    for e in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        r = root(1 - e, 1.0)
        print(f" eps={e:.0e} rot={r.real:.8f} /sqrt={r.real/np.sqrt(e):.6f} decay={r.imag:.4e}"
              f" per rad={r.imag/r.real:.4e} /eps={r.imag/r.real/e:.5f}")

    print("=== Table modes caption: direct iteration alpha=0.999, k=0, 24000 updates ===")
    alpha = 0.999
    N = 24000
    r = root(alpha, 1.0)
    print(f" root: rot={r.real:.6f} decay={r.imag:.4e}")
    w = weights(alpha, N + 2)
    # (a) one-way past at real phi_h = Re phi
    ph = np.array([r.real])
    Wh = np.array([complex(W_jonq(r.real, alpha))])
    t0 = time.time()
    J = iterate(alpha, [1.0], [1.0], N, past=(ph, Wh), w=w)[:, 0]
    print(f" (iteration time {time.time()-t0:.1f}s)")
    for lo in (0, N // 2):
        n = np.arange(lo, N + 1)
        rot = np.polyfit(n, np.unwrap(np.angle(J[lo:])), 1)[0]
        dec = -np.polyfit(n, np.log(np.abs(J[lo:])), 1)[0]
        print(f" one-way past, fit n>={lo}: rot={rot:.6f} decay={dec:.4e}")
    rot_e = np.angle(J[N] / J[0]) % (2 * pi)
    print(f" one-way past, endpoints: decay={-np.log(abs(J[N]/J[0]))/N:.4e}")
    # (b) zero past
    J = iterate(alpha, [1.0], [1.0], N, w=w)[:, 0]
    for lo in (0, N // 2):
        n = np.arange(lo, N + 1)
        rot = np.polyfit(n, np.unwrap(np.angle(J[lo:])), 1)[0]
        dec = -np.polyfit(n, np.log(np.abs(J[lo:])), 1)[0]
        print(f" zero past, raw log-fit n>={lo}: rot={rot:.6f} decay={dec:.4e}")
    for lo in (0, N // 2):
        zz = prony2(J[lo:])
        for z in zz:
            print(f" zero past, order-2 linear prediction n>={lo}: rot={np.angle(z):+.6f} decay={-np.log(abs(z)):.4e}")
    Jr = J.real
    # zero crossings of the real solution -> rotation; envelope from peaks
    sgn = np.sign(Jr)
    zc = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    zcx = zc + Jr[zc] / (Jr[zc] - Jr[zc + 1])
    print(f" zero past, zero crossings: rot={pi*(len(zcx)-1)/(zcx[-1]-zcx[0]):.6f} (imag part max {np.abs(J.imag).max():.1e})")
    pk = np.array([i for i in range(1, N) if abs(Jr[i]) >= abs(Jr[i - 1]) and abs(Jr[i]) > abs(Jr[i + 1])])
    for lo in (0, len(pk) // 2):
        print(f" zero past, peak envelope fit (peaks {lo}..): decay={-np.polyfit(pk[lo:], np.log(np.abs(Jr[pk[lo:]])), 1)[0]:.4e}")

    print("=== Lemma modes (4)/(1): exact one-way past, pure mode z0^n continues as z0^n ===")
    for alpha, g in ((0.999, 1.0), (0.99, 0.8), (0.9999, 1.0), (0.5, 1.0)):
        r = root(alpha, g) if alpha > 0.6 else complex(mp.findroot(lambda p: F_phi(p, alpha, g), 0.9 + 0.3j))
        N = 5000
        Wh = np.array([complex(W_jonq(mp.mpc(r), alpha))])
        J = iterate(alpha, [g], [1.0], N, past=(np.array([r]), Wh))[:, 0]
        ex = np.exp(1j * r * np.arange(N + 1))
        print(f" alpha={alpha} ghat={g} phi={r:.8f} |F|={abs(complex(F_phi(mp.mpc(r), alpha, g))):.1e}"
              f" max rel dev over {N} updates = {np.max(np.abs(J/ex - 1)):.2e}")
    print(" control: same but history at real phi_h = Re phi (not an exact mode):")
    alpha = 0.999; r = root(alpha, 1.0); N = 5000
    J = iterate(alpha, [1.0], [1.0], N, past=(np.array([r.real]), np.array([complex(W_jonq(r.real, alpha))])))[:, 0]
    ex = np.exp(1j * r * np.arange(N + 1))
    print(f"  max |J/z0^n - 1| = {np.max(np.abs(J/ex - 1)):.2e}")


def sec_orient():
    print("=== Prop orientation (1): flat weights, one-way past, exact continuation ===")
    for alpha in (0.999, 0.9, 0.5):
        phi = np.arccos((1 + alpha) / 2)
        N = 5000
        z = np.exp(1j * phi)
        w = np.ones(N + 2)
        Wh = np.array([1 / (z - 1)])
        J = iterate(alpha, [1.0], [1.0], N, past=(np.array([phi]), Wh), w=w)[:, 0]
        ex = z ** np.arange(N + 1)
        A, B = fit_two(J, phi, N - 2 * int(2 * pi / phi), N + 1)
        print(f" alpha={alpha}: max|J_n - e^(i phi n)| = {np.max(np.abs(J-ex)):.2e}, counter amp |B/A| = {abs(B/A):.1e}")
    print(" flat weights, zero past (for contrast):")
    alpha = 0.999; phi = np.arccos((1 + alpha) / 2); N = 5000
    J = iterate(alpha, [1.0], [1.0], N, w=np.ones(N + 2))[:, 0]
    A, B = fit_two(J, phi, N - 2 * int(2 * pi / phi), N + 1)
    print(f"  |A|={abs(A):.4f} |B|={abs(B):.4f}")

    print("=== Prop orientation (2),(3),(4) at k=0, power law ===")
    for e in (1e-2, 1e-3, 1e-4):
        alpha = 1 - e
        r = root(alpha, 1.0)
        cyc = int(round(2 * pi / r.real))
        N = int(round(100 / r.real))
        w = weights(alpha, N + 2 * cyc + 5)
        Wh = np.array([complex(W_jonq(r.real, alpha))])
        J = iterate(alpha, [1.0], [1.0], N, past=(np.array([r.real]), Wh), w=w)[:, 0]
        A, B = fit_two(J, r, N + 1 - 2 * cyc, N + 1)
        print(f" eps={e:.0e} N={N}: one-way past  |B/A|={abs(B/A):.2e} |B|^2/(|A|^2+|B|^2)={abs(B)**2/(abs(A)**2+abs(B)**2):.2e}")
        J = iterate(alpha, [1.0], [1.0], N, w=w)[:, 0]
        A, B = fit_two(J, r, N + 1 - 2 * cyc, N + 1)
        print(f"            zero past     |A|={abs(A):.4f} |B|={abs(B):.4f} counter frac={abs(B)**2/(abs(A)**2+abs(B)**2):.3f}")
        hist = np.exp(1j * r.real * np.arange(-2 * cyc, 0))[:, None]
        J = iterate(alpha, [1.0], [1.0], N, w=w, hist=hist)[:, 0]
        A, B = fit_two(J, r, N + 1 - 2 * cyc, N + 1)
        print(f"            two-cycle past |A|={abs(A):.4f} |B|={abs(B):.4f} counter frac={abs(B)**2/(abs(A)**2+abs(B)**2):.3f}")


def packet_modes(k0=1.0, cut=1e-9):
    Lbox, Np = 40.0, 128
    x = (np.arange(Np) - Np // 2) * Lbox / Np
    psi = np.exp(-x**2 / 2 + 1j * k0 * x)
    k = 2 * pi * np.fft.fftfreq(Np, d=Lbox / Np)
    c = np.fft.fft(psi) / np.sqrt(Np)       # unitary normalisation; phases from x-offset irrelevant
    keep = np.abs(c) > cut * np.abs(c).max()
    return k[keep], c[keep]


def packet_row(e, r2, verbose=True):
    alpha = 1 - e
    k, c = packet_modes()
    g = np.exp(-r2 * k**2 / 6)
    phis = np.array([root(alpha, gg) for gg in g])
    r0 = root(alpha, 1.0)
    N = int(round(0.5 / (r2 / 12) / r0.real))
    cyc = int(round(2 * pi / r0.real))
    w = weights(alpha, N + 2 * cyc + 5)
    Wh = np.array([complex(W_jonq(p.real, alpha)) for p in phis])
    t0 = time.time()
    J = iterate(alpha, g, c, N, past=(phis.real.copy(), Wh), w=w)
    JN = J[N]
    t = r0.real * N                               # rest radians
    n0 = np.linalg.norm(c)

    def dist(a, b):                               # global-phase aligned distance / initial norm
        th = np.angle(np.vdot(b, a))
        return np.linalg.norm(a - np.exp(1j * th) * b) / n0

    schr_t = c * np.exp(-1j * (r2 / 12) * k**2 * t)          # time = actual rest radians
    schr_05 = c * np.exp(-1j * 0.5 * k**2)                   # time = exactly 0.5
    exact = c * np.exp(1j * phis * N)
    exact_ph = c * np.exp(1j * phis.real * N)
    lead_ph = c * np.exp(-1j * (1 - np.sqrt(g)) * t)
    out = dict(eps=e, r2=r2, N=N, modes=len(k),
               err=dist(JN, schr_t), err05=dist(JN, schr_05),
               err_normed=dist(JN / np.linalg.norm(JN) * n0, schr_t),
               floor_exact=dist(exact, schr_t), floor_phase=dist(exact_ph, schr_t),
               floor_lead=dist(lead_ph, schr_t),
               cal_vs_exact=dist(JN, exact),
               loss=1 - np.linalg.norm(JN)**2 / n0**2)
    # counter-rotating: per-mode two-mode fit over the last two rest cycles
    AB = np.array([fit_two(J[:, m], phis[m], N + 1 - 2 * cyc, N + 1) for m in range(len(k))])
    a2, b2 = np.sum(np.abs(AB[:, 0])**2), np.sum(np.abs(AB[:, 1])**2)
    out['counter_norm2'] = b2 / (a2 + b2)
    out['counter_amp'] = np.sqrt(b2 / (a2 + b2))
    # predicted loss
    k2 = np.sum(np.abs(c)**2 * k**2) / n0**2
    out['loss_pred'] = 1 - np.exp(-2 * (pi / 4 * e * t + np.sqrt(e) * k2 * (r2 / 12) * t))
    out['loss_exactmodes'] = 1 - np.linalg.norm(exact)**2 / n0**2
    # two-cycle history
    hist = c[None, :] * np.exp(1j * phis.real[None, :] * np.arange(-2 * cyc, 0)[:, None])
    J2 = iterate(alpha, g, c, N, w=w, hist=hist)
    AB = np.array([fit_two(J2[:, m], phis[m], N + 1 - 2 * cyc, N + 1) for m in range(len(k))])
    a2, b2 = np.sum(np.abs(AB[:, 0])**2), np.sum(np.abs(AB[:, 1])**2)
    out['twocycle_counter'] = b2 / (a2 + b2)
    out['secs'] = time.time() - t0
    if verbose:
        print(" " + " ".join(f"{kk}={vv:.3e}" if isinstance(vv, float) else f"{kk}={vv}" for kk, vv in out.items()), flush=True)
    return out


def sec_packet():
    print("=== Table packet ===")
    for e, r2 in ((1e-2, 0.06), (1e-3, 0.06), (1e-4, 0.24), (1e-4, 0.12), (1e-4, 0.06)):
        packet_row(e, r2)


def sec_closure():
    print("=== Table closure (Prop born), 60 radians; my conventions: box/packets as Table packet ===")
    ka, ca = packet_modes(1.0)
    kb, cb = packet_modes(-1.5)
    kall = np.union1d(ka, kb)
    a = np.zeros(len(kall), complex); b = np.zeros(len(kall), complex)
    a[np.searchsorted(kall, ka)] = ca
    b[np.searchsorted(kall, kb)] = cb
    k = kall
    for r2 in (0.06, 0.12):
        for e in (1e-2, 1e-3, 1e-4):
            alpha = 1 - e
            g = np.exp(-r2 * k**2 / 6)
            phis = np.array([root(alpha, gg) for gg in g])
            r0 = root(alpha, 1.0)
            N = int(round(60 / r0.real))
            w = weights(alpha, N + 5)
            Wh = np.array([complex(W_jonq(p.real, alpha)) for p in phis])
            res = {}
            for name, past in (("one-way", (phis.real.copy(), Wh)), ("zero", None)):
                Ja = iterate(alpha, g, a, N, past=past, w=w)
                Jb = iterate(alpha, g, b, N, past=past, w=w)
                aa = np.sum(np.abs(Ja)**2, axis=1); bb = np.sum(np.abs(Jb)**2, axis=1)
                ab = np.sum(Ja * np.conj(Jb), axis=1)
                rows = []
                for seed in range(8):
                    rng = np.random.default_rng(seed)
                    U = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                    V = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                    v = rng.normal(size=2) + 1j * rng.normal(size=2); v /= np.linalg.norm(v)
                    P = np.outer(v, v.conj())
                    f = lambda X: np.trace(X).real
                    UU, VV, UV = U @ U.conj().T, V @ V.conj().T, U @ V.conj().T
                    trQ = aa * f(UU) + bb * f(VV) + 2 * (ab * np.trace(UV)).real
                    trPQ = aa * f(P @ UU) + bb * f(P @ VV) + 2 * (ab * np.trace(P @ UV)).real
                    p = trPQ / trQ
                    t = r0.real * N
                    rows.append((np.log(trQ[0] / trQ[-1]) / t, np.max(np.abs(p - p[0])),
                                 trQ.max() / trQ.min(), (trQ.max() - trQ.min()) / trQ[0]))
                rows = np.array(rows)
                res[name] = rows
            k2a = np.sum(np.abs(a)**2 * k**2) / np.sum(np.abs(a)**2)
            k2b = np.sum(np.abs(b)**2 * k**2) / np.sum(np.abs(b)**2)
            pred = [2 * (pi / 4 * e + np.sqrt(e) * (r2 / 12) * kk) for kk in (k2a, k2b)]
            o, z = res["one-way"], res["zero"]
            print(f" r2={r2} eps={e:.0e} N={N}: one-way trQ drift/rad {o[:,0].min():.2e}..{o[:,0].max():.2e}"
                  f" (Prop born (1) prediction for the two packets {pred[0]:.2e}, {pred[1]:.2e});"
                  f" readout drift {o[:,1].min():.1e}..{o[:,1].max():.1e} |"
                  f" zero past: trQ max/min {z[:,2].min():.2f}..{z[:,2].max():.2f},"
                  f" (max-min)/trQ0 {z[:,3].min():.2f}..{z[:,3].max():.2f},"
                  f" readout drift {z[:,1].min():.1e}..{z[:,1].max():.1e}", flush=True)


def sec_readout():
    print("=== Table readout ===")
    rng = np.random.default_rng(12345)
    cplx = lambda *s: rng.normal(size=s) + 1j * rng.normal(size=s)
    sig = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
    d1 = d2 = d2b = 0
    for _ in range(200):
        J = cplx(2, 2)
        v = J.reshape(4)                               # |J>> = sum J_ij |i>|j>
        rho = np.outer(v, v.conj()).reshape(2, 2, 2, 2)
        red = np.einsum('ijkj->ik', rho)
        Q = J @ J.conj().T
        d1 = max(d1, np.linalg.norm(red - Q))
        C = 2 * abs(np.linalg.det(J)) / np.trace(Q).real
        x0 = np.trace(Q).real / 2
        xv = np.array([np.trace(Q @ s).real / 2 for s in sig])
        d2 = max(d2, abs(C - np.sqrt(1 - xv @ xv / x0**2)))
        vn = v / np.linalg.norm(v)
        yy = np.kron(sig[1], sig[1])
        d2b = max(d2b, abs(C - abs(vn @ yy @ vn)))     # Wootters concurrence |<psi*|sy sy|psi>|
    print(f" row1 ||Tr_right|J>><<J| - JJ^dag|| (200 random J, max) = {d1:.1e}   [paper 1.8e-15]")
    print(f" row2 |C - 1/gamma| max = {d2:.1e}; C vs Wootters concurrence max = {d2b:.1e}   [paper 1.2e-15]")
    d3 = 0
    for N in (2, 4, 16, 1024):
        Psi = cplx(N); eta = cplx(N)
        J = np.outer(Psi, eta.conj())
        Q = J @ J.conj().T
        Phi, _ = np.linalg.qr(cplx(N, N))
        p_tr = np.einsum('im,ij,jm->m', Phi.conj(), Q, Phi).real / np.trace(Q).real
        p_b = np.abs(Phi.conj().T @ Psi)**2 / np.vdot(Psi, Psi).real
        d3 = max(d3, np.max(np.abs(p_tr - p_b)))
        assert abs(p_tr.sum() - 1) < 1e-12
    print(f" row3 trace readout vs Born, N in 2,4,16,1024 (rank-one J): max diff = {d3:.1e}   [paper 3.7e-16]")
    d4 = d5 = 0
    for _ in range(50):
        JA, JB = cplx(2, 2), cplx(2, 2)
        BA, _ = np.linalg.qr(cplx(2, 2)); BB, _ = np.linalg.qr(cplx(2, 2))
        PA = [np.outer(BA[:, m], BA[:, m].conj()) for m in range(2)]
        PB = [np.outer(BB[:, m], BB[:, m].conj()) for m in range(2)]
        QA, QB = JA @ JA.conj().T, JB @ JB.conj().T
        Q = np.kron(JA, JB) @ np.kron(JA, JB).conj().T
        for m in range(2):
            for n in range(2):
                pmn = np.trace(np.kron(PA[m], PB[n]) @ Q).real / np.trace(Q).real
                d4 = max(d4, abs(pmn - np.trace(PA[m] @ QA).real / np.trace(QA).real * np.trace(PB[n] @ QB).real / np.trace(QB).real))
        JAB = cplx(4, 4)
        assert np.linalg.matrix_rank(JAB) == 4
        Q = JAB @ JAB.conj().T
        QAr = np.einsum('ijkj->ik', Q.reshape(2, 2, 2, 2))
        for m in range(2):
            marg = sum(np.trace(np.kron(PA[m], PB[n]) @ Q).real for n in range(2)) / np.trace(Q).real
            d5 = max(d5, abs(marg - np.trace(PA[m] @ QAr).real / np.trace(QAr).real))
    print(f" row4 product carriers |p(m,n)-pA pB| max = {d4:.1e}   [paper 1.1e-16]")
    print(f" row5 rank-4 J_AB marginal mismatch max = {d5:.1e}   [paper 0]")
    Phi = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    eta = cplx(4)
    J = np.outer(Phi, eta.conj())
    Q = J @ J.conj().T; Q = Q / np.trace(Q).real
    print(f" row6 Phi+ eta^dag: rank={np.linalg.matrix_rank(J)}, purity tr Q^2={np.trace(Q@Q).real:.12f},"
          f" ||Tr_B Q - I/2|| = {np.linalg.norm(np.einsum('ijkj->ik', Q.reshape(2,2,2,2)) - np.eye(2)/2):.1e}   [paper 1, I/2]")
    J = np.diag([2.0, 2, 0, 0])
    from itertools import combinations
    mx = max(abs(np.linalg.det(J[np.ix_(r, c)])) for r in combinations(range(4), 2) for c in combinations(range(4), 2))
    print(f" row7 diag(2,2,0,0): det={np.linalg.det(J):.0f}, largest 2x2 minor={mx:.0f}, tr P={np.trace(J/2):.0f},"
          f" ||2J - JJ^dag||={np.linalg.norm(2*J - J@J.T):.0f}   [paper 0, 4, 2]")

    def boost(eta, n):
        ns = sum(ni * s for ni, s in zip(n, sig))
        return np.cosh(eta / 2) * np.eye(2) + np.sinh(eta / 2) * ns

    def conc(J):
        return 2 * abs(np.linalg.det(J)) / np.trace(J @ J.conj().T).real
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    Uq, _ = np.linalg.qr(cplx(2, 2))
    J0 = 1.7 * Uq                                       # carrier at rest: Q = rho I
    for eta in (0.5, 1.0, 2.0):
        print(f" row8 boost eta={eta}: C={conc(boost(eta, n) @ J0):.6f} sech={1/np.cosh(eta):.6f}")
    print(f" row9 boosts 0.7 then 0.9 (collinear): C={conc(boost(0.9, n) @ boost(0.7, n) @ J0):.6f} sech(1.6)={1/np.cosh(1.6):.6f}   [paper 0.387978]")
    n2 = rng.normal(size=3); n2 /= np.linalg.norm(n2)
    print(f"      (non-collinear axes, for contrast: C={conc(boost(0.9, n2) @ boost(0.7, n) @ J0):.6f})")
    Jr = cplx(2, 2)
    print(f"      boost on a carrier NOT at rest (random J, C0={conc(Jr):.4f}): C={conc(boost(1.0, n) @ Jr):.4f}; SU(2) rotation: C={conc(Uq @ Jr):.4f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    secs = dict(modes=sec_modes, orient=sec_orient, packet=sec_packet, closure=sec_closure, readout=sec_readout)
    for name, fn in secs.items():
        if which in (name, "all"):
            fn()
