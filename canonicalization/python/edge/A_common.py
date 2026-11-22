# Agent A: common special-function code (independent implementation, mpmath only).
import mpmath as mp

mp.mp.dps = 50


def w(j, alpha):
    """Memory weight w_j = (j^alpha - (j-1)^alpha)/Gamma(alpha+1)."""
    j = mp.mpf(j)
    return (j ** alpha - (j - 1) ** alpha) / mp.gamma(alpha + 1)


def W_direct(z, alpha, tol_digits=None):
    """Direct summation sum_{j>=1} w_j z^-j, only for |z| > 1."""
    z = mp.mpc(z)
    r = abs(z)
    assert r > 1
    digits = tol_digits or mp.mp.dps
    N = int(mp.ceil((digits + 8) * mp.log(10) / mp.log(r))) + 50
    ga = mp.gamma(alpha + 1)
    s = mp.mpc(0)
    x = 1 / z
    xp = mp.mpc(1)
    prev = mp.mpf(0)
    for j in range(1, N + 1):
        xp *= x
        cur = mp.mpf(j) ** alpha
        s += (cur - prev) * xp
        prev = cur
    return s / ga


def reduce_phi(phi):
    """Shift Re phi into (-pi, pi]; the full k-sum is exactly 2pi-periodic."""
    phi = mp.mpc(phi)
    k = mp.floor((phi.real + mp.pi) / (2 * mp.pi))
    red = phi - 2 * mp.pi * k
    if red.real <= -mp.pi:
        red += 2 * mp.pi
    return red


def W_jonq(phi, alpha):
    """eq. jonquiere with principal powers:
    W = (1 - e^{-i phi}) sum_{k in Z} (i(phi + 2 pi k))^(-alpha-1).
    k = 0 term taken with the principal power; k != 0 terms summed as two
    Hurwitz zeta functions (valid, with principal powers, for |Re phi| < 2 pi)."""
    phi = reduce_phi(phi)
    s = mp.mpf(alpha) + 1
    I = mp.mpc(0, 1)
    t0 = (I * phi) ** (-s)
    u = phi / (2 * mp.pi)
    tp = mp.exp(-I * mp.pi * s / 2) * mp.zeta(s, 1 + u)   # k >= 1
    tm = mp.exp(+I * mp.pi * s / 2) * mp.zeta(s, 1 - u)   # k <= -1
    return (1 - mp.exp(-I * phi)) * (t0 + (2 * mp.pi) ** (-s) * (tp + tm))


def W_jonq_ksum(phi, alpha, K):
    """Same formula, but the k-sum done term by term (principal powers), |k| <= K,
    with an Euler-Maclaurin-free crude tail (integral estimate). Used only to
    confirm the Hurwitz reduction."""
    phi = mp.mpc(phi)
    s = mp.mpf(alpha) + 1
    I = mp.mpc(0, 1)
    tot = mp.mpc(0)
    for k in range(-K, K + 1):
        tot += (I * (phi + 2 * mp.pi * k)) ** (-s)
    return (1 - mp.exp(-I * phi)) * tot


def W_of_z(z, alpha):
    """Continuation as a function of z (principal log): cut on 0 < z < 1."""
    z = mp.mpc(z)
    return W_jonq(-mp.mpc(0, 1) * mp.log(z), alpha)


def F_phi(phi, alpha, ghat):
    return mp.exp(mp.mpc(0, 1) * phi) - alpha + (1 - alpha) * ghat * W_jonq(phi, alpha)


def paper_phi(eps, ghat):
    eps = mp.mpf(eps)
    ghat = mp.mpf(ghat)
    return mp.sqrt(eps * ghat) + mp.mpc(0, 1) * (eps * (1 - ghat) / 2 + mp.pi / 4 * eps ** 1.5 * mp.sqrt(ghat))


def corot_root(eps, ghat, guess=None):
    """Co-rotating root of the exact F, phi with Re phi > 0."""
    eps = mp.mpf(eps)
    ghat = mp.mpf(ghat)
    alpha = 1 - eps
    if guess is None:
        guess = paper_phi(eps, ghat)
    f = lambda p: F_phi(p, alpha, ghat)
    # scale-aware secant start
    x0 = guess
    x1 = guess * (1 + mp.mpf(10) ** (-8))
    root = mp.findroot(f, (x0, x1), solver='secant', tol=mp.mpf(10) ** (-(mp.mp.dps - 12)), maxsteps=200, verify=False)
    return root
