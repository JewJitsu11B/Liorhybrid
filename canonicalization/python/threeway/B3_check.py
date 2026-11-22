# B3: exact-rational check of the three-way identity, standard library only.
# No sympy, numpy or mpmath. Gaussian rationals built on fractions.Fraction.
import random
from fractions import Fraction as F


class G:
    """Gaussian rational a + b i."""
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = F(a)
        self.b = F(b)

    def __add__(s, o): return G(s.a + o.a, s.b + o.b)
    def __sub__(s, o): return G(s.a - o.a, s.b - o.b)
    def __neg__(s): return G(-s.a, -s.b)
    def __mul__(s, o): return G(s.a * o.a - s.b * o.b, s.a * o.b + s.b * o.a)
    def conj(s): return G(s.a, -s.b)
    def scale(s, q): return G(s.a * q, s.b * q)
    def __eq__(s, o): return s.a == o.a and s.b == o.b
    def is_zero(s): return s.a == 0 and s.b == 0
    def __repr__(s): return "(%s + %s i)" % (s.a, s.b)


ZERO, ONE, IU = G(0), G(1), G(0, 1)


# 2x2 matrices as ((m00, m01), (m10, m11)) of G
def mat(a, b, c, d): return ((a, b), (c, d))
def madd(A, B): return tuple(tuple(A[i][j] + B[i][j] for j in range(2)) for i in range(2))
def msub(A, B): return tuple(tuple(A[i][j] - B[i][j] for j in range(2)) for i in range(2))
def mscale(A, g): return tuple(tuple(A[i][j] * g for j in range(2)) for i in range(2))
def mmul(A, B):
    return tuple(tuple(A[i][0] * B[0][j] + A[i][1] * B[1][j] for j in range(2)) for i in range(2))
def dagger(A): return tuple(tuple(A[j][i].conj() for j in range(2)) for i in range(2))
def tr(A): return A[0][0] + A[1][1]
def det(A): return A[0][0] * A[1][1] - A[0][1] * A[1][0]
def comm(A, B): return msub(mmul(A, B), mmul(B, A))
def meq(A, B): return all(A[i][j] == B[i][j] for i in range(2) for j in range(2))


ID = mat(ONE, ZERO, ZERO, ONE)
SX = mat(ZERO, ONE, ONE, ZERO)
SY = mat(ZERO, G(0, -1), G(0, 1), ZERO)
SZ = mat(ONE, ZERO, ZERO, G(-1))
PAULI = (SX, SY, SZ)

# real basis of su(2): E_k = i s_k (anti-Hermitian, traceless)
E = tuple(mscale(s, IU) for s in PAULI)


def su2_coords(M):
    """Real coordinates c with M = sum c_k E_k. Raises if M is not in su(2)."""
    # i(c1 sx + c2 sy + c3 sz) = [[i c3, i c1 + c2], [i c1 - c2, -i c3]]
    c = (M[0][1].b, M[0][1].a, M[0][0].b)
    R = mat(ZERO, ZERO, ZERO, ZERO)
    for k in range(3):
        R = madd(R, mscale(E[k], G(c[k])))
    if not meq(R, M):
        raise ValueError("matrix is not in su(2): %r" % (M,))
    return c


def ad(A):
    """3x3 rational matrix of ad_A in basis E: column k = coords of [A, E_k]."""
    cols = [su2_coords(comm(A, E[k])) for k in range(3)]
    return [[cols[k][j] for k in range(3)] for j in range(3)]


def kappa(A, B):
    a, b = ad(A), ad(B)
    return sum(a[i][j] * b[j][i] for i in range(3) for j in range(3))


def real(g, what):
    if g.b != 0:
        raise ValueError("%s is not real: %r" % (what, g))
    return g.a


def traceless(X):
    return msub(X, mscale(ID, tr(X).scale(F(1, 2))))


def part1_residual(X):
    """det X - (tr X)^2/4 - (1/8) kappa(i X0, i X0), exact."""
    A = mscale(traceless(X), IU)
    t = real(tr(X), "tr X")
    d = real(det(X), "det X")
    return d - t * t / 4 - F(1, 8) * kappa(A, A)


def part3_residuals(X):
    """Three exact residuals for rho = X / tr X. Requires tr X != 0."""
    t = real(tr(X), "tr X")
    rho = mscale(X, G(1 / t))
    # r_k = tr(rho s_k), read off independently of the Killing form
    r = [real(tr(mmul(rho, s)), "r_k") for s in PAULI]
    r2 = sum(v * v for v in r)
    # confirm rho = (1/2)(1 + r.s)
    R = ID
    for k in range(3):
        R = madd(R, mscale(PAULI[k], G(r[k])))
    R = mscale(R, G(F(1, 2)))
    form_ok = meq(R, rho)
    A = mscale(msub(rho, mscale(ID, G(F(1, 2)))), IU)
    res_det = real(det(rho), "det rho") - (1 - r2) / 4
    res_pur = real(tr(mmul(rho, rho)), "tr rho^2") - (1 + r2) / 2
    res_kap = -F(1, 8) * kappa(A, A) - r2 / 4
    return form_ok, res_det, res_pur, res_kap


def herm(t, x):
    X = mscale(ID, G(t))
    for k in range(3):
        X = madd(X, mscale(PAULI[k], G(x[k])))
    return X


def rnd(rng, big=False):
    if big:
        return F(rng.randint(-10**12, 10**12), rng.randint(1, 10**12))
    return F(rng.randint(-50, 50), rng.randint(1, 50))


def main():
    rng = random.Random(20261121)

    print("== Killing form on the basis E_k = i s_k (from tr(ad ad)) ==")
    for j in range(3):
        print("  ", [str(kappa(E[j], E[k])) for k in range(3)])
    print("   ad_E1 =", [[str(v) for v in row] for row in ad(E[0])])

    print("== specific points ==")
    pts = [
        ("X = 1", herm(1, (0, 0, 0))),
        ("X = sz", herm(0, (0, 0, 1))),
        ("X = 1 + sz", herm(1, (0, 0, 1))),
        ("X = 2 - 3 sx + sy", herm(2, (-3, 1, 0))),
    ]
    for name, X in pts:
        A = mscale(traceless(X), IU)
        line = "  %-18s tr=%s det=%s kappa=%s part1_res=%s" % (
            name, real(tr(X), "tr"), real(det(X), "det"), kappa(A, A), part1_residual(X))
        if tr(X).is_zero():
            line += " | part3: tr X = 0, rho undefined (skipped)"
        else:
            ok, a, b, c = part3_residuals(X)
            line += " | part3 form_ok=%s res(det,tr rho^2,kappa)=(%s,%s,%s)" % (ok, a, b, c)
        print(line)

    print("== random exact tests ==")
    classes = [
        ("Hermitian t,x small rationals", 5000,
         lambda: herm(rnd(rng), [rnd(rng) for _ in range(3)])),
        ("Hermitian t,x 12-digit rationals", 5000,
         lambda: herm(rnd(rng, True), [rnd(rng, True) for _ in range(3)])),
        ("X = J J^dagger, J Gaussian rational", 5000, None),
        ("rank one X = v v^dagger", 2000, None),
    ]
    for name, n, gen in classes:
        n1 = bad1 = n3 = bad3 = skipped = 0
        worst1 = F(0)
        worst3 = F(0)
        psd_bad = 0
        for _ in range(n):
            if gen is not None:
                X = gen()
            elif name.startswith("X = J"):
                J = mat(*[G(rnd(rng), rnd(rng)) for _ in range(4)])
                X = mmul(J, dagger(J))
                dj = det(J)
                if not (det(X) == dj * dj.conj() and real(tr(X), "tr") >= 0):
                    psd_bad += 1
            else:
                v0, v1 = G(rnd(rng), rnd(rng)), G(rnd(rng), rnd(rng))
                X = mat(v0 * v0.conj(), v0 * v1.conj(), v1 * v0.conj(), v1 * v1.conj())
                if not det(X).is_zero():
                    psd_bad += 1
            if not meq(X, dagger(X)):
                raise ValueError("generated X not Hermitian")
            res = part1_residual(X)
            n1 += 1
            if res != 0:
                bad1 += 1
                if abs(res) > abs(worst1):
                    worst1 = res
            if tr(X).is_zero():
                skipped += 1
                continue
            ok, a, b, c = part3_residuals(X)
            n3 += 1
            if not ok or a != 0 or b != 0 or c != 0:
                bad3 += 1
                for v in (a, b, c):
                    if abs(v) > abs(worst3):
                        worst3 = v
        print("  %-36s part1: %d tested, %d nonzero (worst %s) | part3: %d tested, %d nonzero "
              "(worst %s), %d skipped tr=0 | side-check failures %d"
              % (name, n1, bad1, worst1, n3, bad3, worst3, skipped, psd_bad))

    print("== negative control: wrong constant 1/4 at X = sz ==")
    X = herm(0, (0, 0, 1))
    A = mscale(traceless(X), IU)
    t = real(tr(X), "tr")
    print("   residual with 1/4 instead of 1/8:", real(det(X), "det") - t * t / 4 - F(1, 4) * kappa(A, A))

    print("== non-Hermitian input: is su(2) membership enforced? ==")
    Xn = mat(G(1), G(2), G(0), G(3))
    try:
        part1_residual(Xn)
        print("   unexpectedly accepted")
    except ValueError as e:
        print("   rejected as expected:", str(e)[:60])


if __name__ == "__main__":
    main()
