"""The three-readout relation solved for the radius, and which action keeps which readout.
Labels T4--T7, each in matrix form (M2(C)) and in Clifford form (Cl(3,0), no matrices for elements).

T4  radius:   (tr X)^2/4 - det X = |x|^2 = -(1/8) kappa(iX, iX),   x_j = (1/2) tr(X sigma_j)
              <X>_0^2   - N(X)   = |x|^2 = -(1/8) kappa(IX, IX),   x   = <X>_1
T5  rank one: X = psi psi^dagger (X = J rev(J), J in one minimal left ideal):
              det X = 0 and the Born and Killing readouts coincide, tr X / 2 = |x|
T6  conjugation X -> A X A^{-1} keeps all three readouts (any invertible A, any X)
T7  congruence X -> A X A^dagger keeps det up to |det A|^2 and nothing else:
              a boost of X = 1 gives tr/2 = cosh(eta), |x| = sinh(eta);
              on SU(2) (rotors) congruence is conjugation and all three are kept

Killing forms are computed from ad matrices, tr(ad ad): on gl(2,C) with the basis E_ij, and on the
bivectors of Cl(3,0) with the basis e12, e13, e23. Exit code 0 iff every check passes."""
import sys
import sympy as sp

RES = []


def check(label, ok, text):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + label + " :: " + text)


def zero(e):
    return sp.simplify(sp.expand(e)) == 0


# ---------------------------------------------------------------- matrix side
I = sp.I
one = sp.eye(2)
sig = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -I], [I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
EB = [sp.Matrix(2, 2, lambda r, c, i=i, j=j: 1 if (r, c) == (i, j) else 0) for i in range(2) for j in range(2)]


def ad(M):
    cols = []
    for E in EB:
        C = M * E - E * M
        cols.append([C[i, j] for i in range(2) for j in range(2)])
    return sp.Matrix(4, 4, lambda r, c: cols[c][r])


def kappa(A, B):
    return sp.expand((ad(A) * ad(B)).trace())


t, x1, x2, x3 = sp.symbols('t x1 x2 x3', real=True)
xs = [x1, x2, x3]
r2 = x1 ** 2 + x2 ** 2 + x3 ** 2
X = t * one + sum((xs[j] * sig[j] for j in range(3)), sp.zeros(2, 2))

check("T4 matrix", zero(X.trace() ** 2 / 4 - X.det() - r2) and zero(-kappa(I * X, I * X) / 8 - r2)
      and all(zero((X * sig[j]).trace() / 2 - xs[j]) for j in range(3)),
      "(tr X)^2/4 - det X = |x|^2 = -kappa(iX,iX)/8 ; x_j = tr(X sigma_j)/2")

a, b, c, d = sp.symbols('a b c d', real=True)
psi = sp.Matrix([a + I * b, c + I * d])
P = psi * psi.H
check("T5 matrix", zero(P.det()) and zero((P.trace() / 2) ** 2 + kappa(I * P, I * P) / 8),
      "X = psi psi^dagger: det X = 0 and (tr X/2)^2 = -kappa(iX,iX)/8")

g = sp.symbols('g0:8', real=True)
A = sp.Matrix([[g[0] + I * g[1], g[2] + I * g[3]], [g[4] + I * g[5], g[6] + I * g[7]]])
z = sp.symbols('z0:8', real=True)
Xg = sp.Matrix([[z[0] + I * z[1], z[2] + I * z[3]], [z[4] + I * z[5], z[6] + I * z[7]]])
dA = A.det()
adjA = sp.Matrix([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]])
Yn = A * Xg * adjA                      # = det(A) * A X A^{-1}
check("T6 matrix", zero(Yn.trace() - dA * Xg.trace()) and zero(Yn.det() - dA ** 2 * Xg.det())
      and zero(kappa(I * Yn, I * Yn) - dA ** 2 * kappa(I * Xg, I * Xg)),
      "A X A^{-1} has the trace, determinant and kappa(iX,iX) of X (general complex A, X; cleared of det A)")

Yc = A * X * A.H
eta = sp.symbols('eta', real=True)
Bst = sp.diag(sp.exp(eta / 2), sp.exp(-eta / 2))
Yb = Bst * one * Bst.H
p, q, r, s = sp.symbols('p q r s', real=True)
U = sp.Matrix([[p + I * q, -(r - I * s)], [r + I * s, p - I * q]])
nU = p ** 2 + q ** 2 + r ** 2 + s ** 2
Yu = U * X * U.H
check("T7 matrix",
      zero(Yc.det() - dA * sp.conjugate(dA) * X.det())
      and zero(Bst.det() - 1) and zero(Yb.det() - 1) and zero(Yb.trace() / 2 - sp.cosh(eta).rewrite(sp.exp))
      and zero(-kappa(I * Yb, I * Yb) / 8 - (sp.sinh(eta) ** 2).rewrite(sp.exp))
      and zero(Yu.trace() - nU * X.trace()) and zero(Yu.det() - nU ** 2 * X.det())
      and zero(kappa(I * Yu, I * Yu) - nU ** 2 * kappa(I * X, I * X)),
      "det(A X A^dagger) = |det A|^2 det X ; boost of 1: tr/2 = cosh eta, |x| = sinh eta ; SU(2) keeps all three")

# ---------------------------------------------------------------- Clifford side
BLADES = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]


def blade_mul(u, v):
    w = list(u) + list(v)
    sign = 1
    changed = True
    while changed:
        changed = False
        for i in range(len(w) - 1):
            if w[i] > w[i + 1]:
                w[i], w[i + 1] = w[i + 1], w[i]
                sign = -sign
                changed = True
            elif w[i] == w[i + 1]:
                del w[i:i + 2]
                changed = True
                break
    return sign, tuple(w)


TABLE = {(u, v): blade_mul(u, v) for u in BLADES for v in BLADES}


class MV:
    def __init__(self, cdict=None):
        self.c = {k: sp.expand(v) for k, v in (cdict or {}).items() if sp.expand(v) != 0}

    def __add__(self, o):
        dd = dict(self.c)
        for k, v in o.c.items():
            dd[k] = dd.get(k, 0) + v
        return MV(dd)

    def __sub__(self, o):
        return self + o.scale(-1)

    def scale(self, k):
        return MV({kk: k * v for kk, v in self.c.items()})

    def __mul__(self, o):
        dd = {}
        for u, vu in self.c.items():
            for v, vv in o.c.items():
                sg, w = TABLE[(u, v)]
                dd[w] = dd.get(w, 0) + sg * vu * vv
        return MV(dd)

    def get(self, blade):
        return self.c.get(blade, sp.Integer(0))

    def rev(self):
        return MV({k: (-1) ** (len(k) * (len(k) - 1) // 2) * v for k, v in self.c.items()})

    def conj(self):
        return MV({k: (-1) ** (len(k) * (len(k) + 1) // 2) * v for k, v in self.c.items()})

    def is_zero(self):
        return all(sp.simplify(v) == 0 for v in self.c.values())


def e(*idx):
    return MV({tuple(idx): 1})


ONE, PS = e(), e(1, 2, 3)
BIV = [e(1, 2), e(1, 3), e(2, 3)]


def Nrm(Q):
    return Q * Q.conj()                # central: scalar + pseudoscalar


def kappa_biv(Q, W):
    """tr(ad_Q ad_W) on the bivectors; the scalar and pseudoscalar parts of Q, W are central."""
    def adm(Z):
        cols = []
        for Bk in BIV:
            C = Z * Bk - Bk * Z
            cols.append([C.get((1, 2)), C.get((1, 3)), C.get((2, 3))])
        return sp.Matrix(3, 3, lambda rr, cc: cols[cc][rr])
    return sp.expand((adm(Q) * adm(W)).trace())


Xc = ONE.scale(t) + e(1).scale(x1) + e(2).scale(x2) + e(3).scale(x3)
NX = Nrm(Xc)
IX = PS * Xc
check("T4 Clifford", zero(NX.get((1, 2, 3))) and zero(Xc.get(()) ** 2 - NX.get(()) - r2)
      and zero(-kappa_biv(IX, IX) / 8 - r2)
      and all(zero(Xc.get((j + 1,)) - xs[j]) for j in range(3)),
      "<X>_0^2 - N(X) = |x|^2 = -kappa(IX,IX)/8 ; x = <X>_1")

qs = sp.symbols('q0:8', real=True)
Q = MV({BLADES[i]: qs[i] for i in range(8)})
Jc = Q * (ONE + e(3)).scale(sp.Rational(1, 2))
Xr = Jc * Jc.rev()
NXr = Nrm(Xr)
IXr = PS * Xr
check("T5 Clifford", (Xr - Xr.rev()).is_zero() and zero(NXr.get(())) and zero(NXr.get((1, 2, 3)))
      and zero(Xr.get(()) ** 2 + kappa_biv(IXr, IXr) / 8),
      "X = J rev(J), J = q (1+e3)/2: N(X) = 0 and <X>_0^2 = -kappa(IX,IX)/8")

G = MV({BLADES[i]: g[i] for i in range(8)})
Zg = MV({BLADES[i]: z[i] for i in range(8)})
NG = Nrm(G)
al, be = NG.get(()), NG.get((1, 2, 3))          # N(G) = al + I be, central
Yg = G * Zg * G.conj()                           # = N(G) * G Z G^{-1}
NGm = ONE.scale(al) + PS.scale(be)
# compare with N(G) * (readout of Z): scalar part of Yg, N(Yg), kappa(I Yg, I Yg)
sc_ok = (MV({(): Yg.get(()), (1, 2, 3): Yg.get((1, 2, 3))})
         - NGm * MV({(): Zg.get(()), (1, 2, 3): Zg.get((1, 2, 3))})).is_zero()
n_ok = (Nrm(Yg) - NGm * NGm * Nrm(Zg)).is_zero()
# kappa is a real form on the real Lie algebra of bivectors only when N(G) is real; test rotor-times-scale G
Gr = MV({(): p, (1, 2): q, (1, 3): r, (2, 3): s})
Yr = Gr * Zg * Gr.conj()
k_ok = zero(kappa_biv(Yr, Yr) - nU ** 2 * kappa_biv(Zg, Zg)) and zero(Nrm(Gr).get(()) - nU)
check("T6 Clifford", sc_ok and n_ok and k_ok,
      "G Z G^{-1} keeps the central part and N for any invertible G ; kappa on bivectors for G even (cleared of N(G))")

Yc2 = G * Xc * G.rev()
ch, sh = sp.symbols('ch sh', real=True)          # ch^2 - sh^2 = 1 imposed below
Bc = ONE.scale(ch) + e(3).scale(sh)              # exp(eta e3 / 2) with ch = cosh(eta/2), sh = sinh(eta/2)
Ybc = Bc * Bc.rev()
sub = {ch: sp.cosh(eta / 2), sh: sp.sinh(eta / 2)}
Yrc = Gr * Xc * Gr.rev()
check("T7 Clifford",
      (Nrm(Yc2) - ONE.scale(al ** 2 + be ** 2) * Nrm(Xc)).is_zero()
      and zero(sp.simplify(Ybc.get(()).subs(sub) - sp.cosh(eta))) and zero(sp.simplify(Ybc.get((3,)).subs(sub) - sp.sinh(eta)))
      and zero(sp.simplify(Nrm(Ybc).get(()).subs(sub) - 1))
      and zero(Yrc.get(()) - nU * t) and zero(Nrm(Yrc).get(()) - nU ** 2 * NX.get(()))
      and zero(kappa_biv(PS * Yrc, PS * Yrc) - nU ** 2 * kappa_biv(IX, IX)),
      "N(G X rev G) = |N(G)|^2 N(X) ; boost of 1 = cosh eta + sinh eta e3 ; rotors keep all three")

print("%d of %d pass" % (sum(RES), len(RES)))
sys.exit(0 if all(RES) else 1)
