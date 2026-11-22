"""Native Cl(3,0) checks of the Clifford edition.

Cl(3,0) is implemented as an 8-dimensional real algebra directly from the generator
relations s_j s_k = -s_k s_j (j != k), s_j^2 = +1: the multiplication table on basis
blades is computed below by sorting words.  No 2x2 matrix represents an algebra element
anywhere before the final block REP.  (sympy.Matrix is used only for real linear algebra
on coordinate arrays: ranks, Gram matrices, the 4x4 action on the channel index.)

Centre.  Where the matrix edition has a complex number, the native object is an element of
the centre Zc = span_R{1, I}, I = s1 s2 s3.  Such a value is kept as a multivector whose
grade 1 and grade 2 parts are checked to vanish; Re z = <z>_0, Im z = coefficient of I,
|z|^2 = z z~ (reversion).  N(q) = q qbar is compared as a multivector, never as a complex
number.  The imaginary unit of sympy is not used before the block REP.

Output: one line per check  PASS|FAIL|NOT PORTED <label> <text>.  NOTE lines are findings
that are not checks of a statement of the paper.  Exit status 1 on any FAIL.
"""
import itertools
import random
import sys
from collections import Counter

import sympy as sp
from sympy import Rational, sqrt, symbols
from sympy.polys.matrices import DomainMatrix

# ----------------------------------------------------------------------------------------
# The algebra
# ----------------------------------------------------------------------------------------
BL = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
IDX = {b: i for i, b in enumerate(BL)}
GR = [len(b) for b in BL]


def blade_mul(a, b):
    """Product of two blades from s_j s_k = -s_k s_j (j != k) and s_j s_j = 1 only."""
    w = list(a) + list(b)
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
    return sign, IDX[tuple(w)]


TAB = [[blade_mul(a, b) for b in BL] for a in BL]


class MV:
    __slots__ = ("c",)

    def __init__(self, c):
        self.c = [sp.expand(sp.sympify(v)) for v in c]

    @staticmethod
    def blade(i, v=1):
        c = [0] * 8
        c[i] = v
        return MV(c)

    def __add__(self, o):
        if not isinstance(o, MV):
            o = MV.blade(0, o)
        return MV([a + b for a, b in zip(self.c, o.c)])

    __radd__ = __add__

    def __neg__(self):
        return MV([-a for a in self.c])

    def __sub__(self, o):
        if not isinstance(o, MV):
            o = MV.blade(0, o)
        return MV([a - b for a, b in zip(self.c, o.c)])

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, MV):
            return MV([a * o for a in self.c])
        out = [0] * 8
        for i, a in enumerate(self.c):
            if a == 0:
                continue
            for j, b in enumerate(o.c):
                if b == 0:
                    continue
                sg, k = TAB[i][j]
                out[k] += sg * a * b
        return MV(out)

    def __rmul__(self, o):
        return MV([o * a for a in self.c])

    def __truediv__(self, o):
        return MV([a / o for a in self.c])

    def grades(self, *ks):
        return MV([v if GR[i] in ks else 0 for i, v in enumerate(self.c)])

    def rev(self):       # reversion: sign (-1)^{k(k-1)/2}, flips grades 2, 3
        return MV([(-1) ** (GR[i] * (GR[i] - 1) // 2) * v for i, v in enumerate(self.c)])

    def inv(self):       # grade involution: sign (-1)^k
        return MV([(-1) ** GR[i] * v for i, v in enumerate(self.c)])

    def bar(self):       # Clifford conjugation = reversion o grade involution, flips grades 1, 2
        return self.rev().inv()

    def is_zero(self):
        return all(zero(v) for v in self.c)

    def subs(self, d):
        return MV([v.subs(d) for v in self.c])

    def diff(self, s):
        return MV([sp.diff(v, s) for v in self.c])

    def __repr__(self):
        return " + ".join("(%s) s%s" % (v, "".join(map(str, BL[i]))) for i, v in enumerate(self.c) if v != 0) or "0"


def zero(x):
    x = sp.expand(sp.sympify(x))
    return x == 0 or sp.simplify(x) == 0


ZERO = MV([0] * 8)
ONE = MV.blade(0)
s = [None] + [MV.blade(j) for j in (1, 2, 3)]          # vectors s[1], s[2], s[3]
PS = s[1] * s[2] * s[3]                                 # pseudoscalar I
e = [ONE] + [-(PS * s[j]) for j in (1, 2, 3)]           # channel units e_0 = 1, e_j = -I s_j
E8 = [MV.blade(i) for i in range(8)]


def N(q):
    return q * q.bar()


def comm(a, b):
    return a * b - b * a


def is_central(q):
    return q.grades(1, 2).is_zero()


def is_real(q):
    return q.grades(1, 2, 3).is_zero()


def re_(z):
    return z.c[0]


def im_(z):
    return z.c[7]


def abs2(z):
    """|z|^2 = z z~ for z in the centre; returned as a real expression."""
    assert is_central(z)
    w = z * z.rev()
    assert is_real(w)
    return w.c[0]


def cen(x, y=0):
    """the centre element x + I y"""
    return MV([x, 0, 0, 0, 0, 0, 0, y])


def zcoords(q):
    """coordinates of q over the centre in the basis e_a:  Z_a = <q e_a~>_{0+3}"""
    return [(q * e[a].rev()).grades(0, 3) for a in range(4)]


def gen(prefix):
    return MV(symbols("%s0:8" % prefix, real=True))


RESULTS = []


def check(label, text, ok, tag=""):
    ok = bool(ok)
    RESULTS.append((label, ok))
    print("%s %s %s%s" % ("PASS" if ok else "FAIL", label, ("[%s] " % tag) if tag else "", text), flush=True)


def not_ported(label, reason):
    RESULTS.append((label, None))
    print("NOT PORTED %s %s" % (label, reason), flush=True)


def note(text):
    print("NOTE %s" % text, flush=True)


AI = "algebra-independent"

G = gen("g")
Q = gen("q")
Bq = gen("b")

# ----------------------------------------------------------------------------------------
# ALG: the algebra itself
# ----------------------------------------------------------------------------------------
ok = all((s[j] * s[k] + s[k] * s[j] - (2 if j == k else 0) * ONE).is_zero()
         for j in (1, 2, 3) for k in (1, 2, 3))
check("ALG1", "generator relations s_j s_k + s_k s_j = 2 delta_jk hold in the computed table", ok)
ok = all(((a * b) * c - a * (b * c)).is_zero() for a, b, c in itertools.product(E8, repeat=3))
check("ALG2", "the blade product is associative (all 512 triples)", ok)
ok = all(((a * b).rev() - b.rev() * a.rev()).is_zero() and ((a * b).bar() - b.bar() * a.bar()).is_zero()
         and ((a * b).inv() - a.inv() * b.inv()).is_zero() for a, b in itertools.product(E8, repeat=2))
check("ALG3", "reversion and Clifford conjugation are anti-automorphisms, grade involution an automorphism (64 pairs)", ok)
NQ = N(Q)
ok = is_central(NQ) and (NQ - Q.bar() * Q).is_zero()
check("ALG4", "N(q) = q qbar = qbar q lies in the centre span{1, I} for a general q (8 real symbols)", ok)
ok = (N(Q * Bq) - N(Q) * N(Bq)).is_zero() and (N(Q.rev()) - N(Q).rev()).is_zero()
check("ALG5", "N is multiplicative, N(ab) = N(a) N(b), and N(q~) = N(q)~", ok)
ok = (sum((zcoords(Q)[a] * e[a] for a in range(4)), ZERO) - Q).is_zero()
check("ALG6", "Cl3 is free of rank 4 over the centre on e_a: q = sum_a <q e_a~>_{0+3} e_a", ok)
ok = comm(Q, Bq).grades(0, 3).is_zero()
check("ALG7", "every commutator lies in grades 1+2 (no central part)", ok)

# ----------------------------------------------------------------------------------------
# Carrier
# ----------------------------------------------------------------------------------------
eps = sp.LeviCivita
ok = True
for j in range(1, 4):
    for k in range(1, 4):
        rhs = -sp.KroneckerDelta(j, k) * ONE + sum((eps(j, k, l) * e[l] for l in range(1, 4)), ZERO)
        ok &= (e[j] * e[k] - rhs).is_zero()
check("Q1", "e_j e_k = -delta_jk + eps_jkl e_l  (e_j = -I s_j)", ok)

xr = symbols("x0:4", real=True)
yr = symbols("y0:4", real=True)
Zc = [cen(xr[a], yr[a]) for a in range(4)]
J = sum((Zc[a] * e[a] for a in range(4)), ZERO)
JJ = J * J.rev()                                           # the closure
normZ = sp.expand(sum(abs2(z) for z in Zc))                # sum |Z_a|^2, computed in the centre
sumZ2 = sum((z * z for z in Zc), ZERO)                     # sum Z_a^2, an element of the centre

# --- Born ---------------------------------------------------------------------------------
check("B0", "alpha J - (1-alpha) J' at alpha = 2 equals 2J + J'  (J, J' general elements)",
      ((2 * Q - (1 - 2) * Bq) - (2 * Q + Bq)).is_zero())
check("B1", "<J J~>_0 = sum_a |Z_a|^2", zero(JJ.c[0] - normZ) and zero(normZ - sum(a ** 2 for a in xr + yr)))
check("B2", "N(J) = J Jbar = sum_a Z_a^2, an element of the centre", is_central(N(J)) and (N(J) - sumZ2).is_zero())
check("B3", "J J~ is reversion-even: a paravector (grades 2, 3 vanish)",
      (JJ - JJ.rev()).is_zero() and JJ.grades(2, 3).is_zero())
NJJ = N(JJ)
check("B4", "N(J J~) = |N(J)|^2 = N(J) N(J)~ is a real scalar, and <J J~>_0 is a sum of squares (closed future cone)",
      is_real(NJJ) and zero(NJJ.c[0] - abs2(N(J))) and zero(JJ.c[0] - sum(a ** 2 for a in xr + yr)))

ph = symbols("phi", real=True)
Etrig = cen(sp.cos(ph), sp.sin(ph))
Cs, Sn = symbols("C S", real=True)
Eph = cen(Cs, Sn)                                         # duality rotation, C^2 + S^2 = 1


def zero_circle(q):
    """q = 0 modulo C^2 + S^2 = 1"""
    return all(sp.rem(sp.expand(v), Sn ** 2 + Cs ** 2 - 1, Sn) == 0 for v in q.c)


u = symbols("u0:4", real=True)
Uq = sum((u[a] * e[a] for a in range(4)), ZERO)           # |u| times a rotor
u2 = sum(c ** 2 for c in u)
Uel = Eph * Uq
ok = (Etrig.diff(ph) - PS * Etrig).is_zero() and (Etrig.subs({ph: 0}) - ONE).is_zero()   # e^{I phi} = cos + I sin
ok &= zero_circle(Uel * Uel.rev() - u2 * ONE) and zero_circle(Uel.rev() * Uel - u2 * ONE)
check("B5", "U = e^{I phi}(u.e), e^{I phi} = cos phi + I sin phi (solves E' = I E, E(0) = 1): U U~ = U~ U = |u|^2",
      ok)
JU = J * Uel
check("B6", "J J~ invariant under J -> J U, U U~ = 1  (here (JU)(JU)~ = |u|^2 J J~)",
      zero_circle(JU * JU.rev() - u2 * JJ))
check("B7", "congruence covariance (GJ)(GJ)~ = G (J J~) G~ for a general G",
      ((G * J) * (G * J).rev() - G * JJ * G.rev()).is_zero())

v4 = symbols("p0:4", real=True)
P = sum((v4[a] * e[a] for a in range(4)), ZERO)
zre = symbols("zr0:4", real=True)
zim = symbols("zi0:4", real=True)
zs = [cen(zre[a], zim[a]) for a in range(4)]              # general centre coefficients
Jz = sum((zs[a] * e[a] for a in range(4)), ZERO)
p2 = sum(c ** 2 for c in v4)


def channel_matrix(f):
    """matrix over the centre of the Zc-linear map f on the basis e_a; returns (re, im) parts."""
    cols = [zcoords(f(e[b])) for b in range(4)]
    return (sp.Matrix(4, 4, lambda a, b: cols[b][a].c[0]), sp.Matrix(4, 4, lambda a, b: cols[b][a].c[7]))


def apply_channel(M, z):
    return [sum((M[a, b] * z[b] for b in range(4)), ZERO) for a in range(4)]


Lm, Lim = channel_matrix(lambda q: P * q)
Rm, Rim = channel_matrix(lambda q: q * P)
okL = all((zcoords(P * Jz)[a] - apply_channel(Lm, zs)[a]).is_zero() for a in range(4))
okR = all((zcoords(Jz * P)[a] - apply_channel(Rm, zs)[a]).is_zero() for a in range(4))
check("B8", "left rotor multiplication acts on (Z_a) by a REAL matrix L (no I part), L^T L = |p|^2",
      Lim == sp.zeros(4, 4) and okL and (Lm.T * Lm - p2 * sp.eye(4)).expand() == sp.zeros(4, 4))
check("B9", "right rotor multiplication acts on (Z_a) by a REAL matrix R, R^T R = |p|^2",
      Rim == sp.zeros(4, 4) and okR and (Rm.T * Rm - p2 * sp.eye(4)).expand() == sp.zeros(4, 4))
check("B10", "det L = det R = |p|^4 (> 0, so both land in SO(4) at |p| = 1)",
      zero(Lm.det() - p2 ** 2) and zero(Rm.det() - p2 ** 2))
check("B11", "L and R commute (two commuting rotor factors)", (Lm * Rm - Rm * Lm).expand() == sp.zeros(4, 4))
Lz = apply_channel(Lm, zs)
check("B12", "sum Z_a^2 (a centre element) is preserved by L up to |p|^2, for Z_a in the centre",
      (sum((w * w for w in Lz), ZERO) - p2 * sum((w * w for w in zs), ZERO)).is_zero())

# inverse of a general G:  Gadj = Gbar N(G)~,  G Gadj = Gadj G = |N(G)|^2 =: n2 (real scalar)
Gadj = G.bar() * N(G).rev()
n2 = abs2(N(G))
ok_inv = (G * Gadj - n2 * ONE).is_zero() and (Gadj * G - n2 * ONE).is_zero()
Ev = symbols("E0:4", real=True)
rv_ = symbols("r0:4", real=True)
Epar = Ev[0] * ONE + sum((Ev[j] * s[j] for j in (1, 2, 3)), ZERO)
rpar = rv_[0] * ONE + sum((rv_[j] * s[j] for j in (1, 2, 3)), ZERO)
# E' rho' = (Gadj~ E Gadj)(G rho G~) / n2^2 ; grouped by associativity (ALG2)
lhs = ((Gadj.rev() * Epar) * (Gadj * G)) * (rpar * G.rev())
check("B13", "<E rho>_0 invariant under rho -> G rho G~, E -> G~^-1 E G^-1  (G^-1 = Gbar N(G)~/|N(G)|^2, "
             "E, rho general paravectors; identity cleared of the denominator |N(G)|^4)",
      ok_inv and zero(lhs.c[0] - n2 ** 2 * (Epar * rpar).c[0]))
lhs = (Gadj.rev() * (Gadj * G)) * G.rev()
check("B14", "sum_n E_n = 1 maps to G~^-1 G^-1 = (G G~)^-1  (so it stays 1 only if G G~ = 1)",
      ok_inv and (lhs - n2 ** 2 * ONE).is_zero())

# ----------------------------------------------------------------------------------------
# Lorentz signature
# ----------------------------------------------------------------------------------------
Hm = lambda q: (q + q.rev()) / 2
Am = lambda q: (q - q.rev()) / 2
check("L1", "H(q) = (q + q~)/2 is a projector and equals <q>_0 + <q>_1",
      (Hm(Hm(Q)) - Hm(Q)).is_zero() and (Hm(Q) - Q.grades(0, 1)).is_zero())
check("L2", "A(q) = (q - q~)/2 = <q>_2 + <q>_3 is a projector, H + A = 1, H A = 0",
      (Am(Am(Q)) - Am(Q)).is_zero() and (Hm(Q) + Am(Q) - Q).is_zero() and Hm(Am(Q)).is_zero()
      and (Am(Q) - Q.grades(2, 3)).is_zero())
check("L3", "H commutes with q -> G q G~ (any G)", (Hm(G * Q * G.rev()) - G * Hm(Q) * G.rev()).is_zero())
check("L4", "A commutes with q -> G q G~", (Am(G * Q * G.rev()) - G * Am(Q) * G.rev()).is_zero())
check("L5", "I * (paravector slice) = generator slice: A(I H(q)) = I H(q)",
      (Am(PS * Hm(Q)) - PS * Hm(Q)).is_zero())

t, x, y, z = symbols("t x y z", real=True)
xv = x * s[1] + y * s[2] + z * s[3]
X = t * ONE + xv
Hrank = sp.Matrix([Hm(b).c for b in E8]).rank()
check("L6", "X = t + x is reversion-fixed; the image of H has 4 real dimensions",
      (X - X.rev()).is_zero() and Hrank == 4)
mink = t ** 2 - x ** 2 - y ** 2 - z ** 2
check("L7", "N(X) = t^2 - x^2 - y^2 - z^2 (a real scalar)", (N(X) - mink * ONE).is_zero())
check("L8", "N(G X G~) = |N(G)|^2 N(X), |N(G)|^2 = N(G) N(G)~",
      (N(G * X * G.rev()) - n2 * mink * ONE).is_zero())
check("L9", "G = -1 acts trivially on paravectors (kernel contains +-1)",
      ((-ONE) * X * (-ONE).rev() - X).is_zero())
hq = [PS * e[j] for j in (1, 2, 3)]
ok = all((hq[j] * hq[j] - ONE).is_zero() for j in range(3))
ok &= all((hq[j] * hq[k] + hq[k] * hq[j]).is_zero() for j in range(3) for k in range(3) if j != k)
ok &= all((hq[j] - s[j + 1]).is_zero() for j in range(3))
check("L10", "(I e_j)^2 = +1 and I e_1, I e_2, I e_3 anticommute: they are the generators s_j", ok)
words = [ONE, hq[0], hq[1], hq[2], hq[0] * hq[1], hq[1] * hq[2], hq[2] * hq[0], hq[0] * hq[1] * hq[2]]
check("L11", "the 8 words in I e_j are R-independent and span Cl3 (real dim 8); the isomorphism with M2(C) is REP1-REP2",
      sp.Matrix([w.c for w in words]).rank() == 8)
check("L12", "paravector norm X Xbar = (t + x)(t - x) = t^2 - |x|^2 (Minkowski)",
      (X.bar() - (t * ONE - xv)).is_zero() and (X * (t * ONE - xv) - mink * ONE).is_zero())

tp = symbols("t", positive=True)
Xp = X.subs({t: tp})
rhoh = Xp / (2 * Xp.c[0])
vv = symbols("v", real=True)
v2expr = (x ** 2 + y ** 2 + z ** 2) / tp ** 2
check("L13", "2<rho^2>_0 = 1 - 2 N(rho) at 2<rho>_0 = 1",
      zero(2 * rhoh.c[0] - 1) and is_real(N(rhoh)) and zero(2 * (rhoh * rhoh).c[0] - (1 - 2 * N(rhoh).c[0])))
check("L14", "2<rho^2>_0 = (1 + v^2)/2, v = |x|/t", zero(2 * (rhoh * rhoh).c[0] - (1 + v2expr) / 2))
gam = 1 / sqrt(1 - vv ** 2)
check("L15", "1 - (1 + v^2)/2 = 1/(2 gamma^2)", zero(sp.simplify(1 - (1 + vv ** 2) / 2 - 1 / (2 * gam ** 2))), AI)

imgs = [Uq * s[j] * Uq.rev() for j in (1, 2, 3)]
Rot = sp.Matrix(3, 3, lambda i_, j_: (s[i_ + 1] * imgs[j_]).c[0])
check("L16", "rotor conjugation x -> R x R~ keeps vectors vectors and acts by a 3x3 R with R^T R = 1, det R = +1 (|u| = 1)",
      all((q - q.grades(1)).is_zero() for q in imgs)
      and (Rot.T * Rot - u2 ** 2 * sp.eye(3)).expand() == sp.zeros(3, 3) and zero(Rot.det() - u2 ** 3))
boost = Rational(5, 4) * ONE + Rational(3, 4) * s[3]     # exp(ln 2 * s_3)
bX = boost * X * boost.rev()
check("L17", "a boost (paravector with N = 1, here 5/4 + 3/4 s_3) preserves N(X) but changes <X>_0",
      (N(boost) - ONE).is_zero() and (boost - boost.rev()).is_zero()
      and (N(bX) - N(X)).is_zero() and not zero(bX.c[0] - X.c[0]))

# ----------------------------------------------------------------------------------------
# Purity from the channels
# ----------------------------------------------------------------------------------------
tJ = JJ.c[0]
absN2 = abs2(N(J))
abssum2 = abs2(sumZ2)
check("P1", "v^2 = 1 - 4 N(rho) = 1 - |sum Z^2|^2/(sum |Z|^2)^2, rho = J J~/(2<J J~>_0)  (cross-multiplied)",
      is_real(NJJ) and zero(NJJ.c[0] * normZ ** 2 - abssum2 * tJ ** 2))
lag = 4 * sum(im_(Zc[a] * Zc[b].rev()) ** 2 for a in range(4) for b in range(a + 1, 4))
check("P2", "(sum |Z|^2)^2 - |sum Z^2|^2 = 4 sum_{a<b} Im(Z_a Z_b~)^2  (Lagrange, in the centre)",
      zero(normZ ** 2 - abssum2 - lag))
wr, wi = symbols("wr wi", real=True)
w = cen(wr, wi)
check("P3", "|w|^2 - Re(w^2) = 2 (Im w)^2 for w in the centre", zero(abs2(w) - re_(w * w) - 2 * im_(w) ** 2))
realsub = {yr[a]: 0 for a in range(4)}
Jre = J.subs(realsub)
check("P4", "real Z_a: J is even, J J~ = J Jbar = sum Z_a^2 (a scalar), so rho = 1/2 and v = 0",
      (Jre - Jre.grades(0, 2)).is_zero() and (JJ.subs(realsub) - sum(c ** 2 for c in xr) * ONE).is_zero()
      and (N(Jre) - JJ.subs(realsub)).is_zero() and zero(lag.subs(realsub)))
vecJ = JJ.grades(1)
vsq_t2 = (vecJ * vecJ)                                     # |x|^2 of the closure paravector t + x
check("P5", "v = 1 <=> N(J) = 0: with v = |x|/t the Bloch radius of the closure, (1 - v^2) <J J~>_0^2 = |N(J)|^2",
      is_real(vsq_t2) and zero(tJ ** 2 - vsq_t2.c[0] - absN2))
lam_r, lam_i = symbols("lr li", real=True)
lam = cen(lam_r, lam_i)
Js = lam * J
check("P6", "v invariant under a common central rescaling Z_a -> lambda Z_a, lambda in the centre",
      zero(abs2(N(Js)) * tJ ** 2 - absN2 * (Js * Js.rev()).c[0] ** 2))

# ----------------------------------------------------------------------------------------
# Generator slice and Killing forms (all from tr(ad ad) on a basis)
# ----------------------------------------------------------------------------------------
def kapC(A, B):
    """Killing form of Cl3 as a Lie algebra over its centre (rank 4, basis e_a):
    trace over the centre of ad_A ad_B.  Value: an element of the centre."""
    tot = ZERO
    for a in range(4):
        tot = tot + zcoords(comm(A, comm(B, e[a])))[a]
    return tot


def kapR8(A, B):
    """real Killing form of Cl3 (8 real dimensions): real trace of ad_A ad_B on the blades."""
    return sp.expand(sum(comm(A, comm(B, E8[i])).c[i] for i in range(8)))


V6 = [e[1], e[2], e[3], s[1], s[2], s[3]]                  # rotations (bivectors), boosts (vectors)


def coords6(q):
    c = [(q * e[j].rev()).c[0] for j in (1, 2, 3)] + [(q * s[j]).c[0] for j in (1, 2, 3)]
    assert (q - sum((c[i] * V6[i] for i in range(6)), ZERO)).is_zero(), "left grades 1+2"
    return c


def kapR6(A, B):
    """real Killing form of the Lie algebra grades 1+2 (6 real dimensions)."""
    return sp.expand(sum(coords6(comm(A, comm(B, V6[i])))[i] for i in range(6)))


def bivcoords(q):
    c = [(q * e[j].rev()).c[0] for j in (1, 2, 3)]
    assert (q - sum((c[i] * e[i + 1] for i in range(3)), ZERO)).is_zero(), "left the bivectors"
    return c


def kapBiv(A, B):
    """Killing form of the bivector Lie algebra (3 real dimensions)."""
    return sp.expand(sum(bivcoords(comm(A, comm(B, e[j])))[j - 1] for j in (1, 2, 3)))


c0, c1, c2, c3 = symbols("c0:4", real=True)
Agen = c0 * PS + c1 * e[1] + c2 * e[2] + c3 * e[3]
odd_rank = sp.Matrix([Am(b).c for b in E8]).rank()
check("K1", "A = y0 I + y_j e_j is reversion-odd; the reversion-odd elements are exactly grades 3+2 (4 real dims)",
      (Agen + Agen.rev()).is_zero() and (Agen - Agen.grades(2, 3)).is_zero() and odd_rank == 4)
check("K2", "N(A) = -(y0)^2 + |y|^2 (a real scalar)",
      (N(Agen) - (-c0 ** 2 + c1 ** 2 + c2 ** 2 + c3 ** 2) * ONE).is_zero())
check("K3", "N(I X) = -N(X) for a paravector X (also for every q: N(I q) = -N(q))",
      (N(PS * X) + N(X)).is_zero() and (N(PS * Q) + N(Q)).is_zero())

Qp, Bp = Q.grades(1, 2), Bq.grades(1, 2)
kQB = kapC(Q, Bq)
check("K4", "kappa(A,B) = tr_Zc(ad_A ad_B) = 8(<A'B'>_0 + <A'B'>_3), A' = <A>_1 + <A>_2 "
            "(Cl3 as a Lie algebra over its centre; value in the centre)",
      is_central(kQB) and (kQB - 8 * (Qp * Bp).grades(0, 3)).is_zero())
check("K5", "the centre span{1, I} is in the radical of kappa (degenerate on the generator slice)",
      kapC(PS, Bq).is_zero() and kapC(ONE, Bq).is_zero())
Asu = Agen.subs({c0: 0})
ysq = c1 ** 2 + c2 ** 2 + c3 ** 2
kA = kapC(Asu, Asu)
check("K6", "on bivectors: N(A) = |y|^2 = -A^2 = -(1/8) kappa(A,A), kappa from tr(ad ad) both over the centre on Cl3 "
            "and on the 3-dim bivector Lie algebra",
      (N(Asu) - ysq * ONE).is_zero() and (Asu * Asu + ysq * ONE).is_zero()
      and (kA + 8 * ysq * ONE).is_zero() and zero(kapBiv(Asu, Asu) + 8 * ysq))
NG = N(G)
Gb = G.bar()
ok = True
for a_, b_ in itertools.product(E8, repeat=2):
    ok &= (kapC(G * a_ * Gb, G * b_ * Gb) - NG * NG * kapC(a_, b_)).is_zero()
random.seed(7)
rq = lambda: Rational(random.randint(-9, 9), random.randint(1, 9))
for _ in range(3):
    Gn = MV([rq() for _ in range(8)])
    zN = N(Gn)
    Gi = Gn.bar() * (zN.rev() / abs2(zN))
    ok &= (Gn * Gi - ONE).is_zero() and (kapC(Gn * Q * Gi, Gn * Bq * Gi) - kQB).is_zero()
check("K7", "kappa is Ad-invariant: kappa(G A Gbar, G B Gbar) = N(G)^2 kappa(A,B) on all 64 blade pairs, general G "
            "(kappa is bilinear over the centre, G^-1 = Gbar/N(G)); plus 3 exact random G with general A, B",
      ok)

# uniqueness of the invariant form on the bivector Lie algebra, structure constants computed natively
fs = [[bivcoords(comm(e[i_], e[j_])) for j_ in (1, 2, 3)] for i_ in (1, 2, 3)]   # [e_i, e_j] = f_ijl e_l
ok_f = all(fs[i_][j_][l] == 2 * eps(i_ + 1, j_ + 1, l + 1) for i_ in range(3) for j_ in range(3) for l in range(3))
bsym = sp.symbols("b0:9")
Bf = sp.Matrix(3, 3, bsym)
eqs = []
for i_, j_, k_ in itertools.product(range(3), repeat=3):
    eqs.append(sum(fs[i_][j_][l] * Bf[l, k_] for l in range(3)) + sum(fs[i_][k_][l] * Bf[j_, l] for l in range(3)))
sol = sp.linsolve(eqs, bsym)
free = set().union(*[s_.free_symbols for s_ in list(sol)[0]])
check("K8", "ad-invariant bilinear forms on the bivector Lie algebra ([e_i,e_j] = 2 eps_ijl e_l) are 1-dimensional",
      ok_f and len(free) == 1)

K6 = sp.Matrix(6, 6, lambda i_, j_: kapR6(V6[i_], V6[j_]))
K6_claim = sp.Matrix(6, 6, lambda i_, j_: 16 * (V6[i_] * V6[j_]).c[0])
K6_2re = sp.Matrix(6, 6, lambda i_, j_: 2 * re_(kapC(V6[i_], V6[j_])))
Q12, B12 = Q.grades(1, 2), Bq.grades(1, 2)
ok = K6 == K6_claim and K6 == K6_2re and zero(kapR6(Q12, B12) - 16 * (Q12 * B12).c[0])
ok &= zero(kapR8(Q, Bq) - 16 * (Qp * Bp).c[0])
check("K9", "real Killing form of grades 1+2 (6-dim, tr(ad ad)) = 16<AB>_0 = 2 Re kappa; the real Killing form of all "
            "of Cl3 (8-dim) is 16<A'B'>_0", ok)
ev = K6.eigenvals()
npos = sum(m for val, m in ev.items() if val > 0)
nneg = sum(m for val, m in ev.items() if val < 0)
check("K10", "its signature is (3,3): bivectors (rotations) negative, vectors (boosts) positive",
      (npos, nneg) == (3, 3) and all(K6[i_, i_] < 0 for i_ in range(3)) and all(K6[i_, i_] > 0 for i_ in range(3, 6)))

# ----------------------------------------------------------------------------------------
# Bell / CHSH: Cl3 (x)_R Cl3 (64 real dims) and its quotient Cl3 (x)_Zc Cl3 (32 real dims)
# ----------------------------------------------------------------------------------------
class TT:
    """element of Cl3 (x)_R Cl3: dict {(i, j): coeff} on blade_i (x) blade_j"""

    def __init__(self, d=None):
        self.d = {}
        for k, v in (d or {}).items():
            v = sp.expand(v)
            if v != 0:
                self.d[k] = v

    def __add__(self, o):
        d = dict(self.d)
        for k, v in o.d.items():
            d[k] = d.get(k, 0) + v
        return TT(d)

    def __sub__(self, o):
        return self + o * (-1)

    def __mul__(self, o):
        if not isinstance(o, TT):
            return TT({k: v * o for k, v in self.d.items()})
        d = {}
        for (i, j), a in self.d.items():
            for (k, l), b in o.d.items():
                s1, m = TAB[i][k]
                s2, n = TAB[j][l]
                d[(m, n)] = d.get((m, n), 0) + s1 * s2 * a * b
        return TT(d)

    def rev(self):
        sg = lambda i: (-1) ** (GR[i] * (GR[i] - 1) // 2)
        return TT({(i, j): sg(i) * sg(j) * v for (i, j), v in self.d.items()})

    def is_zero(self):
        return not self.d

    def coeff(self, i, j):
        return self.d.get((i, j), sp.Integer(0))


def ten(a, b):
    return TT({(i, j): a.c[i] * b.c[j] for i in range(8) for j in range(8)})


# every blade is (+-1) I^k p with p in {1, s1, s2, s3}: found from the native product
DEC = {}
for i in range(8):
    for k_, a_ in itertools.product((0, 1), range(4)):
        cand = (PS if k_ else ONE) * (ONE if a_ == 0 else s[a_])
        for sg in (1, -1):
            if (E8[i] - sg * cand).is_zero():
                DEC[i] = (sg, k_, a_)
assert len(DEC) == 8


def zred(T):
    """canonical representative modulo I (x) 1 = 1 (x) I: the pseudoscalar is moved to the left factor."""
    out = TT()
    for (i, j), v in T.d.items():
        sg, k_, a_ = DEC[j]
        left = E8[i] * (PS if k_ else ONE)
        out = out + ten(left, ONE if a_ == 0 else s[a_]) * (sg * v)
    return out


ONE2 = ten(ONE, ONE)
II = ten(PS, PS)
I1 = ten(PS, ONE)
rho2 = (ONE2 + ten(e[1], e[1]) + ten(e[2], e[2]) + ten(e[3], e[3])) * Rational(1, 4)
a1, a2, a3, b1, b2, b3 = symbols("a1 a2 a3 b1 b2 b3", real=True)
adot = a1 * s[1] + a2 * s[2] + a3 * s[3]
bdot = b1 * s[1] + b2 * s[2] + b3 * s[3]
adb = a1 * b1 + a2 * b2 + a3 * b3

okR = (rho2 * rho2 - rho2).is_zero() and 4 * rho2.coeff(0, 0) == 1 and (rho2 - rho2.rev()).is_zero()
# rank one in Cl3 (x)_Zc Cl3: rho T rho = lambda(T) rho with lambda(T) in the centre, on a real basis (32 elements)
okZ = zred(rho2 * rho2 - rho2).is_zero()
for i, a_ in itertools.product(range(8), range(4)):
    T = ten(E8[i], ONE if a_ == 0 else s[a_])
    Tr = zred(T * rho2)
    lamT = ONE2 * (4 * Tr.coeff(0, 0)) + I1 * (4 * Tr.coeff(7, 0))
    okZ &= zred(rho2 * T * rho2 - lamT * rho2).is_zero()
okD17 = zred(rho2 - (ONE2 - ten(s[1], s[1]) - ten(s[2], s[2]) - ten(s[3], s[3])) * Rational(1, 4)).is_zero()
check("C1", "joint closure rho = (1 + sum e_j (x) e_j)/4 in Cl3 (x)_Zc Cl3: idempotent, reversion-fixed, 4<rho>_0 = 1, "
            "and primitive (rho T rho = lambda(T) rho, lambda in the centre): the singlet", okR and okZ)
Ered = 4 * zred(ten(adot, bdot) * rho2).coeff(0, 0)
check("C2", "E(a,b) = 4<(a (x) b) rho>_0 = -a.b in Cl3 (x)_Zc Cl3", zero(Ered + adb))
check("C3", "each marginal is a fair coin: 4<(a (x) 1) rho>_0 = 0 = 4<(1 (x) b) rho>_0",
      zero(zred(ten(adot, ONE) * rho2).coeff(0, 0)) and zero(zred(ten(ONE, bdot) * rho2).coeff(0, 0)))
# what happens in the 64-dimensional real tensor product
raw = ten(adot, bdot) * rho2
Pp2, Pm2 = (ONE2 + II) * Rational(1, 2), (ONE2 - II) * Rational(1, 2)
split = ((Pp2 * Pp2 - Pp2).is_zero() and (Pm2 * Pm2 - Pm2).is_zero() and (Pp2 * Pm2).is_zero()
         and not (rho2 * Pp2).is_zero() and not (rho2 * Pm2).is_zero())
note("C1/C2 in the 64-dim real tensor product Cl3 (x)_R Cl3: rho is still idempotent (%s) but NOT primitive: "
     "it splits as rho(1 + I(x)I)/2 + rho(1 - I(x)I)/2, both non-zero (%s); and the 1(x)1 coefficient of "
     "4 (a(x)b) rho is %s while its I(x)I coefficient is %s. So E(a,b) = -a.b needs I(x)1 = 1(x)I, "
     "that is the tensor product over the centre, as the paper writes it; it FAILS over R."
     % (okR, split, sp.expand(4 * raw.coeff(0, 0)), sp.factor(4 * raw.coeff(7, 7))))
note("D17 over R: (1 + sum e_j(x)e_j)/4 = (1 - sum s_j(x)s_j)/4 is %s in Cl3 (x)_R Cl3, %s in Cl3 (x)_Zc Cl3"
     % ((rho2 - (ONE2 - ten(s[1], s[1]) - ten(s[2], s[2]) - ten(s[3], s[3])) * Rational(1, 4)).is_zero(), okD17))

local = [A_ * (B_ - Bp_) + Ap * (B_ + Bp_) for A_, Ap, B_, Bp_ in itertools.product([1, -1], repeat=4)]
check("C4", "deterministic +-1 answer sheets give S = +-2 exactly", set(local) == {2, -2}, AI)
Ef = lambda a_, b_: -sum(p_ * q_ for p_, q_ in zip(a_, b_))
bb, bbp = (1, 0, 0), (0, 1, 0)
aa = tuple((p_ - q_) / sqrt(2) for p_, q_ in zip(bb, bbp))
aap = tuple((p_ + q_) / sqrt(2) for p_, q_ in zip(bb, bbp))
S_ = Ef(aa, bb) - Ef(aa, bbp) + Ef(aap, bb) + Ef(aap, bbp)
# the same S from the native correlation function
En = lambda a_, b_: Ered.subs(dict(zip((a1, a2, a3, b1, b2, b3), tuple(a_) + tuple(b_))))
S_nat = En(aa, bb) - En(aa, bbp) + En(aap, bb) + En(aap, bbp)
check("C5", "|S| = 2 sqrt 2 at b perp b', a ~ b-b', a' ~ b+b' (also with E taken from the native C2 correlation)",
      zero(sp.Abs(sp.simplify(S_)) - 2 * sqrt(2)) and zero(sp.Abs(sp.simplify(S_nat)) - 2 * sqrt(2)), AI)
cth = symbols("c", real=True)
bound = sqrt(2 - 2 * cth) + sqrt(2 + 2 * cth)
crit = sp.solve(sp.diff(bound, cth), cth)
check("C6", "|b-b'| + |b+b'| is maximised at b.b' = 0 with value 2 sqrt 2",
      crit == [0] and zero(bound.subs(cth, 0) - 2 * sqrt(2))
      and zero(sp.expand((sp.expand(bound ** 2) - 4) ** 2) - 16 * (1 - cth ** 2)), AI)

# ----------------------------------------------------------------------------------------
# Uniqueness of alpha = 2
# ----------------------------------------------------------------------------------------
n1, n2_, n3 = symbols("n1 n2 n3", real=True)
Rn = n1 * s[1] + n2_ * s[2] + n3 * s[3]
nsq = n1 ** 2 + n2_ ** 2 + n3 ** 2
check("U1", "n = n.s: reversion-fixed, <n>_0 = 0, n^2 = |n|^2",
      (Rn - Rn.rev()).is_zero() and Rn.c[0] == 0 and (Rn * Rn - nsq * ONE).is_zero())
lr, li = symbols("lambda_r lambda_i", real=True)
lmb = cen(lr, li)
Jl = ONE + lmb * Rn
check("U2", "J_l J_l~ = (1 + |l|^2 |n|^2) + (l + l~) n, l in the centre",
      (Jl * Jl.rev() - ((1 + abs2(lmb) * nsq) * ONE + (lmb + lmb.rev()) * Rn)).is_zero())
ar = symbols("ar", real=True)
Jl3 = ONE + lmb * s[3]
Dl = ar * Jl3 - Jl3 * Jl3.rev()
eqsU = [v for v in Dl.c if v != 0] + [lr ** 2 + li ** 2 - 1]
sols = sp.solve(eqsU, [ar, lr, li], dict=True)
check("U3", "alpha J = J J~ with |lambda| = 1 forces alpha = 2, lambda = +-1 (equations read off the components "
            "of alpha J_l - J_l J_l~ at n = s_3; general n by U2)",
      sorted((s_[ar], s_[lr], s_[li]) for s_ in sols) == [(2, -1, 0), (2, 1, 0)])


def zero_unit_n(q):
    return all(sp.rem(sp.expand(v), nsq - 1, n3) == 0 for v in q.c)


ok = True
for sgn in (1, -1):
    Pj = (ONE + sgn * Rn) / 2
    ok &= (Pj * Pj - Pj - (nsq - 1) / 4 * ONE).is_zero() and zero_unit_n(Pj * Pj - Pj)
    for T in E8:                                            # primitive: P T P = 2<T P>_{0+3} P
        ok &= zero_unit_n(Pj * T * Pj - 2 * (T * Pj).grades(0, 3) * Pj)
check("U4", "J/2 = (1 +- n)/2 is an idempotent for unit n, and primitive: P T P = 2<T P>_{0+3} P for all T", ok)

# ----------------------------------------------------------------------------------------
# Representation theory (character arithmetic, no algebra) -- rerun unchanged
# ----------------------------------------------------------------------------------------
def weights(j):
    j = Rational(j)
    return [j - k for k in range(int(2 * j) + 1)]


def rep(j1, j2):
    return Counter((m1, m2) for m1 in weights(j1) for m2 in weights(j2))


def tensor(c1_, c2_):
    out = Counter()
    for (m1, m2), k1 in c1_.items():
        for (p1, p2), k2 in c2_.items():
            out[(m1 + p1, m2 + p2)] += k1 * k2
    return out


def decompose(c):
    c = Counter(c)
    out = Counter()
    while +c:
        top = max(c, key=lambda mw: (mw[0] + mw[1], mw[0]))
        mult = c[top]
        out[top] += mult
        for wgt, k in rep(*top).items():
            c[wgt] -= mult * k
        c = +c
        assert all(val >= 0 for val in c.values())
    return dict(out)


half = Rational(1, 2)
vec = rep(half, half)
dec = decompose(tensor(vec, vec))
check("R1", "(1/2,1/2) x (1/2,1/2) = (0,0) + (1,0) + (0,1) + (1,1)",
      dec == {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1}, AI)
sq = tensor(vec, vec)
double = Counter({(2 * m1, 2 * m2): k for (m1, m2), k in vec.items()})
sym = Counter({k: (sq[k] + double.get(k, 0)) // 2 for k in sq})
alt = +Counter({k: (sq[k] - double.get(k, 0)) // 2 for k in sq})
check("R2", "Sym^2 = (0,0)+(1,1), Alt^2 = (1,0)+(0,1) [2-forms; grades 1+2 in the Clifford edition]",
      decompose(sym) == {(1, 1): 1, (0, 0): 1} and decompose(alt) == {(1, 0): 1, (0, 1): 1}, AI)
check("R3", "dimension count 16 = 1 + 3 + 3 + 9", sum(sq.values()) == 16 == 1 + 3 + 3 + 9, AI)
check("R4", "(1/2,0) x (0,1/2) = (1/2,1/2): complexified congruence rep is the vector",
      decompose(tensor(rep(half, 0), rep(0, half))) == {(half, half): 1}, AI)


def dec_su2(c):
    c = Counter(c)
    out = Counter()
    while +c:
        top = max(c)
        mult = c[top]
        out[top] += mult
        for wgt in weights(top):
            c[wgt] -= mult
        c = +c
    return dict(out)


t2 = Counter()
for m in weights(half):
    for p in weights(half):
        t2[m + p] += 1
check("R5", "1/2 x 1/2 = 0 + 1 for SU(2): adjoint on the generator slice and on the channel index is 1 + 3",
      dec_su2(t2) == {1: 1, 0: 1}, AI)
Pc = P.rev()                                               # for even P, P~ = Pbar = |p|^2 P^-1
Cm, Cim = channel_matrix(lambda q: P * q * Pc)
okc = all((zcoords(P * Jz * Pc)[a] - apply_channel(Cm, zs)[a]).is_zero() for a in range(4))
check("R6", "rotor conjugation p J p~ on (Z_a): real, block 1 + 3, e_0 singlet, 3x3 block in SO(3) (|p| = 1)",
      Cim == sp.zeros(4, 4) and okc and zero(Cm[0, 0] - p2)
      and all(Cm[0, k] == 0 and Cm[k, 0] == 0 for k in (1, 2, 3))
      and (Cm[1:, 1:].T * Cm[1:, 1:] - p2 ** 2 * sp.eye(3)).expand() == sp.zeros(3, 3)
      and zero(Cm[1:, 1:].det() - p2 ** 3))
s_, a_, uu, tau = symbols("s a u tau", positive=True)
frac = sp.integrate(uu ** (a_ - 1) * sp.exp(s_ * (tau - uu)), (uu, 0, sp.oo)) / sp.gamma(a_)
check("R7", "I^alpha e^{s tau} = s^-alpha e^{s tau} (fractional integral, real s > 0 as in the matrix suite)",
      zero(sp.simplify(frac - s_ ** (-a_) * sp.exp(s_ * tau))), AI)

# ----------------------------------------------------------------------------------------
# Exact random spot checks
# ----------------------------------------------------------------------------------------
random.seed(20261119)
ok = True
for _ in range(25):
    Zv = [cen(rq(), rq()) for _ in range(4)]
    pv = [rq() for _ in range(4)]
    qv = [rq() for _ in range(4)]
    Jn = sum((Zv[a] * e[a] for a in range(4)), ZERO)
    Pn = sum((pv[a] * e[a] for a in range(4)), ZERO)
    Qn = sum((qv[a] * e[a] for a in range(4)), ZERO)
    Zr = zcoords(Pn * Jn * Qn.rev())
    scale = sum(c ** 2 for c in pv) * sum(c ** 2 for c in qv)
    ok &= (sum((c * c for c in Zr), ZERO) - scale * sum((c * c for c in Zv), ZERO)).is_zero()
    ok &= zero(sum(abs2(c) for c in Zr) - scale * sum(abs2(c) for c in Zv))
check("X1", "25 exact random cases: J -> p J q~ (p, q even) keeps sum Z^2 and sum |Z|^2 (up to |p|^2 |q|^2)", ok)

# ----------------------------------------------------------------------------------------
# Dictionary checks with a native statement (the comparisons with matrices are in REP)
# ----------------------------------------------------------------------------------------
check("D1", "s_j s_k + s_k s_j = 2 delta; I = s1 s2 s3 is central, I^2 = -1",
      all((s[j] * s[k] + s[k] * s[j] - (2 if j == k else 0) * ONE).is_zero() for j in (1, 2, 3) for k in (1, 2, 3))
      and (PS * PS + ONE).is_zero() and all(comm(PS, b).is_zero() for b in E8) and (PS - E8[7]).is_zero())
ok = all((e[j] - e[j].grades(2)).is_zero() and (PS * e[j] - s[j]).is_zero() and (e[j].rev() + e[j]).is_zero()
         for j in (1, 2, 3))
ok &= all(((e[a] * e[b].rev()).c[0] == (1 if a == b else 0)) for a in range(4) for b in range(4))
ok &= (e[1] - s[3] * s[2]).is_zero() and (e[2] - s[1] * s[3]).is_zero() and (e[3] - s[2] * s[1]).is_zero()
check("D7", "e_j = -I s_j are bivectors (e_1 = s3 s2, e_2 = s1 s3, e_3 = s2 s1), I e_j = s_j, e_j~ = -e_j, "
            "<e_a e_b~>_0 = delta_ab", ok)
check("D8", "<J J~>_0 = sum |Z_a|^2; J Jbar = sum Z_a^2 (same as B1, B2)",
      zero(JJ.c[0] - normZ) and (N(J) - sumZ2).is_zero())
check("D9", "paravector X Xbar = t^2 - |x|^2; X^2 - 2tX + N(X) = 0, so the spectrum is t +- |x|",
      (N(X) - mink * ONE).is_zero() and (X * X - 2 * t * X + N(X)).is_zero())
check("D10", "N(G X G~) = |N(G)|^2 N(X) (same as L8)", (N(G * X * G.rev()) - n2 * mink * ONE).is_zero())
v1, v2, v3 = symbols("v1 v2 v3", real=True)
rhov = (ONE + v1 * s[1] + v2 * s[2] + v3 * s[3]) / 2
vsq = v1 ** 2 + v2 ** 2 + v3 ** 2
check("D11", "rho = (1 + v)/2: 2<rho^2>_0 = (1 + v^2)/2, N(rho) = (1 - v^2)/4",
      zero(2 * (rhov * rhov).c[0] - (1 + vsq) / 2) and (N(rhov) - (1 - vsq) / 4 * ONE).is_zero())
check("D12", "A = y0 I + y.e: N(A) = -y0^2 + |y|^2; N(I X) = -N(X) (same as K2, K3)",
      (N(Agen) - (-c0 ** 2 + ysq) * ONE).is_zero() and (N(PS * X) + N(X)).is_zero())
u1, u2_, u3 = symbols("u1 u2 u3", real=True)
Bv1 = Asu
Bv2 = u1 * e[1] + u2_ * e[2] + u3 * e[3]
kb = kapC(Bv1, Bv2)
check("D13", "kappa(A,B) = 8<A'B'>_{0+3} (K4); on bivectors kappa = 8<AB>_0 is real, N(A) = -A^2 = -kappa(A,A)/8",
      (kQB - 8 * (Qp * Bp).grades(0, 3)).is_zero() and is_real(kb) and zero(kb.c[0] - 8 * (Bv1 * Bv2).c[0])
      and zero(kapBiv(Bv1, Bv2) - 8 * (Bv1 * Bv2).c[0]) and (N(Bv1) + kapC(Bv1, Bv1) / 8).is_zero())
check("D14", "real Killing form (tr(ad ad) on grades 1+2): +16|x|^2 on vectors, -16|y|^2 on bivectors, cross term 0",
      zero(kapR6(xv, xv) - 16 * (x ** 2 + y ** 2 + z ** 2)) and zero(kapR6(Bv1, Bv1) + 16 * ysq)
      and zero(kapR6(xv, Bv1)))
Qe = Q.grades(0, 2)
check("D15", "on the even subalgebra reversion = Clifford conjugation, so J J~ = J Jbar is a real scalar",
      (Qe.rev() - Qe.bar()).is_zero() and (Qe * Qe.rev() - N(Qe)).is_zero() and is_real(N(Qe)))
Ppl, Pmi = (ONE + s[3]) / 2, (ONE - s[3]) / 2
ok = (Ppl * Ppl - Ppl).is_zero() and (Pmi * Pmi - Pmi).is_zero() and (Ppl * Pmi).is_zero() and (Ppl + Pmi - ONE).is_zero()
ok &= ((Bq * (Q * Ppl)) * Pmi).is_zero()
ok &= sp.Matrix([(b * Ppl).c for b in E8]).rank() == 4 and sp.Matrix([(b * Pmi).c for b in E8]).rank() == 4
ok &= sp.Matrix([(b * Ppl).c for b in E8] + [(b * Pmi).c for b in E8]).rank() == 8
check("D16", "P+- = (1 +- s_3)/2 are complementary idempotents; Cl P+ and Cl P- are left ideals of real dim 4 "
             "with Cl = Cl P+ (+) Cl P-", ok)
check("D17", "joint closure (1 + sum e_j (x) e_j)/4 = (1 - sum s_j (x) s_j)/4 in Cl3 (x)_Zc Cl3; E(a,b) = -a.b",
      okD17 and zero(Ered + adb))

# ----------------------------------------------------------------------------------------
# Naturality N1-N3
# ----------------------------------------------------------------------------------------
xs = symbols("X0:8", real=True)
Jx = MV(xs)
PARA = [ONE, s[1], s[2], s[3]]
SL = [s[1], s[2], s[3], PS * s[1], PS * s[2], PS * s[3]]   # grades 1+2: tangent to N = 1
UN = [PS, PS * s[1], PS * s[2], PS * s[3]]                 # grades 3+2: tangent to U U~ = 1
eps_ = symbols("epsilon")
tanN = (N(ONE + eps_ * Q).diff(eps_).subs({eps_: 0}) - 2 * Q.grades(0, 3)).is_zero()
tanU = (((ONE + eps_ * Q) * (ONE + eps_ * Q).rev()).diff(eps_).subs({eps_: 0}) - 2 * Q.grades(0, 1)).is_zero()


def derivative(Qm, V):
    out = ZERO
    for i in range(8):
        if V.c[i] != 0:
            out = out + Qm.diff(xs[i]) * V.c[i]
    return out


def residuals(Qm, right_invariant):
    out = {}
    for gi, A in enumerate(SL):
        Eq = derivative(Qm, A * Jx) - A * Qm - Qm * A.rev()
        for comp in range(8):
            if Eq.c[comp] != 0:
                for mon, cf in sp.Poly(Eq.c[comp], *xs).terms():
                    out[("L", gi, comp, mon)] = cf
    if right_invariant:
        for gi, B in enumerate(UN):
            Eq = derivative(Qm, Jx * B)
            for comp in range(8):
                if Eq.c[comp] != 0:
                    for mon, cf in sp.Poly(Eq.c[comp], *xs).terms():
                        out[("R", gi, comp, mon)] = cf
    return out


def dimension(degree, right_invariant):
    monos = [sp.Mul(*c) for c in itertools.combinations_with_replacement(xs, degree)]
    cols = [residuals(m * Pb, right_invariant) for Pb in PARA for m in monos]
    keys = sorted(set().union(*[set(c) for c in cols]), key=str)
    M = sp.Matrix(len(keys), len(cols), lambda r, c: cols[c].get(keys[r], 0))
    return len(cols) - DomainMatrix.from_Matrix(M).to_field().rank()


check("N1", "real-linear maps Cl3 -> paravectors equivariant for J -> GJ, X -> G X G~ (N(G) = 1; generators = grades 1+2): "
            "dimension 0", tanN and dimension(1, False) == 0)
check("N2", "real-quadratic equivariant maps: dimension 4", dimension(2, False) == 4)
clo = Jx * Jx.rev()
ok = all((derivative(clo, A * Jx) - A * clo - clo * A.rev()).is_zero() for A in SL)
ok &= all(derivative(clo, Jx * B).is_zero() for B in UN)
check("N3", "quadratic, also invariant under J -> JU (U U~ = 1; generators = grades 3+2): dimension 1, "
            "and the closure J J~ is that map", tanU and dimension(2, True) == 1 and ok)

# ----------------------------------------------------------------------------------------
# Area W1-W4
# ----------------------------------------------------------------------------------------
uw = symbols("uu0:4", real=True)
vw = symbols("vv0:4", real=True)
para = lambda w_: w_[0] * ONE + w_[1] * s[1] + w_[2] * s[2] + w_[3] * s[3]
eta = lambda P_, Q_: (P_ * Q_.bar()).c[0]
plane = lambda P_, Q_: (P_ * Q_.bar() - Q_ * P_.bar()) / 4
Uw, Vw = para(uw), para(vw)
Aw = plane(Uw, Vw)
gram = sp.expand(eta(Uw, Uw) * eta(Vw, Vw) - eta(Uw, Vw) ** 2)
ok = (N(Uw) - eta(Uw, Uw) * ONE).is_zero() and (Uw * Vw.bar() + Vw * Uw.bar() - 2 * eta(Uw, Vw) * ONE).is_zero()
check("W1", "A = (1/4)(U Vbar - V Ubar) lies in grades 1+2 (no central part) and N(A) = (1/4) det h, a real scalar "
            "(eta(u,v) = <U Vbar>_0, N(U) = eta(u,u))",
      ok and Aw.grades(0, 3).is_zero() and (N(Aw) - gram / 4 * ONE).is_zero())
check("W2", "real Killing norm (tr(ad ad) on grades 1+2): kappa(A,A) = 16<A^2>_0 = -4 det h, so -det h = kappa(A,A)/4 = -4 N(A)",
      zero(kapR6(Aw, Aw) + 4 * gram) and zero(16 * (Aw * Aw).c[0] + 4 * gram))
ok = True
for B in SL:
    dU, dV = B * Uw + Uw * B.rev(), B * Vw + Vw * B.rev()
    ok &= (plane(dU, Vw) + plane(Uw, dV) - comm(B, Aw)).is_zero()
check("W3", "equivariance: dU = BU + UB~, dV likewise (B in grades 1+2) gives dA = [B, A]", ok)
eb = [para([1 if k == j else 0 for k in range(4)]) for j in range(4)]
At, As = plane(eb[0], eb[1]), plane(eb[1], eb[2])
gt = eta(eb[0], eb[0]) * eta(eb[1], eb[1]) - eta(eb[0], eb[1]) ** 2
gs = eta(eb[1], eb[1]) * eta(eb[2], eb[2]) - eta(eb[1], eb[2]) ** 2
check("W4", "a timelike plane is boost-like (A a vector, -det h > 0); a spacelike plane is rotation-like (A a bivector, -det h < 0)",
      (At - At.grades(1)).is_zero() and not At.is_zero() and -gt > 0
      and (As - As.grades(2)).is_zero() and not As.is_zero() and -gs < 0)

# ----------------------------------------------------------------------------------------
# Three-way relation
# ----------------------------------------------------------------------------------------
IX = PS * X
kC = kapC(IX, IX)
Ix = PS * xv
kB = kapBiv(Ix, Ix)
check("T1", "N(X) = <X>_0^2 + (1/8) kappa(IX, IX) for a paravector X, kappa = tr(ad ad) over the centre on Cl3; "
            "the same with kappa of the bivector Lie algebra on I x (the central part I t drops out)",
      is_real(kC) and (N(X) - (X.c[0] ** 2 + kC.c[0] / 8) * ONE).is_zero()
      and (N(X) - (X.c[0] ** 2 + kB / 8) * ONE).is_zero())
qm = [sp.Mul(*c) for c in itertools.combinations_with_replacement((t, x, y, z), 2)]
qc = symbols("k0:%d" % len(qm))
form = sum(c * m for c, m in zip(qc, qm))
eqsT = []
for j in (1, 2, 3):
    dX = comm(e[j], X)                                      # infinitesimal rotor action, stays a paravector
    assert (dX - dX.grades(0, 1)).is_zero()
    d = sum(sp.diff(form, var) * dX.c[i] for i, var in enumerate((t, x, y, z)))
    eqsT += sp.Poly(sp.expand(d), t, x, y, z).coeffs()
Mt = sp.Matrix([[sp.expand(q_).coeff(c) for c in qc] for q_ in eqsT])
al_, be_ = symbols("al be")
fit = sp.solve(sp.Poly(sp.expand(N(X).c[0] - al_ * X.c[0] ** 2 - be_ * kC.c[0]), t, x, y, z).coeffs(), [al_, be_], dict=True)
check("T2", "rotor-invariant quadratic forms on the paravectors are 2-dimensional, so the constants 1 and 1/8 are forced",
      len(qc) - Mt.rank() == 2 and fit == [{al_: 1, be_: Rational(1, 8)}])
kR = kapR6(Ix, Ix)
note("three-way relation with the REAL Killing form of grades 1+2 (16<AB>_0): kappa_R(Ix,Ix) = %s, so the constant "
     "is 1/16 there, not 1/8 (%s). The 1/8 belongs to kappa over the centre (the paper's complex Lie algebra) "
     "and to the Killing form of the bivector algebra."
     % (sp.factor(kR), (N(X) - (X.c[0] ** 2 + kR / 16) * ONE).is_zero()))

# ----------------------------------------------------------------------------------------
# REP: the only block where matrices appear
# ----------------------------------------------------------------------------------------
print("--- REP: Pauli representation (matrices appear only here) ---")
i_ = sp.I
I2 = sp.eye(2)
pauli = [None, sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -i_], [i_, 0]]), sp.Matrix([[1, 0], [0, -1]])]


def word(b):
    M = I2
    for j in b:
        M = M * pauli[j]
    return M


RB = [word(b) for b in BL]                                 # images of the 8 blades


def rep_mv(q):
    return sum((q.c[i] * RB[i] for i in range(8)), sp.zeros(2, 2))


mz = lambda M: M.applyfunc(lambda v: sp.expand(sp.expand_complex(v))) == sp.zeros(*M.shape)
ok = all(mz(RB[i] * RB[j] - TAB[i][j][0] * RB[TAB[i][j][1]]) for i in range(8) for j in range(8))
check("REP1", "blades -> Pauli words is multiplicative on all 64 blade pairs (and 1 -> 1): an algebra homomorphism", ok and RB[0] == I2)
rows = sp.Matrix([[f(c) for c in M for f in (sp.re, sp.im)] for M in RB])
check("REP2", "the 8 images are real-independent: the homomorphism is injective, onto M2(C) by dimension 8", rows.rank() == 8)
MQ = rep_mv(Q)
check("REP-D3", "rep(q~) = rep(q)^dagger (reversion = Hermitian conjugate)", mz(rep_mv(Q.rev()) - MQ.H))
check("REP-D4", "rep(qbar) = adj rep(q), rep(N(q)) = det rep(q) 1 (Clifford conjugate = adjugate, N = det)",
      mz(rep_mv(Q.bar()) - MQ.adjugate()) and mz(rep_mv(N(Q)) - MQ.det() * I2))
check("REP-D5", "<q>_0 = (1/2) Re tr rep(q); rep(<q>_0 + <q>_3) = (1/2) tr rep(q) 1; rep(I) = i 1",
      zero(Q.c[0] - sp.re(sp.expand(MQ.trace())) / 2) and mz(rep_mv(Q.grades(0, 3)) - MQ.trace() / 2 * I2)
      and mz(rep_mv(PS) - i_ * I2))
check("REP-D6", "rep(grades 0+1) = Hermitian part, rep(grades 2+3) = anti-Hermitian part",
      mz(rep_mv(Q.grades(0, 1)) - (MQ + MQ.H) / 2) and mz(rep_mv(Q.grades(2, 3)) - (MQ - MQ.H) / 2))
check("REP-D18", "rep(q q~) = rep(q) rep(q)^dagger (U U~ = 1 is unitarity); rep(q qbar) = det (N = 1 is det = 1)",
      mz(rep_mv(Q * Q.rev()) - MQ * MQ.H) and mz(rep_mv(N(Q)) - MQ.det() * I2))
check("REP-D7", "rep(e_j) = -i sigma_j (the channel units of the matrix edition)",
      all(mz(rep_mv(e[j]) - (-i_) * pauli[j]) for j in (1, 2, 3)))
kron = sp.kronecker_product
KB = {(i, j): kron(RB[i], RB[j]) for i in range(8) for j in range(8)}
ok = mz(KB[(7, 0)] - KB[(0, 7)])                           # I (x) 1 and 1 (x) I have the same image
rows = sp.Matrix([[f(c) for c in KB[(i, a)] for f in (sp.re, sp.im)] for i in range(8) for a in range(4)])
rows64 = sp.Matrix([[f(c) for c in KB[k] for f in (sp.re, sp.im)] for k in sorted(KB)])
check("REP-C", "the Kronecker product kills I(x)1 - 1(x)I and is injective on Cl3 (x)_Zc Cl3 (rank 32 = dim M4(C)); "
               "on the 64-dim Cl3 (x)_R Cl3 its rank is also only 32, so the matrix edition's Bell block is the tensor "
               "product over the centre",
      ok and rows.rank() == 32 and rows64.rank() == 32)

# ----------------------------------------------------------------------------------------
nfail = sum(1 for r in RESULTS if r[1] is False)
nnp = sum(1 for r in RESULTS if r[1] is None)
npass = sum(1 for r in RESULTS if r[1] is True)
print("\n%d passed, %d failed, %d not ported, %d total" % (npass, nfail, nnp, len(RESULTS)))
sys.exit(1 if nfail else 0)
