"""
Symbolic (sympy) verification of the algebra and representation-theory claims in
onesheet_letter_native_v9.pdf.  Independent of the Lean development in ../lean.

Every check is exact (symbolic or exact rational arithmetic), never floating point.
Run:  python check_onesheet_sympy.py
"""
import itertools
import random
import sys
from collections import Counter

import sympy as sp
from sympy import I, Matrix, Rational, conjugate, eye, im, re, simplify, sqrt, symbols, zeros

RESULTS = []


def is_zero(x):
    if isinstance(x, sp.MatrixBase):
        return all(is_zero(e) for e in x)
    x = sp.expand(sp.expand_complex(sp.sympify(x)))
    if x == 0:
        return True
    return simplify(x) == 0


def check(tag, section, claim, ok):
    ok = bool(ok)
    RESULTS.append((tag, section, claim, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag:>4}  {section:<14} {claim}")


def dag(M):
    return M.H


# ----------------------------------------------------------------------------
# Carrier: J = sum_a Z_a e_a,  e_0 = 1, e_j = -i sigma_j
# ----------------------------------------------------------------------------
s0 = eye(2)
sx = Matrix([[0, 1], [1, 0]])
sy = Matrix([[0, -I], [I, 0]])
sz = Matrix([[1, 0], [0, -1]])
sig = [sx, sy, sz]
e = [s0] + [-I * s for s in sig]

xr = symbols("x0:4", real=True)
yr = symbols("y0:4", real=True)
Z = [xr[a] + I * yr[a] for a in range(4)]
J = sum((Z[a] * e[a] for a in range(4)), zeros(2, 2))
JJd = (J * dag(J)).applyfunc(sp.expand)
normZ = sum(xr[a] ** 2 + yr[a] ** 2 for a in range(4))          # sum |Z_a|^2
sumZ2 = sp.expand(sum(z ** 2 for z in Z))                        # sum Z_a^2


def generic_complex_matrix(prefix):
    a = symbols(f"{prefix}r0:4", real=True)
    b = symbols(f"{prefix}i0:4", real=True)
    return Matrix(2, 2, [a[k] + I * b[k] for k in range(4)])


G = generic_complex_matrix("g")
Q = generic_complex_matrix("q")

# --- quaternion relations -----------------------------------------------------
eps = sp.LeviCivita
ok = True
for j in range(1, 4):
    for k in range(1, 4):
        rhs = -sp.KroneckerDelta(j, k) * s0 + sum(
            (eps(j, k, l) * e[l] for l in range(1, 4)), zeros(2, 2))
        ok &= is_zero(e[j] * e[k] - rhs)
check("Q1", "carrier", "e_j e_k = -delta_jk + eps_jkl e_l  (e_j = -i sigma_j)", ok)

# --- CAL at alpha = 2 ----------------------------------------------------------
al, Jx, Jxp = symbols("alpha J_x J_xp")
check("B0", "Born", "alpha J - (1-alpha) J' at alpha=2 equals 2J + J'",
      is_zero((al * Jx - (1 - al) * Jxp).subs(al, 2) - (2 * Jx + Jxp)))

# --- Born quadratic -----------------------------------------------------------
check("B1", "Born", "(1/2) tr(J J^dag) = sum_a |Z_a|^2",
      is_zero(JJd.trace() / 2 - normZ))
check("B2", "Born", "det J = sum_a Z_a^2", is_zero(J.det() - sumZ2))
check("B3", "Born", "J J^dag is Hermitian", is_zero(JJd - dag(JJd)))
check("B4", "Born", "det(J J^dag) = |det J|^2 >= 0 and tr >= 0 (positive cone)",
      is_zero(JJd.det() - sp.expand(sumZ2 * conjugate(sumZ2))))

# right U(2): U = e^{i phi} * unit quaternion
ph = symbols("phi", real=True)
u = symbols("u0:4", real=True)
Uq = sum((u[a] * e[a] for a in range(4)), zeros(2, 2))       # |u|^2 * SU(2) element
u2 = sum(c ** 2 for c in u)
check("B5", "Born", "U = e^{i phi} (u.e) satisfies U U^dag = |u|^2  (so U(2) at |u|=1)",
      is_zero(sp.exp(I * ph) * Uq * dag(sp.exp(I * ph) * Uq) - u2 * s0))
JU = J * (sp.exp(I * ph) * Uq)
check("B6", "Born", "J J^dag invariant under J -> J U, U in U(2)",
      is_zero((JU * dag(JU)).applyfunc(sp.expand) - u2 * JJd))
check("B7", "Born", "congruence covariance (GJ)(GJ)^dag = G (J J^dag) G^dag",
      is_zero(((G * J) * dag(G * J)).applyfunc(sp.expand)
              - (G * JJd * dag(G)).applyfunc(sp.expand)))

# SO(4) on the channel index: left/right unit-quaternion multiplication
v4 = symbols("p0:4", real=True)
P = sum((v4[a] * e[a] for a in range(4)), zeros(2, 2))
zs = symbols("z0:4")
Jz = sum((zs[a] * e[a] for a in range(4)), zeros(2, 2))


def coords(M):
    """coordinates of a 2x2 complex matrix in the basis e_a (complex coefficients)."""
    return [sp.expand((M * dag(e[a])).trace() / 2) for a in range(4)]


def lin_matrix(vec):
    return Matrix(4, 4, lambda r, c: sp.diff(vec[r], zs[c]))


Lm = lin_matrix(coords(P * Jz))
Rm = lin_matrix(coords(Jz * P))
p2 = sum(c ** 2 for c in v4)
check("B8", "Born", "left quaternion mult acts on (Z_a) by a REAL matrix L with L^T L = |p|^2",
      all(im(x) == 0 for x in Lm) and is_zero(Lm.T * Lm - p2 * eye(4)))
check("B9", "Born", "right quaternion mult acts on (Z_a) by a REAL matrix R with R^T R = |p|^2",
      all(im(x) == 0 for x in Rm) and is_zero(Rm.T * Rm - p2 * eye(4)))
check("B10", "Born", "det L = det R = |p|^4 (> 0, so both land in SO(4) at |p| = 1)",
      is_zero(Lm.det() - p2 ** 2) and is_zero(Rm.det() - p2 ** 2))
check("B11", "Born", "L and R commute (two commuting SU(2) factors)", is_zero(Lm * Rm - Rm * Lm))
# hence real orthogonal O preserves sum Z^2 (bilinear) and sum |Z|^2 (hermitian)
zvec = Matrix(zs)
check("B12", "Born", "sum Z_a^2 preserved by L (bilinear form, complex Z)",
      is_zero(sp.expand(((Lm * zvec).T * (Lm * zvec))[0] - p2 * (zvec.T * zvec)[0])))

# POVM pairing
E = generic_complex_matrix("E")
E = (E + dag(E)) / 2
rho = JJd
Ginv = G.inv()
Ep = dag(Ginv) * E * Ginv
rhop = G * rho * dag(G)
check("B13", "Born", "tr(E rho) invariant under rho -> G rho G^dag, E -> G^-dag E G^-1",
      is_zero(sp.simplify((Ep * rhop).trace() - (E * rho).trace())))
check("B14", "Born", "sum_n E_n = 1 maps to (G G^dag)^-1 (so needs G unitary)",
      is_zero(sp.simplify(dag(Ginv) * Ginv - (G * dag(G)).inv())))

# ----------------------------------------------------------------------------
# Lorentz signature
# ----------------------------------------------------------------------------
Hm = lambda q: (q + dag(q)) / 2
Am = lambda q: (q - dag(q)) / 2
check("L1", "Lorentz", "H is a projector, H(H(q)) = H(q)", is_zero(Hm(Hm(Q)) - Hm(Q)))
check("L2", "Lorentz", "A is a projector, H + A = 1, H A = 0",
      is_zero(Am(Am(Q)) - Am(Q)) and is_zero(Hm(Q) + Am(Q) - Q) and is_zero(Hm(Am(Q))))
check("L3", "Lorentz", "H commutes with q -> G q G^dag (any G in GL(2,C))",
      is_zero((Hm(G * Q * dag(G)) - G * Hm(Q) * dag(G)).applyfunc(sp.expand)))
check("L4", "Lorentz", "A commutes with q -> G q G^dag",
      is_zero((Am(G * Q * dag(G)) - G * Am(Q) * dag(G)).applyfunc(sp.expand)))
check("L5", "Lorentz", "i * (Hermitian slice) = anti-Hermitian slice, equivariantly",
      is_zero(Am(I * Hm(Q)) - I * Hm(Q)))

t, x, y, z = symbols("t x y z", real=True)
X = t * s0 + x * sx + y * sy + z * sz
check("L6", "Lorentz", "X = t + x.sigma is Hermitian; H(M2(C)) is 4 real dims",
      is_zero(X - dag(X)) and len(sp.Matrix([sp.re(c) for c in Hm(Q)]).free_symbols) == 4)
check("L7", "Lorentz", "det X = t^2 - x^2 - y^2 - z^2",
      is_zero(X.det() - (t ** 2 - x ** 2 - y ** 2 - z ** 2)))
detG = G.det()
check("L8", "Lorentz", "det(G X G^dag) = |det G|^2 det X",
      is_zero(sp.expand((G * X * dag(G)).det()) - sp.expand(detG * conjugate(detG) * X.det())))
check("L9", "Lorentz", "G = -1 acts trivially (kernel contains +-1)",
      is_zero((-s0) * X * dag(-s0) - X))

# Cl(3,0): h = i (central, h^2 = -1); quaternion units i,j,k = e_1,e_2,e_3
hq = [I * e[j] for j in range(1, 4)]
ok = all(is_zero(hq[j] * hq[j] - s0) for j in range(3))
ok &= all(is_zero(hq[j] * hq[k] + hq[k] * hq[j]) for j in range(3) for k in range(3) if j != k)
check("L10", "Lorentz", "(h i)^2 = +1 and hi, hj, hk anticommute: Cl(3,0) relations", ok)
blades = [s0, hq[0], hq[1], hq[2], hq[0] * hq[1], hq[1] * hq[2], hq[2] * hq[0],
          hq[0] * hq[1] * hq[2]]
realrows = Matrix([[f(c) for c in b for f in (re, im)] for b in blades])
check("L11", "Lorentz", "the 8 blades are R-independent: Cl(3,0) ~= M2(C) (real dim 8)",
      realrows.rank() == 8)
check("L12", "Lorentz", "paravector norm p * bar(p) = t^2 - |x|^2 (Minkowski)",
      is_zero(X * (t * s0 - x * sx - y * sy - z * sz)
              - (t ** 2 - x ** 2 - y ** 2 - z ** 2) * s0))

# purity
tp = symbols("t", positive=True)
Xp = X.subs(t, tp)
rhoh = Xp / Xp.trace()
vv = symbols("v", real=True)
v2expr = (x ** 2 + y ** 2 + z ** 2) / tp ** 2
check("L13", "Lorentz", "tr rho^2 = 1 - 2 det rho  (Cayley-Hamilton at unit trace)",
      is_zero(sp.simplify((rhoh * rhoh).trace() - (1 - 2 * rhoh.det()))))
check("L14", "Lorentz", "tr rho^2 = (1 + v^2)/2, v = |x|/t",
      is_zero(sp.simplify((rhoh * rhoh).trace() - (1 + v2expr) / 2)))
gam = 1 / sqrt(1 - vv ** 2)
check("L15", "Lorentz", "1 - tr rho^2 = 1/(2 gamma^2)",
      is_zero(sp.simplify(1 - (1 + vv ** 2) / 2 - 1 / (2 * gam ** 2))))

# SU(2) conjugation acts on the Bloch vector by SO(3)
U1 = Uq  # |u|^2-scaled SU(2)
Rot = Matrix(3, 3, lambda i_, j_: sp.expand((sig[i_] * U1 * sig[j_] * dag(U1)).trace() / 2))
check("L16", "Lorentz", "SU(2) conjugation = real 3x3 R with R^T R = 1, det R = +1 (|u| = 1)",
      all(im(c) == 0 for c in Rot)
      and is_zero(Rot.T * Rot - u2 ** 2 * eye(3)) and is_zero(Rot.det() - u2 ** 3))
boost = sp.diag(2, Rational(1, 2))          # det 1, Hermitian: a boost
check("L17", "Lorentz", "a boost preserves det X but changes tr X",
      is_zero((boost * X * dag(boost)).det() - X.det())
      and not is_zero((boost * X * dag(boost)).trace() - X.trace()))

# ----------------------------------------------------------------------------
# Purity from the channels
# ----------------------------------------------------------------------------
rhoJ = JJd / JJd.trace()
v2_from_rho = 1 - 4 * rhoJ.det()                      # v^2 = 1 - 4 det rho
v2_claim1 = 1 - sp.expand(sumZ2 * conjugate(sumZ2)) / normZ ** 2
lag = 4 * sum(sp.expand(im(sp.expand(Z[a] * conjugate(Z[b])))) ** 2
              for a in range(4) for b in range(a + 1, 4))
check("P1", "purity", "v^2 = 1 - 4 det rho = 1 - |sum Z^2|^2 / (sum |Z|^2)^2",
      is_zero(sp.simplify(v2_from_rho - v2_claim1)))
check("P2", "purity", "... = 4 sum_{a<b} Im(Z_a conj Z_b)^2 / (sum |Z|^2)^2  (Lagrange)",
      is_zero(sp.expand(normZ ** 2 - sp.expand(sumZ2 * conjugate(sumZ2)) - lag)))
wr, wi = symbols("wr wi", real=True)
w = wr + I * wi
check("P3", "purity", "|w|^2 - Re(w^2) = 2 (Im w)^2",
      is_zero(sp.expand(w * conjugate(w)) - re(sp.expand(w ** 2)) - 2 * wi ** 2))
realsub = {yr[a]: 0 for a in range(4)}
check("P4", "purity", "real Z_a: J J^dag = (sum Z_a^2) 1, so rho = 1/2, v = 0",
      is_zero(JJd.subs(realsub) - sum(c ** 2 for c in xr) * s0)
      and is_zero(lag.subs(realsub)))
check("P5", "purity", "v = 1  <=>  det J = sum Z_a^2 = 0 (numerator identity)",
      is_zero(sp.expand((1 - v2_claim1) * normZ ** 2 - sumZ2 * conjugate(sumZ2))))
lam_r, lam_i = symbols("lr li", real=True)
lam = lam_r + I * lam_i
Zs = [sp.expand(lam * zz) for zz in Z]
num_s = sp.expand(sum(c ** 2 for c in Zs)) * conjugate(sp.expand(sum(c ** 2 for c in Zs)))
den_s = sum(sp.expand(c * conjugate(c)) for c in Zs) ** 2
check("P6", "purity", "v invariant under common complex rescaling Z_a -> lambda Z_a",
      is_zero(sp.expand(sp.expand(num_s) * normZ ** 2
                        - sp.expand(sumZ2 * conjugate(sumZ2)) * sp.expand(den_s))))

# ----------------------------------------------------------------------------
# Generator slice u(2) and Killing form
# ----------------------------------------------------------------------------
c0, c1, c2, c3 = symbols("c0:4", real=True)
Agen = c0 * I * s0 + c1 * e[1] + c2 * e[2] + c3 * e[3]
check("K1", "Killing", "A = y0 (i1) + y_j e_j is anti-Hermitian (spans u(2))",
      is_zero(Agen + dag(Agen)))
check("K2", "Killing", "det A = -(y0)^2 + |y|^2",
      is_zero(Agen.det() - (-c0 ** 2 + c1 ** 2 + c2 ** 2 + c3 ** 2)))
check("K3", "Killing", "det(iX) = -det X", is_zero((I * X).det() + X.det()))

unit = [Matrix(2, 2, lambda r, c, k=k: 1 if 2 * r + c == k else 0) for k in range(4)]


def ad_gl2(M):
    cols = []
    for b in unit:
        C = M * b - b * M
        cols.append([C[k // 2, k % 2] for k in range(4)])
    return Matrix(cols).T


Bq = generic_complex_matrix("b")
kap = lambda M, N: sp.expand((ad_gl2(M) * ad_gl2(N)).trace())
check("K4", "Killing", "kappa(A,B) = tr(ad_A ad_B) = 4 tr(AB) - 2 trA trB on gl(2)",
      is_zero(kap(Q, Bq) - sp.expand(4 * (Q * Bq).trace() - 2 * Q.trace() * Bq.trace())))
check("K5", "Killing", "centre i1 is in the radical of kappa (degenerate on u(2))",
      is_zero(kap(I * s0, Bq)))
Asu = Agen.subs(c0, 0)
ysq = c1 ** 2 + c2 ** 2 + c3 ** 2
check("K6", "Killing", "on su(2): det A = |y|^2 = -(1/2) tr A^2 = -(1/8) kappa(A,A)",
      is_zero(Asu.det() - ysq) and is_zero(-(Asu * Asu).trace() / 2 - ysq)
      and is_zero(-kap(Asu, Asu) / 8 - ysq))
gconj = kap(G * Q * Ginv, G * Bq * Ginv)
check("K7", "Killing", "kappa is Ad-invariant: kappa(GAG^-1, GBG^-1) = kappa(A,B)",
      is_zero(sp.simplify(gconj - kap(Q, Bq))))

# uniqueness of the invariant form on su(2): solve B([x,y],z) + B(y,[x,z]) = 0
bsym = sp.symbols("b0:9")
Bf = Matrix(3, 3, bsym)
f = lambda i_, j_, k_: eps(i_ + 1, j_ + 1, k_ + 1)     # [e_i,e_j] = 2 eps_ijk e_k
eqs = []
for i_, j_, k_ in itertools.product(range(3), repeat=3):
    eqs.append(sum(f(i_, j_, l) * Bf[l, k_] for l in range(3))
               + sum(f(i_, k_, l) * Bf[j_, l] for l in range(3)))
sol = sp.linsolve(eqs, bsym)
free = set().union(*[s_.free_symbols for s_ in list(sol)[0]])
check("K8", "Killing", "ad-invariant bilinear forms on su(2) are 1-dimensional (Schur)",
      len(free) == 1)

# real Killing form of sl(2,C)_R, basis: 3 rotations e_j, 3 boosts sigma_j
basis6 = [e[1], e[2], e[3], sx, sy, sz]


def coords6(M):
    m = [sp.expand((M * s).trace() / 2) for s in sig]   # M = sum m_k sigma_k
    return [-im(mk) for mk in m] + [re(mk) for mk in m]


def ad6(M):
    return Matrix([coords6(M * b - b * M) for b in basis6]).T


K6 = Matrix(6, 6, lambda i_, j_: (ad6(basis6[i_]) * ad6(basis6[j_])).trace())
K6_claim = Matrix(6, 6, lambda i_, j_: 2 * re(4 * (basis6[i_] * basis6[j_]).trace()))
ev = K6.eigenvals()
npos = sum(m for val, m in ev.items() if val > 0)
nneg = sum(m for val, m in ev.items() if val < 0)
check("K9", "Killing", "real Killing form of sl(2,C)_R equals 2 Re kappa_C", is_zero(K6 - K6_claim))
check("K10", "Killing", "its signature is (3,3): rotations negative, boosts positive",
      (npos, nneg) == (3, 3) and all(K6[i_, i_] < 0 for i_ in range(3))
      and all(K6[i_, i_] > 0 for i_ in range(3, 6)))

# ----------------------------------------------------------------------------
# Bell / CHSH
# ----------------------------------------------------------------------------
kron = sp.kronecker_product
rho2 = (kron(s0, s0) + sum((kron(e[j], e[j]) for j in range(1, 4)), zeros(4, 4))) / 4
a1, a2, a3, b1, b2, b3 = symbols("a1 a2 a3 b1 b2 b3", real=True)
adot = a1 * sx + a2 * sy + a3 * sz
bdot = b1 * sx + b2 * sy + b3 * sz
check("C1", "Bell", "joint closure is a rank-one projector with unit trace (the singlet)",
      is_zero(rho2 * rho2 - rho2) and rho2.trace() == 1 and rho2.rank() == 1
      and is_zero(rho2 - dag(rho2)))
check("C2", "Bell", "E(a,b) = tr(rho a.sigma (x) b.sigma) = -a.b",
      is_zero(sp.expand((rho2 * kron(adot, bdot)).trace()) + (a1 * b1 + a2 * b2 + a3 * b3)))
check("C3", "Bell", "each marginal is a fair coin: tr(rho a.sigma (x) 1) = 0 = tr(rho 1 (x) b.sigma)",
      is_zero((rho2 * kron(adot, s0)).trace()) and is_zero((rho2 * kron(s0, bdot)).trace()))
local = [A_ * (B_ - Bp) + Ap * (B_ + Bp)
         for A_, Ap, B_, Bp in itertools.product([1, -1], repeat=4)]
check("C4", "Bell", "deterministic +-1 answer sheets give S = +-2 exactly",
      set(local) == {2, -2})
Ef = lambda a_, b_: -sum(p_ * q_ for p_, q_ in zip(a_, b_))
bb, bbp = (1, 0, 0), (0, 1, 0)
aa = tuple((p_ - q_) / sqrt(2) for p_, q_ in zip(bb, bbp))
aap = tuple((p_ + q_) / sqrt(2) for p_, q_ in zip(bb, bbp))
S = Ef(aa, bb) - Ef(aa, bbp) + Ef(aap, bb) + Ef(aap, bbp)
check("C5", "Bell", "|S| = 2 sqrt 2 at b perp b', a ~ b-b', a' ~ b+b'",
      is_zero(sp.Abs(sp.simplify(S)) - 2 * sqrt(2)))
cth = symbols("c", real=True)
bound = sqrt(2 - 2 * cth) + sqrt(2 + 2 * cth)       # |b-b'| + |b+b'|, c = b.b'
crit = sp.solve(sp.diff(bound, cth), cth)
check("C6", "Bell", "|b-b'| + |b+b'| is maximised at b.b' = 0 with value 2 sqrt 2",
      crit == [0] and is_zero(bound.subs(cth, 0) - 2 * sqrt(2))
      # bound^2 = 4 + 4 sqrt(1 - c^2) <= 8, checked without radicals:
      and is_zero(sp.expand((sp.expand(bound ** 2) - 4) ** 2) - 16 * (1 - cth ** 2)))

# ----------------------------------------------------------------------------
# Uniqueness of alpha = 2
# ----------------------------------------------------------------------------
n1, n2, n3 = symbols("n1 n2 n3", real=True)
Rn = n1 * sx + n2 * sy + n3 * sz
nsq = n1 ** 2 + n2 ** 2 + n3 ** 2
check("U1", "uniqueness", "R_n = n.sigma: Hermitian, R_n^2 = |n|^2",
      is_zero(Rn - dag(Rn)) and is_zero((Rn * Rn).applyfunc(sp.expand) - nsq * s0))
lr, li = symbols("lambda_r lambda_i", real=True)
lmb = lr + I * li
alc = symbols("alpha")
# with R^2 = 1 and R independent of 1:  J J^dag = (1 + |lambda|^2) + (lambda + conj lambda) R
coef_1 = sp.expand(1 + lmb * conjugate(lmb)) - alc
coef_R = sp.expand(lmb + conjugate(lmb) - alc * lmb)
sols = sp.solve([re(sp.expand(coef_1)), re(coef_R.subs(alc, symbols("ar", real=True))),
                 lr ** 2 + li ** 2 - 1], [alc, lr, li], dict=True)
Jl = s0 + lmb * Rn
lhs = (Jl * dag(Jl)).applyfunc(sp.expand)
rhs_struct = (1 + (lr ** 2 + li ** 2) * nsq) * s0 + 2 * lr * Rn
check("U2", "uniqueness", "J_l J_l^dag = (1 + |l|^2 |n|^2) + (l + conj l) R_n",
      is_zero(lhs - rhs_struct.applyfunc(sp.expand)))
ar = symbols("ar", real=True)
sols = sp.solve([2 - ar, 2 * lr - ar * lr, -ar * li, lr ** 2 + li ** 2 - 1],
                [ar, lr, li], dict=True)
check("U3", "uniqueness", "alpha J = J J^dag with |lambda| = 1 forces alpha = 2, lambda = +-1",
      sorted((s_[ar], s_[lr], s_[li]) for s_ in sols) == [(2, -1, 0), (2, 1, 0)])
unitn = {n3: 1, n1: 0, n2: 0}
ok = True
for sgn in (1, -1):
    Pj = ((s0 + sgn * Rn) / 2)
    ok &= is_zero(sp.expand((Pj * Pj - Pj).subs(n3, sp.sqrt(1 - n1 ** 2 - n2 ** 2))))
check("U4", "uniqueness", "J/2 = (1 +- R_n)/2 is a projector for unit n", ok)

# ----------------------------------------------------------------------------
# Representation theory: Clebsch-Gordan for SL(2,C) ~ (j1, j2) labels
# ----------------------------------------------------------------------------
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
check("R1", "rep theory", "(1/2,1/2) x (1/2,1/2) = (0,0) + (1,0) + (0,1) + (1,1)",
      dec == {(0, 0): 1, (1, 0): 1, (0, 1): 1, (1, 1): 1})
# symmetric / antisymmetric squares via characters: chi_S2(g) = (chi(g)^2 + chi(g^2))/2
sq = tensor(vec, vec)
double = Counter({(2 * m1, 2 * m2): k for (m1, m2), k in vec.items()})
sym = Counter({k: (sq[k] + double.get(k, 0)) // 2 for k in sq})
alt = +Counter({k: (sq[k] - double.get(k, 0)) // 2 for k in sq})   # drop zero-multiplicity weights
check("R2", "rep theory", "Sym^2 = (0,0)+(1,1) [metric + traceless sym], Alt^2 = (1,0)+(0,1) [2-forms]",
      decompose(sym) == {(1, 1): 1, (0, 0): 1} and decompose(alt) == {(1, 0): 1, (0, 1): 1})
check("R3", "rep theory", "dimension count 16 = 1 + 3 + 3 + 9",
      sum(sq.values()) == 16 == 1 + 3 + 3 + 9)
# (1/2,0) x (0,1/2) = (1/2,1/2): the congruence rep q -> G q G^dag (complexified)
check("R4", "rep theory", "(1/2,0) x (0,1/2) = (1/2,1/2): complexified congruence rep is the vector",
      decompose(tensor(rep(half, 0), rep(0, half))) == {(half, half): 1})
# adjoint of U(2) on u(2): 1 + 3 ; SU(2) on channel index a: 1 + 3
su2 = lambda j: Counter(weights(j))


def dec_su2(c):
    c = Counter(c); out = Counter()
    while +c:
        top = max(c); mult = c[top]; out[top] += mult
        for wgt in weights(top):
            c[wgt] -= mult
        c = +c
    return dict(out)


t2 = Counter()
for m in weights(half):
    for p in weights(half):
        t2[m + p] += 1
check("R5", "rep theory", "1/2 x 1/2 = 0 + 1 for SU(2): adjoint on u(2) and on the channel index is 1 + 3",
      dec_su2(t2) == {1: 1, 0: 1})
# conjugation by a unit quaternion fixes e_0 and rotates e_1..e_3
Pc = sum((v4[a] * (e[a] if a == 0 else -e[a]) for a in range(4)), zeros(2, 2))   # conj(p)
conjmat = lin_matrix(coords(P * Jz * Pc))
check("R6", "rep theory", "p J p^-1 on (Z_a): block 1 + 3, e_0 singlet, 3x3 block in SO(3)",
      is_zero(conjmat[0, 0] - p2) and all(conjmat[0, k] == 0 and conjmat[k, 0] == 0 for k in (1, 2, 3))
      and is_zero(conjmat[1:, 1:].T * conjmat[1:, 1:] - p2 ** 2 * eye(3)))

# memory kernel: characters of (R,+) diagonalise the fractional integral
s_, a_, uu, tau = symbols("s a u tau", positive=True)
frac = sp.integrate(uu ** (a_ - 1) * sp.exp(s_ * (tau - uu)), (uu, 0, sp.oo)) / sp.gamma(a_)
check("R7", "rep theory", "I^alpha e^{s tau} = s^-alpha e^{s tau}  (Liouville lower limit -inf, Re s > 0)",
      is_zero(sp.simplify(frac - s_ ** (-a_) * sp.exp(s_ * tau))))

# ----------------------------------------------------------------------------
# Exact random spot checks of the group-invariance statements (rational arithmetic)
# ----------------------------------------------------------------------------
random.seed(20261119)
rq = lambda: Rational(random.randint(-9, 9), random.randint(1, 9))
ok = True
for _ in range(25):
    Zv = [rq() + I * rq() for _ in range(4)]
    pv = [rq() for _ in range(4)]
    qv = [rq() for _ in range(4)]
    Jn = sum((Zv[a] * e[a] for a in range(4)), zeros(2, 2))
    Pn = sum((pv[a] * e[a] for a in range(4)), zeros(2, 2))
    Qn = sum((qv[a] * e[a] for a in range(4)), zeros(2, 2))
    Jr = Pn * Jn * dag(Qn)
    Zr = [((Jr * dag(e[a])).trace() / 2) for a in range(4)]
    scale = sum(c ** 2 for c in pv) * sum(c ** 2 for c in qv)
    ok &= is_zero(sum(c ** 2 for c in Zr) - scale * sum(c ** 2 for c in Zv))
    ok &= is_zero(sum(c * conjugate(c) for c in Zr) - scale * sum(c * conjugate(c) for c in Zv))
check("X1", "spot check", "25 exact random cases: J -> pJq^dag keeps sum Z^2 and sum |Z|^2 (up to |p|^2|q|^2)", ok)

# ----------------------------------------------------------------------------
nfail = sum(1 for r in RESULTS if not r[3])
print(f"\n{len(RESULTS) - nfail} passed, {nfail} failed, {len(RESULTS)} total")
sys.exit(1 if nfail else 0)
