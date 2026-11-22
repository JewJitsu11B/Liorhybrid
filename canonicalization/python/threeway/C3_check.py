"""C3: invariant-theory proof of  det X = (tr X)^2/4 + (1/8) kappa(i X0, i X0)  on Herm(2).

Every ingredient of the symmetry argument is checked by exact sympy computation.
Killing form is built from its definition tr(ad_A ad_B), ad from commutators in a basis.
"""
import itertools
import sympy as sp

I = sp.I
s0 = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -I], [I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])
paulis = [sx, sy, sz]


def comm(A, B):
    return A * B - B * A


# ---------------------------------------------------------------- Killing form of su(2)
E = [I * s for s in paulis]  # real basis of su(2): traceless anti-Hermitian
for e in E:
    assert e.H == -e and e.trace() == 0


def coords(A, basis):
    """Coordinates of A in the given basis, by solving the linear system entrywise."""
    c = sp.symbols('c0:%d' % len(basis))
    M = sum((ci * b for ci, b in zip(c, basis)), sp.zeros(*A.shape)) - A
    sol = sp.solve(list(M), c, dict=True)
    assert len(sol) == 1 and len(sol[0]) == len(basis), "not in span / not unique"
    return [sp.simplify(sol[0][ci]) for ci in c]


def ad_matrices(basis):
    n = len(basis)
    ads = []
    for e in basis:
        m = sp.zeros(n, n)
        for k, ek in enumerate(basis):
            col = coords(comm(e, ek), basis)   # [e, e_k] = sum_j col[j] e_j
            for j in range(n):
                m[j, k] = col[j]
        ads.append(m)
    return ads


ad = ad_matrices(E)
gram = sp.Matrix(3, 3, lambda j, k: (ad[j] * ad[k]).trace())
print("kappa(E_j, E_k) on basis E_k = i sigma_k (from tr(ad ad)):")
sp.pprint(gram)
assert gram == -8 * sp.eye(3)


def kappa(A, B):
    a = sp.Matrix(coords(A, E))
    b = sp.Matrix(coords(B, E))
    return sp.expand((a.T * gram * b)[0, 0])


# ---------------------------------------------------------------- the two sides
t, x1, x2, x3 = sp.symbols('t x1 x2 x3', real=True)
v = [t, x1, x2, x3]
herm_basis = [s0] + paulis


def herm(c):
    return sum((ci * b for ci, b in zip(c, herm_basis)), sp.zeros(2, 2))


def herm_coords(X):
    return coords(X, herm_basis)


def traceless(X):
    return X - sp.Rational(1, 2) * X.trace() * s0


def f_det(X):
    return sp.expand(X.det())


def f_tr2(X):
    return sp.expand(X.trace() ** 2)


def f_kap(X):
    return kappa(I * traceless(X), I * traceless(X))


def f_rhs(X):
    return sp.expand(f_tr2(X) / 4 + f_kap(X) / 8)


X = herm(v)
assert X.H == X
readouts = {'det': f_det(X), 'tr^2': f_tr2(X), 'kappa': f_kap(X)}
print("\nreadouts in coordinates X = t 1 + x.s:")
for k, val in readouts.items():
    print("  %-6s = %s" % (k, val))

# ---------------------------------------------------------------- (i) both sides: real quadratics
for name, p in list(readouts.items()) + [('rhs', f_rhs(X))]:
    P = sp.Poly(p, *v)
    assert P.is_homogeneous and P.total_degree() == 2, name
    assert all(c.is_real for c in P.coeffs()), name
print("\n(i-a) det, tr^2, kappa, rhs are homogeneous real quadratics in (t,x1,x2,x3): OK")

# (i-b) group invariance, exact.  U = a1 + i(a2 sx + a3 sy + a4 sz) is n^(1/2) times a
# general SU(2) element, n = a1^2+..+a4^2, U U^dagger = n 1.  For a homogeneous quadratic f,
# invariance under SU(2) is equivalent to f(U X U^dagger) = n^2 f(X) identically.
a = sp.symbols('a1:5', real=True)
U = a[0] * s0 + I * (a[1] * sx + a[2] * sy + a[3] * sz)
n = sum(ai ** 2 for ai in a)
assert (U * U.H).applyfunc(sp.expand) == n * s0
assert sp.expand(U.det() - n) == 0
Y = (U * X * U.H).applyfunc(sp.expand)
assert Y.H.applyfunc(sp.expand) == Y
for name, f in [('det', f_det), ('tr^2', f_tr2), ('kappa', f_kap), ('rhs', f_rhs)]:
    assert sp.expand(f(Y) - n ** 2 * f(X)) == 0, name
print("(i-b) f(U X U^dagger) = f(X) for all U in SU(2), f in {det, tr^2, kappa, rhs}: OK"
      " (polynomial identity in 4+4 symbols)")

# ---------------------------------------------------------------- (ii) dimension of invariants
def invariant_quadratics(hbasis, gens, label):
    """Nullspace of the infinitesimal action X -> [A, X], A in gens, on quadratics."""
    d = len(hbasis)
    ys = sp.symbols('y0:%d' % d, real=True)
    Xg = sum((yi * b for yi, b in zip(ys, hbasis)), sp.zeros(*hbasis[0].shape))
    monos = [ys[i] * ys[j] for i in range(d) for j in range(i, d)]
    qs = sp.symbols('q0:%d' % len(monos))
    Q = sum(qi * m for qi, m in zip(qs, monos))
    eqs = []
    for A in gens:
        dX = comm(A, Xg)                       # tangent vector, again Hermitian
        assert dX.H.applyfunc(sp.expand) == dX.applyfunc(sp.expand)
        dy = coords(dX, hbasis)                # the linear vector field in coordinates
        LQ = sp.expand(sum(sp.diff(Q, yi) * dyi for yi, dyi in zip(ys, dy)))
        eqs += sp.Poly(LQ, *ys).coeffs()
    M = sp.Matrix([[sp.diff(e, qi) for qi in qs] for e in eqs])
    ns = M.nullspace()
    print("(ii) %s: quadratics %d, equations %d, rank %d, invariant dimension %d"
          % (label, len(monos), M.rows, M.rank(), len(ns)))
    return ys, monos, ns


ys, monos, ns = invariant_quadratics(herm_basis, E, "Herm(2) under su(2)")
assert len(ns) == 2
inv_basis = [sp.expand(sum(c * m for c, m in zip(vec, monos))) for vec in ns]
print("     basis of invariant quadratics:", inv_basis)

# the three readouts lie in that space and span it; any two are independent
def vec_of(p):
    P = sp.Poly(p.subs(dict(zip(v, ys))), *ys)
    return sp.Matrix([P.coeff_monomial(m) for m in monos])


N = sp.Matrix.hstack(*ns)
R = {k: vec_of(p) for k, p in readouts.items()}
for k, r in R.items():
    assert sp.Matrix.hstack(N, r).rank() == 2, k
assert sp.Matrix.hstack(*R.values()).rank() == 2
for k1, k2 in itertools.combinations(R, 2):
    assert sp.Matrix.hstack(R[k1], R[k2]).rank() == 2, (k1, k2)
rel = sp.Matrix.hstack(*R.values()).nullspace()
assert len(rel) == 1
print("     det, tr^2, kappa all lie in it, span it, pairwise independent")
print("     unique linear relation (coefficients of det, tr^2, kappa), up to scale:",
      list(rel[0] / rel[0][0]))

# ---------------------------------------------------------------- (iii) two evaluations
pts = {'1': s0, 'sz': sz}
ev = sp.Matrix([[sp.expand(sum(c * m for c, m in zip(vec, monos))).subs(
    dict(zip(ys, herm_coords(P)))) for vec in ns] for P in pts.values()])
assert ev.rank() == 2
print("\n(iii) evaluation at {1, sz} on the invariant space has rank", ev.rank(),
      "-> two evaluations determine an invariant quadratic")
for name, P in pts.items():
    L, Rr = f_det(P), f_rhs(P)
    print("     X = %-2s: det = %s, tr^2 = %s, kappa(iX0,iX0) = %s, rhs = %s"
          % (name, L, f_tr2(P), f_kap(P), Rr))
    assert L == Rr

al, be = sp.symbols('alpha beta')
sol = sp.solve([f_det(P) - al * f_tr2(P) - be * f_kap(P) for P in pts.values()], [al, be], dict=True)
print("     solving det = alpha tr^2 + beta kappa at the two points:", sol)
assert sol == [{al: sp.Rational(1, 4), be: sp.Rational(1, 8)}]

# conclusion cross-check (not part of the proof): direct expansion
assert sp.expand(f_det(X) - f_rhs(X)) == 0
print("\ncross-check by direct expansion: det X - rhs =", sp.expand(f_det(X) - f_rhs(X)))

# ---------------------------------------------------------------- scope probes
# (a) other evaluation pairs: which pairs would NOT have sufficed
for name, pair in {'{1, 2*1}': [s0, 2 * s0], '{sz, sx}': [sz, sx], '{1, 1+sz}': [s0, s0 + sz]}.items():
    m = sp.Matrix([[b.subs(dict(zip(ys, herm_coords(P)))) for b in inv_basis] for P in pair])
    print("scope: evaluation pair %-10s has rank %d" % (name, m.rank()))

# (b) Herm(3) under su(3): invariant quadratics, and det is not one of them
def Eij(i, j, nn=3):
    m = sp.zeros(nn, nn)
    m[i, j] = 1
    return m


h3 = [sp.eye(3)]
h3 += [Eij(i, j) + Eij(j, i) for i in range(3) for j in range(i + 1, 3)]
h3 += [-I * Eij(i, j) + I * Eij(j, i) for i in range(3) for j in range(i + 1, 3)]
h3 += [Eij(0, 0) - Eij(1, 1), Eij(1, 1) - Eij(2, 2)]
su3 = [I * h for h in h3[1:]]
y3, m3, ns3 = invariant_quadratics(h3, su3, "Herm(3) under su(3)")
X3 = sum((yi * b for yi, b in zip(y3, h3)), sp.zeros(3, 3))
print("     deg det on Herm(3) =", sp.Poly(sp.expand(X3.det()), *y3).total_degree(),
      "(cubic, so not in the quadratic invariant space)")
print("\nALL CHECKS PASSED")
