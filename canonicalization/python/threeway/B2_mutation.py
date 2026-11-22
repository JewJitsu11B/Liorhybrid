# Agent B2: mutation testing and counterexample hunt for the three-way identity.
# Exact sympy arithmetic throughout. Killing form is tr(ad_A ad_B) built from commutators.
import sympy as sp

I = sp.I


def comm(a, b):
    return a * b - b * a


def hs(a, b):
    """Hilbert-Schmidt pairing tr(a^dagger b)."""
    return (a.H * b).trace()


class LieAlg:
    """Lie algebra given by a matrix basis orthogonal under tr(a^dagger b)."""

    def __init__(self, basis):
        self.basis = basis
        n = len(basis)
        for j in range(n):
            for k in range(n):
                if j != k:
                    assert sp.simplify(hs(basis[j], basis[k])) == 0, "basis not orthogonal"
        self.norms = [hs(b, b) for b in basis]
        # ad matrices: column k of ad_{E_m} holds coordinates of [E_m, E_k]
        self.ad_basis = []
        for m in range(n):
            M = sp.zeros(n, n)
            for k in range(n):
                c = comm(basis[m], basis[k])
                co = self.coords(c)
                # closure check: the commutator must be reproduced by its coordinates
                rec = sp.zeros(*c.shape)
                for j in range(n):
                    rec += co[j] * basis[j]
                assert sp.simplify(rec - c) == sp.zeros(*c.shape), "not closed"
                for j in range(n):
                    M[j, k] = co[j]
            self.ad_basis.append(M)

    def coords(self, A):
        return [sp.expand(hs(self.basis[j], A) / self.norms[j]) for j in range(len(self.basis))]

    def in_span(self, A):
        co = self.coords(A)
        rec = sp.zeros(*A.shape)
        for j, b in enumerate(self.basis):
            rec += co[j] * b
        return sp.simplify(rec - A) == sp.zeros(*A.shape)

    def ad(self, A):
        co = self.coords(A)
        n = len(self.basis)
        M = sp.zeros(n, n)
        for j in range(n):
            M += co[j] * self.ad_basis[j]
        return M

    def kappa(self, A, B):
        return sp.expand((self.ad(A) * self.ad(B)).trace())


one2 = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -I], [I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])

su2 = LieAlg([I * sx, I * sy, I * sz])
u2 = LieAlg([I * one2, I * sx, I * sy, I * sz])

print("== normalization ==")
print("kappa_su2 Gram on basis (i sx, i sy, i sz):",
      [[su2.kappa(a, b) for b in su2.basis] for a in su2.basis])
print("kappa_u2 Gram on basis (i1, i sx, i sy, i sz):",
      [[u2.kappa(a, b) for b in u2.basis] for a in u2.basis])
print("tr(AB) Gram on su2 basis:",
      [[(a * b).trace() for b in su2.basis] for a in su2.basis])

t, x1, x2, x3 = sp.symbols('t x1 x2 x3', real=True)
X = t * one2 + x1 * sx + x2 * sy + x3 * sz
X0 = X - X.trace() / 2 * one2
r2 = x1**2 + x2**2 + x3**2

print()
print("== part 1, symbolic Hermitian X = t + x.s ==")
k = su2.kappa(I * X0, I * X0)
print("tr X =", sp.expand(X.trace()), "| det X =", sp.expand(X.det()), "| kappa(iX0,iX0) =", k)
res = sp.expand(X.det() - (X.trace()**2 / 4 + k / 8))
print("residual det X - [(trX)^2/4 + kappa/8] =", res)
assert res == 0

# same with complex symbols (non-Hermitian X), kappa extended complex-bilinearly
ct, c1, c2, c3 = sp.symbols('ct c1 c2 c3')  # complex


def coords_bilinear(A):
    # complex-bilinear coordinates in basis i sx, i sy, i sz (valid for complex coefficients)
    return [sp.expand((b * A).trace() / (b * b).trace()) for b in su2.basis]


def kappa_bilinear(A, B):
    ca, cb = coords_bilinear(A), coords_bilinear(B)
    MA, MB = sp.zeros(3, 3), sp.zeros(3, 3)
    for j in range(3):
        MA += ca[j] * su2.ad_basis[j]
        MB += cb[j] * su2.ad_basis[j]
    return sp.expand((MA * MB).trace())


XC = ct * one2 + c1 * sx + c2 * sy + c3 * sz
XC0 = XC - XC.trace() / 2 * one2
resC = sp.expand(XC.det() - (XC.trace()**2 / 4 + kappa_bilinear(I * XC0, I * XC0) / 8))
print("general complex X (sl(2,C) bilinear Killing form) residual =", resC)

print()
print("== mutations ==")
trX, detX = X.trace(), X.det()
kap = k
kap_u2_X = u2.kappa(I * X, I * X)            # X instead of X0, Killing form of u(2)
# X instead of X0, ad_{iX} restricted to su(2) (iX normalizes su(2))
adr = sp.zeros(3, 3)
for kk in range(3):
    co = su2.coords(comm(I * X, su2.basis[kk]))
    for j in range(3):
        adr[j, kk] = co[j]
kap_restr_X = sp.expand((adr * adr).trace())
kap_short_X = sp.expand(4 * ((I * X) * (I * X)).trace())   # shortcut 4 tr(AB) applied to iX
trform = sp.expand(((I * X0) * (I * X0)).trace())           # trace form on iX0

mutations = [
    ("M0 original: tr^2/4 + kappa/8", trX**2 / 4 + kap / 8),
    ("M1 coefficient 1/4: tr^2/4 + kappa/4", trX**2 / 4 + kap / 4),
    ("M2 coefficient 1/16: tr^2/4 + kappa/16", trX**2 / 4 + kap / 16),
    ("M3 sign flip: tr^2/4 - kappa/8", trX**2 / 4 - kap / 8),
    ("M4 trace term: tr^2/2 + kappa/8", trX**2 / 2 + kap / 8),
    ("M5a X for X0, Killing form of u(2) by tr(ad ad)", trX**2 / 4 + kap_u2_X / 8),
    ("M5b X for X0, ad_{iX} acting on su(2)", trX**2 / 4 + kap_restr_X / 8),
    ("M5c X for X0, shortcut 4 tr(AB) applied to iX", trX**2 / 4 + kap_short_X / 8),
    ("M6 trace form tr(AB) for Killing: tr^2/4 + tr((iX0)^2)/8", trX**2 / 4 + trform / 8),
]

# concrete points: P1 = J J^dagger with J = [[1,2],[0,1]] -> X = [[5,2],[2,1]] = 3 + 2 sx + 2 sz
# P2 = 2 - 3 sx + sy (indefinite)
points = [
    ("X=[[5,2],[2,1]] (=JJ^dag, J=[[1,2],[0,1]])", {t: 3, x1: 2, x2: 0, x3: 2}),
    ("X=2-3sx+sy", {t: 2, x1: -3, x2: 1, x3: 0}),
]
for name, rhs in mutations:
    resid = sp.factor(sp.expand(detX - rhs))
    status = "SURVIVES (identically zero residual)" if resid == 0 else "KILLED"
    print(name)
    print("   symbolic residual det X - RHS =", resid, "->", status)
    for pn, sub in points:
        print("   at", pn, ": det X =", detX.subs(sub), ", RHS =", sp.nsimplify(rhs.subs(sub)),
              ", residual =", sp.nsimplify(sp.expand(detX - rhs).subs(sub)))

print()
print("== 3x3 analogue ==")
# Gell-Mann matrices
l = [
    sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
    sp.Matrix([[0, -I, 0], [I, 0, 0], [0, 0, 0]]),
    sp.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    sp.Matrix([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
    sp.Matrix([[0, 0, -I], [0, 0, 0], [I, 0, 0]]),
    sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
    sp.Matrix([[0, 0, 0], [0, 0, -I], [0, I, 0]]),
    sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -2]]),   # unnormalized lambda_8 (keeps rationals)
]
su3 = LieAlg([I * m for m in l])
print("kappa_su3 on basis diag:", [su3.kappa(b, b) for b in su3.basis])
print("6*tr(AB) on basis diag:", [sp.expand(6 * (b * b).trace()) for b in su3.basis],
      " <- 6 tr(AB), i.e. kappa = 2n tr(AB) with n=3")

s = sp.symbols('s', real=True)
y = sp.symbols('y1:9', real=True)
Y = s * sp.eye(3)
for j in range(8):
    Y += y[j] * l[j]
assert sp.simplify(Y - Y.H) == sp.zeros(3, 3)
Y0 = Y - Y.trace() / 3 * sp.eye(3)
k3 = su3.kappa(I * Y0, I * Y0)
trY = sp.expand(Y.trace())
detY = sp.expand(Y.det())
detY0 = sp.expand(Y0.det())
e2Y = sp.expand((Y.trace()**2 - (Y * Y).trace()) / 2)
print("kappa_su3(iY0,iY0) =", sp.factor(k3))
print("check kappa = 6 tr((iY0)^2):", sp.expand(k3 - 6 * ((I * Y0) * (I * Y0)).trace()) == 0)

a, b, c = sp.symbols('a b c')
allsyms = (s,) + tuple(y)


def solve_identity(expr, unknowns):
    eqs = sp.Poly(sp.expand(expr), *allsyms).coeffs()
    return sp.solve(eqs, unknowns, dict=True)


print("(i)  det Y = a tr^2 + b kappa           solutions:", solve_identity(detY - a * trY**2 - b * k3, [a, b]))
print("(ii) det Y = a tr^3 + b tr*kappa        solutions:", solve_identity(detY - a * trY**3 - b * trY * k3, [a, b]))
print("(iii) det Y = a tr^3 + b tr*kappa + c det(Y0) solutions:",
      solve_identity(detY - a * trY**3 - b * trY * k3 - c * detY0, [a, b, c]))
print("(iv) e2(Y) = a tr^2 + b kappa  [e2 = sum of principal 2x2 minors] solutions:",
      solve_identity(e2Y - a * trY**2 - b * k3, [a, b]))

# concrete obstruction: same trace, same kappa, different det
for nm, D in [("diag(5,3,-8)", sp.diag(5, 3, -8)), ("diag(7,0,-7)", sp.diag(7, 0, -7)),
              ("1+diag(5,3,-8)", sp.eye(3) + sp.diag(5, 3, -8)), ("1+diag(7,0,-7)", sp.eye(3) + sp.diag(7, 0, -7))]:
    D0 = D - D.trace() / 3 * sp.eye(3)
    print("  ", nm, ": tr =", D.trace(), ", kappa(iD0,iD0) =", su3.kappa(I * D0, I * D0), ", det =", D.det())

print()
print("== what is special about 2x2: general n, e2 identity constants (n=2 and n=3 computed above) ==")
print("n=2: e2 = det; constants (n-1)/(2n) = 1/4 and 1/(4n) = 1/8")
print("n=3: constants from solve (iv) above; det is cubic and needs det(Y0) (cubic Casimir)")
print("DONE")
