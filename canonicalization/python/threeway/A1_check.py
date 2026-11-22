"""A1: three-way identity, Killing form of su(2) built from ad matrices (sympy, exact)."""
import sympy as sp

I = sp.I
one = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -I], [I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])
paulis = [sx, sy, sz]


def comm(a, b):
    return a * b - b * a


# Real basis of su(2): E_k = i sigma_k (anti-Hermitian, traceless).
E = [I * s for s in paulis]
for e in E:
    assert e.H == -e and e.trace() == 0


def coords(M):
    """Real coordinates of M in su(2) w.r.t. E, by solving M = sum c_j E_j (all 4 entries)."""
    c = sp.symbols('c0:3')
    R = (M - sum((c[j] * E[j] for j in range(3)), sp.zeros(2, 2))).applyfunc(sp.expand)
    eqs = []
    for ent in R:
        eqs += [sp.expand(sp.re(ent)), sp.expand(sp.im(ent))]
    # entries are linear in c with complex coefficients; split using real symbols
    sol = sp.solve(list(R), c, dict=True)
    assert len(sol) == 1 and set(sol[0].keys()) == set(c), "not in span of basis"
    out = [sp.simplify(sol[0][cj]) for cj in c]
    # confirm reconstruction exactly
    back = sum((out[j] * E[j] for j in range(3)), sp.zeros(2, 2))
    assert (back - M).applyfunc(sp.simplify) == sp.zeros(2, 2)
    return out


def ad(M):
    """3x3 matrix of ad_M in basis E: column k holds coords of [M, E_k]."""
    cols = [coords(comm(M, E[k])) for k in range(3)]
    return sp.Matrix(3, 3, lambda j, k: cols[k][j])


adE = [ad(e) for e in E]
print("ad matrices of basis E_k = i sigma_k:")
for k, m in enumerate(adE):
    print(" ad_E%d =" % (k + 1), m.tolist())

# Jacobi / representation check: ad_[a,b] = [ad_a, ad_b]
for a in range(3):
    for b in range(3):
        assert ad(comm(E[a], E[b])) == comm(adE[a], adE[b])

gram = sp.Matrix(3, 3, lambda a, b: (adE[a] * adE[b]).trace())
print("Killing Gram matrix kappa(E_a, E_b) =", gram.tolist())


def kappa(A, B):
    a = coords(A)
    b = coords(B)
    adA = sum((a[j] * adE[j] for j in range(3)), sp.zeros(3, 3))
    adB = sum((b[j] * adE[j] for j in range(3)), sp.zeros(3, 3))
    return sp.expand((adA * adB).trace())


# cross-check (verification of the shortcut, not used as a definition): kappa = 4 tr(AB)
for a in range(3):
    for b in range(3):
        assert gram[a, b] == 4 * (E[a] * E[b]).trace()
print("check kappa(E_a,E_b) == 4 tr(E_a E_b): True")

t, x1, x2, x3 = sp.symbols('t x1 x2 x3', real=True)
X = t * one + x1 * sx + x2 * sy + x3 * sz
assert (X.H - X).applyfunc(sp.simplify) == sp.zeros(2, 2)
X0 = X - sp.Rational(1, 2) * X.trace() * one
A = I * X0
n2 = x1**2 + x2**2 + x3**2

trX = sp.expand(X.trace())
detX = sp.expand(X.det())
kap = kappa(A, A)

# Part 2
r2a = sp.expand(trX - 2 * t)
r2b = sp.expand(detX - (t**2 - n2))
r2c = sp.expand(kap + 8 * n2)
print("PART 2 residuals: trX-2t =", r2a, "; detX-(t^2-|x|^2) =", r2b, "; kappa+8|x|^2 =", r2c)
print("  kappa(iX0,iX0) =", sp.factor(kap))

# Part 1
r1 = sp.expand(detX - (trX**2 / 4 + sp.Rational(1, 8) * kap))
print("PART 1 residual: detX - (trX^2/4 + kappa/8) =", r1)

# Part 3: rho = X / trX (requires t != 0), r = x / t
tn = sp.symbols('t', real=True, nonzero=True)
Xn = X.subs(t, tn)
rho = (Xn / Xn.trace()).applyfunc(sp.simplify)
r = [xi / tn for xi in (x1, x2, x3)]
rr = sum(ri**2 for ri in r)
r3form = (rho - sp.Rational(1, 2) * (one + r[0] * sx + r[1] * sy + r[2] * sz)).applyfunc(sp.simplify)
r3a = sp.simplify(rho.det() - (1 - rr) / 4)
r3b = sp.simplify((rho * rho).trace() - (1 + rr) / 2)
Arho = I * (rho - one / 2)
kap_rho = kappa(Arho, Arho)
r3c = sp.simplify(-sp.Rational(1, 8) * kap_rho - rr / 4)
print("PART 3 residuals: rho-(1+r.s)/2 =", r3form.tolist(), "; det =", r3a,
      "; tr(rho^2) =", r3b, "; -kappa/8 - |r|^2/4 =", r3c)

# Part 3 again with independent symbols r1,r2,r3 (no reference to t)
q1, q2, q3 = sp.symbols('r1 r2 r3', real=True)
rho2 = (one + q1 * sx + q2 * sy + q3 * sz) / 2
q2n = q1**2 + q2**2 + q3**2
A2 = I * (rho2 - one / 2)
p3 = [sp.expand(rho2.det() - (1 - q2n) / 4),
      sp.expand((rho2 * rho2).trace() - (1 + q2n) / 2),
      sp.expand(-sp.Rational(1, 8) * kappa(A2, A2) - q2n / 4)]
print("PART 3 (free r) residuals:", p3)

# Wrong-constant sanity: the identity must fail with 1/4 or 1/16
for cst in (sp.Rational(1, 4), sp.Rational(1, 16)):
    print("  with constant", cst, "residual =", sp.expand(detX - (trX**2 / 4 + cst * kap)))

# Basis dependence of normalization: T_k = -(i/2) sigma_k
T = [-I * s / 2 for s in paulis]
print("kappa on T_k = -(i/2) sigma_k:", [kappa(Tk, Tk) for Tk in T])

ok = all(v == 0 for v in [r1, r2a, r2b, r2c, r3a, r3b, r3c] + p3) and r3form == sp.zeros(2, 2)
print("ALL PARTS 1-3 EXACT:", ok)
