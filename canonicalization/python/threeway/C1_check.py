"""C1: one assertion per numbered step of C1_proof.md. Run: python C1_check.py"""
import sympy as sp

I = sp.I
s = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -I], [I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
one = sp.eye(2)
passed = []


def ok(step, cond, note=""):
    assert cond, "STEP %s FAILED %s" % (step, note)
    passed.append(step)
    print("step %-3s PASS  %s" % (step, note))


def zero(M):
    return sp.simplify(sp.expand(M)) == sp.zeros(*M.shape) if isinstance(M, sp.MatrixBase) \
        else sp.simplify(sp.expand(M)) == 0


def comm(A, B):
    return A * B - B * A


def coords(M, basis):
    """Coordinates of M in a basis (linear solve on entries); raises if M is not in the span."""
    n = len(basis)
    c = sp.symbols("c0:%d" % n)
    eqs = list(sp.expand(sum((c[k] * basis[k] for k in range(n)), sp.zeros(*M.shape)) - M))
    sol = sp.solve(eqs, c, dict=True)
    assert len(sol) == 1 and len(sol[0]) == n, "not uniquely in span"
    return [sol[0][ck] for ck in c]


def ad_matrix(A, basis):
    """Matrix of ad_A in the basis: column k holds the coordinates of [A, E_k]."""
    n = len(basis)
    cols = [coords(comm(A, basis[k]), basis) for k in range(n)]
    return sp.Matrix(n, n, lambda j, k: cols[k][j])


def killing(A, B, basis):
    return sp.expand((ad_matrix(A, basis) * ad_matrix(B, basis)).trace())


# generic complex 2x2 (entries are arbitrary commuting symbols: no field assumption used)
p, q, r, w = sp.symbols("p q r w")
G = sp.Matrix([[p, q], [r, w]])
trG, detG = G.trace(), G.det()

# Step 1: Cayley-Hamilton for 2x2
ok(1, zero(G * G - trG * G + detG * one), "X^2 - tr(X) X + det(X) 1 = 0")
# Step 2: det X = ((tr X)^2 - tr(X^2))/2
ok(2, zero(detG - (trG**2 - (G * G).trace()) / 2), "det = ((tr)^2 - tr(X^2))/2")
# Step 3: traceless decomposition
G0 = G - trG / 2 * one
ok(3, zero(G0.trace()) and zero((G * G).trace() - (G0 * G0).trace() - trG**2 / 2),
   "tr X0 = 0, tr(X^2) = tr(X0^2) + (tr X)^2/2")
# Step 4
ok(4, zero(detG - trG**2 / 4 + (G0 * G0).trace() / 2), "det = (tr)^2/4 - tr(X0^2)/2")

# Step 5: Hermitian parametrization
t, x1, x2, x3 = sp.symbols("t x1 x2 x3", real=True)
a1, a2, a3 = sp.symbols("a1 a2 a3", real=True)
h11, h22, hr, hi = sp.symbols("h11 h22 hr hi", real=True)
H = sp.Matrix([[h11, hr + I * hi], [hr - I * hi, h22]])  # general Hermitian
tt = H.trace() / 2
H0 = H - tt * one
xs = [(H0 * s[k]).trace() / 2 for k in range(3)]
rec = tt * one + sum((xs[k] * s[k] for k in range(3)), sp.zeros(2, 2))
A_H = I * H0
ok(5, zero(rec - H) and all(sp.im(sp.expand(v)) == 0 for v in xs + [tt])
   and zero(A_H + A_H.H) and zero(A_H.trace()),
   "Hermitian X = t 1 + x.s with real t, x; A = i X0 is anti-Hermitian traceless")

# Step 6: basis E_k = -(i/2) s_k, brackets [E_j, E_k] = eps_jkl E_l
E = [-I / 2 * sk for sk in s]
ok(6, all(zero(comm(E[j], E[k]) - sum((sp.LeviCivita(j, k, l) * E[l] for l in range(3)), sp.zeros(2, 2)))
          for j in range(3) for k in range(3)), "[E_j, E_k] = eps_jkl E_l")

# Step 7: ad_A from commutators equals the cross-product matrix
A = sum((ak * Ek for ak, Ek in zip((a1, a2, a3), E)), sp.zeros(2, 2))
adA = ad_matrix(A, E)
cross = sp.Matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])
ok(7, zero(adA - cross), "ad_A (built from commutators) = [a]_x")

# Step 8: ad_A^2 = a a^T - |a|^2 1, trace = -2|a|^2
av = sp.Matrix([a1, a2, a3])
n2 = a1**2 + a2**2 + a3**2
ok(8, zero(adA * adA - (av * av.T - n2 * sp.eye(3))) and zero((adA * adA).trace() + 2 * n2),
   "tr(ad_A ad_A) = -2|a|^2")

# Step 9: tr(A^2) = -|a|^2/2, hence kappa(A,A) = 4 tr(A^2); polarized version too
b1, b2, b3 = sp.symbols("b1 b2 b3", real=True)
B = sum((bk * Ek for bk, Ek in zip((b1, b2, b3), E)), sp.zeros(2, 2))
ok(9, zero((A * A).trace() + n2 / 2) and zero(killing(A, A, E) - 4 * (A * A).trace())
   and zero(killing(A, B, E) - 4 * (A * B).trace()),
   "kappa(A,A) = 4 tr(A^2), kappa(A,B) = 4 tr(AB) on su(2)")
print("        kappa on basis E_k = -(i/2) s_k:", sp.Matrix(3, 3, lambda j, k: killing(E[j], E[k], E)).tolist())

# Step 10: A = i X0  =>  tr(X0^2) = -kappa(iX0, iX0)/4 ; readout kappa = -8|x|^2
X = t * one + x1 * s[0] + x2 * s[1] + x3 * s[2]
X0 = X - X.trace() / 2 * one
kap = killing(I * X0, I * X0, E)
ok(10, zero((X0 * X0).trace() + kap / 4) and zero(kap + 8 * (x1**2 + x2**2 + x3**2)),
   "tr(X0^2) = -kappa/4 and kappa(iX0,iX0) = -8|x|^2")

# Step 11: the identity, on Hermitian X in both parametrizations
kapH = killing(I * H0, I * H0, E)
ok(11, zero(X.det() - X.trace()**2 / 4 - kap / 8) and zero(H.det() - H.trace()**2 / 4 - kapH / 8),
   "det X = (tr X)^2/4 + (1/8) kappa(iX0, iX0)")

# Step 12 (scope a): general complex X, complex-bilinear Killing form of sl(2,C)
z1, z2, z3 = sp.symbols("z1 z2 z3")  # complex coordinates in the same basis E (C-basis of sl(2,C))
G0c = coords(I * G0, E)  # exists for every complex traceless matrix
kapC = killing(I * G0, I * G0, E)
ok("12a", zero(detG - trG**2 / 4 - kapC / 8),
   "identity holds for ALL complex 2x2 X with sl(2,C) bilinear Killing form")

# Step 12 (scope b): 3x3 fails; su(3) Killing form is 6 tr(AB), and det is cubic
def eij(j, k):
    M = sp.zeros(3, 3); M[j, k] = 1; return M
B3 = []
for j in range(3):
    for k in range(j + 1, 3):
        B3.append(eij(j, k) - eij(k, j))
        B3.append(I * (eij(j, k) + eij(k, j)))
B3.append(I * (eij(0, 0) - eij(1, 1)))
B3.append(I * (eij(1, 1) - eij(2, 2)))
gram_k = sp.Matrix(8, 8, lambda j, k: 0)
ads = [ad_matrix(Bk, B3) for Bk in B3]
ok12b = all(sp.expand((ads[j] * ads[k]).trace() - 6 * (B3[j] * B3[k]).trace()) == 0
            for j in range(8) for k in range(8))
Y = 2 * sp.eye(3)  # X0 = 0, so kappa term vanishes for any constant
ok("12b", ok12b and Y.det() == 8 and Y.trace()**2 / 4 == 9,
   "su(3): kappa = 6 tr(AB) (not 4); X = 2*1_3 gives det 8 vs (tr)^2/4 = 9")

# Step 12 (scope c): numeric spot checks, including indefinite and J J^dagger
for M in [one, s[2], one + s[2], 2 * one - 3 * s[0] + s[1],
          sp.Matrix([[1, 2 + I], [3, -I]]) * sp.Matrix([[1, 2 + I], [3, -I]]).H]:
    M0 = M - M.trace() / 2 * one
    assert zero(M.det() - M.trace()**2 / 4 - killing(I * M0, I * M0, E) / 8)
ok("12c", True, "spot checks: 1, sz, 1+sz, 2-3sx+sy, J J^dagger")

print("ALL PASSED:", passed)
