"""A2: three-way identity starting from the carrier J (eight real parameters).

det X = (tr X)^2/4 + (1/8) kappa(i X0, i X0),  X = J J^dagger,
kappa(A,B) = tr(ad_A ad_B) on su(2), built from commutators in a basis.
"""
import sympy as sp

a, b, c, d, e, f, g, h = sp.symbols("a b c d e f g h", real=True)
I = sp.I
J = sp.Matrix([[a + I*b, c + I*d], [e + I*f, g + I*h]])
X = (J * J.H).applyfunc(sp.expand)

sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -I], [I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])
one = sp.eye(2)


def comm(A, B):
    return A*B - B*A


def coords(M, basis):
    """Real coordinates of M in the given real basis of su(2); exact solve, checked."""
    u = sp.symbols("u0:3")
    R = (M - sum((u[k]*basis[k] for k in range(3)), sp.zeros(2, 2))).applyfunc(sp.expand)
    eqs = []
    for entry in R:
        eqs += [sp.re(entry), sp.im(entry)]
    sol = sp.solve(eqs, u, dict=True)
    assert len(sol) == 1, "not in span / not unique"
    v = [sp.expand(sol[0][uk]) for uk in u]
    chk = (M - sum((v[k]*basis[k] for k in range(3)), sp.zeros(2, 2))).applyfunc(sp.expand)
    assert chk == sp.zeros(2, 2), "M is not in the real span of the basis"
    return v


def ad_matrices(basis):
    """ad_{E_i} as 3x3 matrices: column k holds coords of [E_i, E_k]."""
    ads = []
    for Ei in basis:
        M = sp.zeros(3, 3)
        for k, Ek in enumerate(basis):
            v = coords(comm(Ei, Ek), basis)
            for j in range(3):
                M[j, k] = v[j]
        ads.append(M)
    return ads


def make_kappa(basis):
    ads = ad_matrices(basis)

    def kappa(A, B):
        ca, cb = coords(A, basis), coords(B, basis)
        adA = sum((ca[k]*ads[k] for k in range(3)), sp.zeros(3, 3))
        adB = sum((cb[k]*ads[k] for k in range(3)), sp.zeros(3, 3))
        return sp.expand((adA*adB).trace())
    return kappa, ads


basis1 = [I*sx, I*sy, I*sz]
# second, deliberately skew and differently scaled basis, to test basis independence
basis2 = [I*sx/2 + I*sy, 3*I*sy - I*sz, I*sz/5 + 2*I*sx]
kappa1, ads1 = make_kappa(basis1)
kappa2, ads2 = make_kappa(basis2)

print("kappa on basis1 (i sx, i sy, i sz), Gram matrix:")
print(sp.Matrix(3, 3, lambda i, j: kappa1(basis1[i], basis1[j])))

# --- Hermitian and PSD checks
print("X Hermitian:", (X - X.H).applyfunc(sp.expand) == sp.zeros(2, 2))
trX = sp.expand(X.trace())
detX = sp.expand(X.det())
detJ = J.det()
absdetJ2 = sp.expand(detJ * sp.conjugate(detJ))
frob2 = a**2 + b**2 + c**2 + d**2 + e**2 + f**2 + g**2 + h**2
print("tr X imaginary part:", sp.im(trX), "| det X imaginary part:", sp.im(detX))
print("tr X == sum of 8 squares (||J||_F^2):", sp.expand(trX - frob2) == 0)
print("det X == |det J|^2:", sp.expand(detX - absdetJ2) == 0)
p, q = sp.expand(sp.re(detJ)), sp.expand(sp.im(detJ))
print("det X == Re(detJ)^2 + Im(detJ)^2 (sum of two squares):", sp.expand(detX - p**2 - q**2) == 0)
disc = sp.expand(trX**2 - 4*detX)
x1 = sp.expand(sp.re(X[0, 1])); x2 = sp.expand(-sp.im(X[0, 1])); x3 = sp.expand((X[0, 0] - X[1, 1])/2)
print("(trX)^2 - 4 detX == 4|x|^2 >= 0:", sp.expand(disc - 4*(x1**2 + x2**2 + x3**2)) == 0)

# --- Part 1
X0 = (X - trX/2*one).applyfunc(sp.expand)
A = (I*X0).applyfunc(sp.expand)
print("A = i X0 anti-Hermitian:", (A + A.H).applyfunc(sp.expand) == sp.zeros(2, 2),
      "| traceless:", sp.expand(A.trace()) == 0)
k1 = kappa1(A, A)
k2 = kappa2(A, A)
print("kappa basis-independent (basis1 vs skew basis2):", sp.expand(k1 - k2) == 0)
res = sp.expand(detX - trX**2/4 - k1/8)
print("PART 1 residual det X - (trX)^2/4 - kappa/8 =", res)
print("number of monomials: detX", len(sp.Poly(detX, a, b, c, d, e, f, g, h).terms()),
      "| kappa", len(sp.Poly(k1, a, b, c, d, e, f, g, h).terms()))

# --- In terms of J alone
trXX = sp.expand((X*X).trace())
print("kappa(iX0,iX0) == -4 tr((JJ^dag)^2) + 2 ||J||_F^4:", sp.expand(k1 + 4*trXX - 2*frob2**2) == 0)
print("kappa(iX0,iX0) == -2(||J||_F^4 - 4|det J|^2):", sp.expand(k1 + 2*(frob2**2 - 4*absdetJ2)) == 0)
print("kappa == -8|x|^2:", sp.expand(k1 + 8*(x1**2 + x2**2 + x3**2)) == 0)
print("J-only form |detJ|^2 == (||J||_F^4 - tr((JJ^dag)^2))/2:",
      sp.expand(absdetJ2 - (frob2**2 - trXX)/2) == 0)
print("shortcut check kappa(A,A) == 4 tr(A^2):", sp.expand(k1 - 4*(A*A).trace()) == 0)

# --- mutation sanity: wrong constants must fail
for cst in (sp.Rational(1, 4), sp.Rational(1, 16), -sp.Rational(1, 8)):
    print("mutated constant", cst, "residual zero?", sp.expand(detX - trX**2/4 - cst*k1) == 0)
