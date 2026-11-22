"""Dictionary checks for the Clifford edition: Cl(3,0) written in the Pauli representation.

Each check compares a Clifford-algebra operation, defined on the grade expansion of a general
multivector, with the matrix operation used in the matrix edition of the paper.
Exact symbolic arithmetic; exits non-zero if any check fails.
"""
import sys
import sympy as sp

I2 = sp.eye(2)
s1 = sp.Matrix([[0, 1], [1, 0]])
s2 = sp.Matrix([[0, -sp.I], [sp.I, 0]])
s3 = sp.Matrix([[1, 0], [0, -1]])
S = [s1, s2, s3]
PS = s1 * s2 * s3  # pseudoscalar

# blades, by grade: 1 | s1 s2 s3 | s2s3 s3s1 s1s2 | s1s2s3
BLADES = [I2, s1, s2, s3, s2 * s3, s3 * s1, s1 * s2, PS]
GRADE = [0, 1, 1, 1, 2, 2, 2, 3]


def mv(c):
    return sum((c[i] * BLADES[i] for i in range(8)), sp.zeros(2, 2))


def signs(c, sg):
    return mv([sg[GRADE[i]] * c[i] for i in range(8)])


def rev(c):  # reversion: grades 0,1 fixed, 2,3 flipped
    return signs(c, {0: 1, 1: 1, 2: -1, 3: -1})


def bar(c):  # Clifford conjugation: grades 0,3 fixed, 1,2 flipped
    return signs(c, {0: 1, 1: -1, 2: -1, 3: 1})


def grades(c, keep):
    return mv([c[i] if GRADE[i] in keep else 0 for i in range(8)])


def Z(m):
    return sp.simplify(m) == sp.zeros(*m.shape)


def z(e):
    return sp.simplify(e) == 0


results = []


def check(label, text, ok):
    results.append((label, text, bool(ok)))


c = sp.symbols("c0:8", real=True)
d = sp.symbols("d0:8", real=True)
M = mv(c)
Mb = mv(d)

# D1 generators
ok = all(Z(S[j] * S[k] + S[k] * S[j] - 2 * (1 if j == k else 0) * I2) for j in range(3) for k in range(3))
ok = ok and Z(PS - sp.I * I2) and Z(PS * PS + I2) and all(Z(PS * s - s * PS) for s in S)
check("D1", "sigma_j sigma_k + sigma_k sigma_j = 2 delta; I = sigma_1 sigma_2 sigma_3 is central, I^2 = -1", ok)

# D2 the eight blades are a real basis of M2(C)
cols = sp.Matrix([[sp.re(b[i]) for b in BLADES] for i in range(4)] + [[sp.im(b[i]) for b in BLADES] for i in range(4)])
check("D2", "the eight blades are a real basis of M2(C)", cols.rank() == 8)

# D3 reversion = conjugate transpose
check("D3", "reversion = Hermitian conjugate", Z(rev(c) - M.H))

# D4 Clifford conjugation = adjugate, M Mbar = det M
check("D4", "Clifford conjugation = adjugate; M Mbar = det M", Z(bar(c) - M.adjugate()) and Z(M * bar(c) - M.det() * I2))

# D5 scalar part and central part
ok = z(c[0] - sp.re(M.trace()) / 2) and Z(grades(c, {0, 3}) - (M.trace() / 2) * I2)
check("D5", "<M>_0 = (1/2) Re tr M; <M>_0 + <M>_3 = (1/2) tr M", ok)

# D6 the projectors H and A are grade projections
ok = Z((M + M.H) / 2 - grades(c, {0, 1})) and Z((M - M.H) / 2 - grades(c, {2, 3}))
check("D6", "(q + q~)/2 = grades 0+1; (q - q~)/2 = grades 2+3", ok)

# D7 channel units
e = [I2] + [-PS * s for s in S]
eps = sp.LeviCivita
ok = all(Z(PS * e[j + 1] - S[j]) for j in range(3))
ok = ok and all(
    Z(e[j + 1] * e[k + 1] + (1 if j == k else 0) * I2 - sum((eps(j, k, l) * e[l + 1] for l in range(3)), sp.zeros(2, 2)))
    for j in range(3)
    for k in range(3)
)
ok = ok and all(Z(e[j].H + e[j]) for j in (1, 2, 3))
check("D7", "e_j = -I sigma_j are bivectors, I e_j = sigma_j, e_j e_k = -delta + eps e_l, e_j~ = -e_j", ok)

# D8 carrier: Born weight and norm
zr = sp.symbols("x0:4", real=True)
zi = sp.symbols("y0:4", real=True)
Zc = [zr[a] + sp.I * zi[a] for a in range(4)]
J = sum((Zc[a] * e[a] for a in range(4)), sp.zeros(2, 2))
born = sp.re((J * J.H).trace()) / 2
ok = z(born - sum(zr[a] ** 2 + zi[a] ** 2 for a in range(4)))
ok = ok and Z(J * J.adjugate() - sum(Zc[a] ** 2 for a in range(4)) * I2)
check("D8", "<J J~>_0 = sum |Z_a|^2; J Jbar = sum Z_a^2", ok)

# D9 paravector norm and positive cone
t, x, y, w = sp.symbols("t x y w", real=True)
X = t * I2 + x * s1 + y * s2 + w * s3
r = sp.sqrt(x**2 + y**2 + w**2)
ok = Z(X * X.adjugate() - (t**2 - x**2 - y**2 - w**2) * I2)
ok = ok and z(sp.expand((sp.Symbol("L") - (t + r)) * (sp.Symbol("L") - (t - r))) - sp.expand(X.charpoly(sp.Symbol("L")).as_expr()))
check("D9", "paravector X Xbar = t^2 - |x|^2; eigenvalues t +- |x| (positive iff t >= |x|)", ok)

# D10 norm under congruence
G = Mb
lhs = (G * X * G.H).det()
ok = z(sp.expand(lhs - G.det() * sp.conjugate(G.det()) * X.det()))
check("D10", "N(G X G~) = |N(G)|^2 N(X)", ok)

# D11 purity for rho = (1 + v)/2
v1, v2, v3 = sp.symbols("v1 v2 v3", real=True)
rho = (I2 + v1 * s1 + v2 * s2 + v3 * s3) / 2
vsq = v1**2 + v2**2 + v3**2
ok = z((rho * rho).trace() - (1 + vsq) / 2) and z(rho.det() - (1 - vsq) / 4)
check("D11", "rho = (1 + v)/2: 2<rho^2>_0 = (1 + v^2)/2, rho rhobar = (1 - v^2)/4", ok)

# D12 generator slice
y0, y1, y2, y3 = sp.symbols("y0 y1 y2 y3", real=True)
A = y0 * PS + y1 * e[1] + y2 * e[2] + y3 * e[3]
ok = Z(A * A.adjugate() - (-y0**2 + y1**2 + y2**2 + y3**2) * I2)
ok = ok and Z((PS * X) * (PS * X).adjugate() + X * X.adjugate())
check("D12", "A = y0 I + y.e: A Abar = -y0^2 + |y|^2; (I X)(I X)bar = -X Xbar", ok)

# D13 Killing form in grade language
cen = lambda m: (m.trace() / 2)
kap = 4 * (M * Mb).trace() - 2 * M.trace() * Mb.trace()
M0 = M - cen(M) * I2
B0 = Mb - cen(Mb) * I2
ok = z(sp.expand(kap - 8 * cen(M0 * B0)))
# on bivectors
Bv1 = y1 * e[1] + y2 * e[2] + y3 * e[3]
u1, u2, u3 = sp.symbols("u1 u2 u3", real=True)
Bv2 = u1 * e[1] + u2 * e[2] + u3 * e[3]
ok = ok and z(4 * (Bv1 * Bv2).trace() - 8 * sp.re((Bv1 * Bv2).trace()) / 2)
ok = ok and Z(Bv1 * Bv1 + (y1**2 + y2**2 + y3**2) * I2)
ok = ok and z(Bv1.det() + (4 * (Bv1 * Bv1).trace()) / 8)
check("D13", "kappa(A,B) = 8<A' B'>_{0+3}, A' = grades 1+2 of A; on bivectors kappa = 8<AB>_0, A Abar = -A^2 = -kappa(A,A)/8", ok)

# D14 real Killing form on grades 1 + 2
Vv = x * s1 + y * s2 + w * s3
realk = lambda a, b: 2 * sp.re(4 * (a * b).trace())
ok = z(realk(Vv, Vv) - 16 * (x**2 + y**2 + w**2)) and z(realk(Bv1, Bv1) + 16 * (y1**2 + y2**2 + y3**2))
ok = ok and z(sp.expand(realk(Vv, Bv1)))
check("D14", "real Killing form 16<AB>_0: +16|x|^2 on vectors, -16|y|^2 on bivectors, cross term 0", ok)

# D15 even elements: reversion = Clifford conjugation
ce = [c[0], 0, 0, 0, c[4], c[5], c[6], 0]
Je = mv(ce)
ok = Z(rev(ce) - bar(ce)) and Z(Je * Je.H - Je.det() * I2)
check("D15", "on the even subalgebra reversion = Clifford conjugation, so J J~ = J Jbar is a scalar", ok)

# D16 minimal left ideals
Pp = (I2 + s3) / 2
Pm = (I2 - s3) / 2
ok = Z(Pp * Pp - Pp) and Z(Pm * Pm - Pm) and Z(Pp * Pm) and Z(Pp + Pm - I2)
ok = ok and Z((Mb * (M * Pp)) * Pm) and Z((M * Pp)[:, 1])
check("D16", "P+- = (1 +- sigma_3)/2 are complementary idempotents; Cl P+ is a left ideal (one column)", ok)

# D17 Bell closure in channel units
kron = sp.kronecker_product
rho2 = (kron(I2, I2) + sum((kron(e[j], e[j]) for j in (1, 2, 3)), sp.zeros(4, 4))) / 4
a = sp.symbols("a1:4", real=True)
b = sp.symbols("b1:4", real=True)
av = sum((a[j] * S[j] for j in range(3)), sp.zeros(2, 2))
bv = sum((b[j] * S[j] for j in range(3)), sp.zeros(2, 2))
E = (kron(av, bv) * rho2).trace()
ok = z(sp.expand(E + sum(a[j] * b[j] for j in range(3))))
ok = ok and Z(rho2 - (kron(I2, I2) - sum((kron(s, s) for s in S), sp.zeros(4, 4))) / 4)
check("D17", "joint closure (1 + sum e_j x e_j)/4 = (1 - sum sigma_j x sigma_j)/4; E(a,b) = -a.b", ok)

# D18 unitary and unit-norm elements
ok = Z(M * rev(c) - M * M.H) and Z(M * bar(c) - M.det() * I2)
check("D18", "U U~ = 1 is unitarity; G Gbar = 1 is det G = 1", ok)

bad = 0
for label, text, ok in results:
    print(("PASS" if ok else "FAIL"), label, text)
    bad += 0 if ok else 1
print("%d of %d passed" % (len(results) - bad, len(results)))
sys.exit(1 if bad else 0)
