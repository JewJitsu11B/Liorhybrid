"""Checks written for the canonicalization sheet, labels G1--G4. Exact sympy; exits non-zero on failure.

G1  corrected Thm feedback (3): the two-sided impulse response solves the update rule exactly
G2  flat weights at k = 0: the companion matrix of the two-step recurrence has det 1, trace 1 + alpha,
    so it is elliptic / parabolic / hyperbolic in SL(2,R) as alpha < 1, = 1, > 1
G3  Thm unitary, long-wavelength exponent: -(i + sqrt(eps) - (pi/4) eps) kappa t - (pi/4) eps t
G4  diffusive ratio: (1/2) sqrt(eps) (1 + sqrt(g)) - (pi/4) eps
"""
import sys
import sympy as sp

res = []


def check(label, text, ok):
    res.append(bool(ok))
    print(("PASS" if ok else "FAIL"), label, text)


# G1
ok = True
for a in (sp.Rational(3, 2), sp.Rational(5, 4), sp.Integer(2), sp.Rational(11, 10)):
    es = ((1 + a) + sp.sqrt((1 + a) ** 2 - 4)) / 2      # e^s, cosh s = (1+a)/2
    sh = (es - 1 / es) / 2

    def J(n):                                            # impulse at n0 = 0
        if n <= 0:
            return -(1 - 1 / es) * es ** n / (2 * sh)
        return (1 - 1 / es) * es ** (-(n - 1)) / (2 * sh)

    for n in range(-6, 8):
        mem = sum(J(n - j) for j in range(1, 30) if n - j > 0)
        m0 = min(n - 1, 0)
        mem += -es ** m0 / (2 * sh)                      # geometric tail over all indices <= m0
        r = sp.simplify(J(n + 1) - a * J(n) - (a - 1) * mem - (1 if n == 0 else 0))
        ok &= (r == 0)
    ok &= sp.simplify(J(0)) < 0 and sp.simplify(J(1)) > 0
check("G1", "impulse response -(1-e^-s)e^{-s(n0-n)}/(2 sinh s), n<=n0; +(1-e^-s)e^{-s(n-n0-1)}/(2 sinh s), n>n0", ok)

# G2
al, z = sp.symbols("alpha z", real=True)
M = sp.Matrix([[1 + al, -1], [1, 0]])
ok = sp.simplify(M.det() - 1) == 0 and sp.simplify(M.trace() - (1 + al)) == 0
ok &= sp.expand((z * sp.eye(2) - M).det() - (z ** 2 - (1 + al) * z + 1)) == 0
disc = sp.expand((1 + al) ** 2 - 4)                      # < 0 elliptic, = 0 parabolic, > 0 hyperbolic
ok &= sp.factor(disc) == (al - 1) * (al + 3)
ok &= disc.subs(al, sp.Rational(1, 2)) < 0 and disc.subs(al, 1) == 0 and disc.subs(al, sp.Rational(3, 2)) > 0
check("G2", "companion [[1+alpha,-1],[1,0]]: det 1, trace 1+alpha; |trace| < 2 iff -3 < alpha < 1", ok)

# G3, G4
eps, kap, t, g = sp.symbols("epsilon kappa t g", positive=True)
first = -sp.I * (1 - sp.sqrt(g)) * t - (sp.sqrt(eps) * (1 - g) / 2 + sp.pi / 4 * eps * sp.sqrt(g)) * t
lw = first.subs(g, (1 - kap) ** 2)                       # sqrt(g) = 1 - kappa, kappa = (hbar/2m) k^2
lw1 = sp.series(lw, kap, 0, 2).removeO()
claim = -(sp.I + sp.sqrt(eps) - sp.pi / 4 * eps) * kap * t - sp.pi / 4 * eps * t
check("G3", "long-wavelength exponent keeps the -(pi/4) eps term", sp.simplify(sp.expand(lw1 - claim)) == 0)
ratio = (sp.sqrt(eps) * (1 - g) / 2 + sp.pi / 4 * eps * sp.sqrt(g) - sp.pi / 4 * eps) / (1 - sp.sqrt(g))
check("G4", "diffusive ratio (1/2) sqrt(eps)(1 + sqrt g) - (pi/4) eps",
      sp.simplify(ratio - (sp.sqrt(eps) * (1 + sp.sqrt(g)) / 2 - sp.pi / 4 * eps)) == 0)

print("%d of %d passed" % (sum(res), len(res)))
sys.exit(0 if all(res) else 1)
