"""Square versus root of the area element, labels S1--S8.

Sheet X^mu(t, s) in Minkowski space, signature (-,+,+,+).
h_ab = dX_a . dX_b (induced metric), D = -det h.
Nambu-Goto integrand: sqrt(D).  Squared (Schild-type) integrand: D.
Exact symbolic checks; exits non-zero on any failure.
"""
import sys
import sympy as sp

results = []


def check(label, text, ok):
    results.append((label, text, bool(ok)))
    print(("PASS" if ok else "FAIL"), label, text, flush=True)


n = 4
eta = sp.diag(-1, 1, 1, 1)
p = sp.symbols("p0:4", real=True)   # dX/dt
q = sp.symbols("q0:4", real=True)   # dX/ds


def dot(a, b):
    return sum(eta[m, m] * a[m] * b[m] for m in range(n))


def Dof(a, b):
    return sp.expand(-(dot(a, a) * dot(b, b) - dot(a, b) ** 2))


D = Dof(p, q)

# S1 squared tangent bivector
Sig = [[p[m] * q[k] - q[m] * p[k] for k in range(n)] for m in range(n)]
SS = sum(eta[m, m] * eta[k, k] * Sig[m][k] ** 2 for m in range(n) for k in range(n))
check("S1", "Sigma^{mu nu} Sigma_{mu nu} = 2 det h = -2 D, Sigma = dX_t ^ dX_s", sp.expand(SS + 2 * D) == 0)

# S2 reparametrization weight
a, b, c, d = sp.symbols("a b c d", real=True)
p2 = [a * p[m] + b * q[m] for m in range(n)]
q2 = [c * p[m] + d * q[m] for m in range(n)]
check("S2", "under a change of sheet coordinates with Jacobian M, D -> (det M)^2 D",
      sp.expand(Dof(p2, q2) - (a * d - b * c) ** 2 * D) == 0)
jac = sp.symbols("j", positive=True)
Dp = sp.symbols("Dp", positive=True)
check("S3", "so sqrt(D) d^2s is invariant for every Jacobian, D d^2s only when |det M| = 1",
      sp.simplify(sp.sqrt(jac ** 2 * Dp) / jac - sp.sqrt(Dp)) == 0
      and sp.simplify(jac ** 2 * Dp / jac - Dp) != 0
      and sp.simplify((jac ** 2 * Dp / jac - Dp).subs(jac, 1)) == 0)

# S4 auxiliary density
e = sp.symbols("e", positive=True)
L = Dp / e + e
sol = sp.solve(sp.diff(L, e), e)
check("S4", "D/e + e is stationary in e at e = sqrt(D), where it equals 2 sqrt(D)",
      sol == [sp.sqrt(Dp)] and sp.simplify(L.subs(e, sol[0]) - 2 * sp.sqrt(Dp)) == 0)

# momenta, as functions of the first derivatives
Pt = [sp.diff(D, p[m]) for m in range(n)]
Ps = [sp.diff(D, q[m]) for m in range(n)]
check("S5", "dD/d(dX_a) . dX_c = 2 D delta^a_c (D is a determinant of the two tangent vectors)",
      sp.expand(sum(Pt[m] * p[m] for m in range(n)) - 2 * D) == 0
      and sp.expand(sum(Ps[m] * q[m] for m in range(n)) - 2 * D) == 0
      and sp.expand(sum(Pt[m] * q[m] for m in range(n))) == 0
      and sp.expand(sum(Ps[m] * p[m] for m in range(n))) == 0)

# a generic sheet
t, s = sp.symbols("t s", real=True)
X = [sp.Function("X%d" % m)(t, s) for m in range(n)]
sub = {}
for m in range(n):
    sub[p[m]] = sp.diff(X[m], t)
    sub[q[m]] = sp.diff(X[m], s)
Df = D.subs(sub)
Ptf = [e_.subs(sub) for e_ in Pt]
Psf = [e_.subs(sub) for e_ in Ps]
EL_S = [sp.diff(Ptf[m], t) + sp.diff(Psf[m], s) for m in range(n)]

# S6 the squared action forces D constant on its solutions
okt = sp.expand(sum(EL_S[m] * sp.diff(X[m], t) for m in range(n)) - sp.diff(Df, t)) == 0
oks = sp.expand(sum(EL_S[m] * sp.diff(X[m], s) for m in range(n)) - sp.diff(Df, s)) == 0
check("S6", "EL_S . dX_c = d_c D identically: on solutions of the squared action, D is constant on the sheet",
      okt and oks)

# S7 relation between the two sets of equations of motion, and the Noether identity of the root
# EL_NG = d_a( P^a / (2 sqrt D) ), expanded by the product rule with d_a D taken from the sheet
EL_NG_times = [2 * Df * EL_S[m] - sp.diff(Df, t) * Ptf[m] - sp.diff(Df, s) * Psf[m] for m in range(n)]
# independent computation by direct differentiation, on a concrete non-trivial sheet
Xc = [t + s ** 2 / 5, sp.cos(s) * sp.cos(t) + t * s / 7, sp.cos(s) * sp.sin(t), s ** 3 / 3 + t ** 2 / 11]
csub = {X[m]: Xc[m] for m in range(n)}
Dc = Df.subs(csub).doit()
ok = True
for m in range(n):
    direct = sp.diff(Ptf[m].subs(csub).doit() / (2 * sp.sqrt(Dc)), t) + sp.diff(Psf[m].subs(csub).doit() / (2 * sp.sqrt(Dc)), s)
    claim = EL_NG_times[m].subs(csub).doit() / (4 * Dc ** sp.Rational(3, 2))
    pt = {t: sp.Rational(3, 10), s: sp.Rational(7, 10)}
    ok &= abs(sp.N((direct - claim).subs(pt), 40)) < sp.Float(10) ** -30
noe_t = sp.expand(sum(EL_NG_times[m] * sp.diff(X[m], t) for m in range(n))) == 0
noe_s = sp.expand(sum(EL_NG_times[m] * sp.diff(X[m], s) for m in range(n))) == 0
check("S7", "4 D^(3/2) EL_NG = 2 D EL_S - d_a D dD/d(dX_a) (spot-checked by direct differentiation); "
            "EL_NG . dX_c = 0 identically (two of the root's equations are empty: reparametrization)",
      ok and noe_t and noe_s)

# S8 a known string: the rigidly rotating rod, X = (t, r cos t, r sin t, 0), r = r(s) any profile
r = sp.Function("r")(s)
Xr = [t, r * sp.cos(t), r * sp.sin(t), sp.Integer(0)]
rsub = {X[m]: Xr[m] for m in range(n)}
Dr = sp.simplify(Df.subs(rsub).doit())
ng = [sp.simplify(EL_NG_times[m].subs(rsub).doit()) for m in range(n)]
sch = [sp.simplify(EL_S[m].subs(rsub).doit()) for m in range(n)]
rp, rpp = sp.diff(r, s), sp.diff(r, s, 2)
ng_all = all(x == 0 for x in ng)
sch_generic = any(x != 0 for x in sch)
# impose D constant: d/ds[(1 - r^2) r'^2] = 0  <=>  r'' = r r'^2 / (1 - r^2)
sch_const = [sp.simplify(x.subs(rpp, r * rp ** 2 / (1 - r ** 2))) for x in sch]
conf = [sp.simplify(x.subs(r, sp.cos(s)).doit()) for x in sch]
check("S8", "rotating rod: D = (1 - r^2) r'^2; root equations hold for every profile r(s); squared equations fail "
            "for a generic profile (for example r = cos s) and hold exactly when D is constant",
      sp.simplify(Dr - (1 - r ** 2) * rp ** 2) == 0 and ng_all and sch_generic
      and all(x == 0 for x in sch_const) and any(x != 0 for x in conf))

bad = sum(1 for _, _, ok_ in results if not ok_)
print("%d of %d passed" % (len(results) - bad, len(results)))
sys.exit(1 if bad else 0)
