# Agent E: exact and symbolic checks of the alpha = 2 paper. Independent code (sympy, mpmath).
# Prints one PASS|FAIL line per check, exits non-zero if any check fails.
import sys, itertools, time
import sympy as sp
import mpmath as mp

mp.mp.dps = 30
FAILS = []
NOTES = []


def check(label, cond, text):
    ok = bool(cond)
    print(("PASS " if ok else "FAIL ") + label + " :: " + text, flush=True)
    if not ok:
        FAILS.append(label + " :: " + text)


def note(text):
    NOTES.append(text)
    print("NOTE " + text, flush=True)


def z(e):
    return sp.simplify(e) == 0


T0 = time.time()
I = sp.I
pi = sp.pi

# ---------------------------------------------------------------- Def profile
a, k, r, y, x = sp.symbols('a k r y x', positive=True)
yy = sp.symbols('yy', real=True)
ft1 = sp.integrate(sp.exp(-I * k * yy) / (2 * a), (yy, -a, a))
check("Def profile d=1", z(ft1.rewrite(sp.sin) - sp.sin(k * a) / (k * a)), "FT of uniform density on [-c tau, c tau] is sinc")
ft3 = (4 * pi / k) * sp.integrate(r * sp.DiracDelta(r - a) / (4 * pi * a**2) * sp.sin(k * r), (r, 0, sp.oo))
check("Def profile d=3", z(ft3 - sp.sin(k * a) / (k * a)), "radial FT of delta(|y|-c tau)/(4 pi c^2 tau^2) is sinc")
mass3 = sp.integrate(4 * pi * r**2 * sp.DiracDelta(r - a) / (4 * pi * a**2), (r, 0, sp.oo))
check("Def profile d=3 mass", z(mass3 - 1), "shell measure has total mass 1")
mass2 = sp.integrate(2 * pi * r / (2 * pi * a * sp.sqrt(a**2 - r**2)), (r, 0, a))
check("Def profile d=2 mass", z(mass2 - 1), "d=2 density has total mass 1")
ok = True
for j in range(0, 9):
    # termwise: J0(kr) = sum (-1)^j (kr/2)^(2j)/(j!)^2
    mom = sp.integrate(x**(2 * j + 1) / sp.sqrt(1 - x**2), (x, 0, 1))  # r = a x
    lhs = sp.Rational((-1)**j, sp.factorial(j)**2 * 4**j) * mom
    rhs = sp.Rational((-1)**j, sp.factorial(2 * j + 1))
    ok = ok and sp.simplify(lhs - rhs) == 0
check("Def profile d=2", ok, "Hankel transform of d=2 density equals sinc, power series terms j=0..8")
ka_ = mp.mpf('2.1') * mp.mpf('1.3')  # r = a sin(theta) removes the endpoint singularity
v2 = mp.quad(lambda th_: mp.besselj(0, ka_ * mp.sin(th_)) * mp.sin(th_), [0, mp.pi / 2])
check("Def profile d=2 numeric", abs(v2 - mp.sin(ka_) / ka_) < mp.mpf(10)**-25, "quadrature at k=2.1, c tau=1.3, 30 digits")
B = sp.symbols('B', positive=True)
proj = sp.integrate(1 / sp.sqrt(B**2 - yy**2), (yy, -B, B))  # B^2 = a^2 - y1^2
check("Def profile projection", z(proj / (2 * pi * a) - 1 / (2 * a)), "d=2 density projects to the uniform d=1 density (hat-box)")
th = sp.symbols('theta', positive=True)
# sphere -> plane: two hemispheres, dA_sphere = dA_plane / cos(theta), cos = sqrt(a^2-r^2)/a
check("Def profile projection 3->2", z(2 / (4 * pi * a**2) * a / sp.sqrt(a**2 - r**2) - 1 / (2 * pi * a * sp.sqrt(a**2 - r**2))), "sphere projects to the d=2 density")

# ---------------------------------------------------------------- Def memory
al, T, tau, tc = sp.symbols('alpha T tau tau_c', positive=True)
bnd = sp.integrate(tau**(al - 1) / sp.gamma(al), (tau, 0, T))
check("Def memory bound", z(sp.gammasimp(bnd - T**al / sp.gamma(al + 1))), "int_0^T tau^(alpha-1)/Gamma(alpha) = T^alpha/Gamma(alpha+1)")
check("Def memory vs Axiom order2", z((tau**(al - 1) / sp.gamma(al)).subs(al, 2) - tau), "at alpha=2 the lag weight is tau, kernel tau_c^-2 tau nu_tau")

# ---------------------------------------------------------------- Prop symbol
s, b = sp.symbols('s b', positive=True)
closed = ((s - I * b)**(1 - al) - (s + I * b)**(1 - al)) / (2 * I * b * (al - 1))
# Own derivation: expand sinc in powers of b and integrate termwise (valid for 0 < b < s, all alpha > 0,
# dominated by e^{-s tau} sinh(b tau)/(b tau)); then continue analytically in b and s.
NJ = 7
own = sum((-1)**j * b**(2 * j) / sp.factorial(2 * j + 1) * sp.rf(al, 2 * j) * s**(-al - 2 * j) for j in range(NJ))
ser = sp.series(closed, b, 0, 2 * NJ).removeO()
diff = sp.simplify(sp.expand_func(sp.expand((ser - own) * s**al)))
check("Prop symbol series", diff == 0, "closed form = termwise Gamma integrals, coefficients b^0..b^12, symbolic alpha")
check("Prop symbol k=0", z(sp.limit(closed, b, 0) - s**(-al)), "limit b->0 is s^-alpha")
m2 = sp.simplify(closed.subs(al, 2))
check("Thm lorentz m_2", z(m2 - 1 / (s**2 + b**2)), "closed form at alpha=2 is 1/(s^2+c^2k^2)")
d2 = sp.integrate(tau * sp.exp(-s * tau) * sp.sin(b * tau) / (b * tau), (tau, 0, sp.oo))
check("Cor green symbol", z(d2 - 1 / (s**2 + b**2)), "direct Laplace integral of tau sinc(b tau) is 1/(s^2+b^2): sign +, normalization 1")
d3 = sp.integrate(tau**2 / 2 * sp.exp(-s * tau) * sp.sin(b * tau) / (b * tau), (tau, 0, sp.oo))
check("Prop symbol alpha=3 direct", z(d3 - closed.subs(al, 3)), "direct integral at alpha=3 equals closed form")


def m_closed(alv, sv, bv):
    if alv == 1:
        return mp.atan(bv / sv) / bv if mp.im(sv) == 0 else (mp.log(sv + 1j * bv) - mp.log(sv - 1j * bv)) / (2j * bv)
    return ((sv - 1j * bv)**(1 - alv) - (sv + 1j * bv)**(1 - alv)) / (2j * bv * (alv - 1))


def m_quad(alv, sv, bv):
    f = lambda t: t**(alv - 1) / mp.gamma(alv) * mp.exp(-sv * t) * (mp.sin(bv * t) / (bv * t))
    # tau = x^(1/alpha) on [0,1] removes the tau^(alpha-1) endpoint singularity
    g = lambda x_: (1 / alv) / mp.gamma(alv) * mp.exp(-sv * x_**(1 / alv)) * mp.sinc(bv * x_**(1 / alv))
    Tm = int(90 / mp.re(sv)) + 2
    return mp.quad(g, [0, 1]) + mp.quad(f, mp.linspace(1, Tm, 4 * (Tm - 1) + 1))


worst = 0
for alv in [mp.mpf('0.3'), mp.mpf('0.5'), mp.mpf(1), mp.mpf('1.5'), mp.mpf(2), mp.mpf('2.5'), mp.mpf(3)]:
    for sv, bv in [(mp.mpf('0.8'), mp.mpf('0.6')), (mp.mpc('0.5', '1.2'), mp.mpf(2)), (mp.mpf('0.7'), mp.mpf(3))]:
        e = abs(m_quad(alv, sv, bv) - m_closed(alv, sv, bv))
        worst = max(worst, e)
check("Prop symbol quadrature", worst < mp.mpf(10)**-25, "definition vs closed form, alpha in {0.3,0.5,1,1.5,2,2.5,3}, real and complex s, b<s and b>s; max abs diff %s" % mp.nstr(worst, 3))
note("Prop symbol: the paper's proof uses int tau^(beta-1) e^(-z tau) = Gamma(beta) z^(-beta) with beta = alpha-1; that integral diverges for 0<alpha<=1 (beta<=0). The formula still holds there (checked by series and by quadrature at alpha=0.3, 0.5); the proof needs analytic continuation in alpha or the termwise series used here.")
worst = 0
for alv in [mp.mpf('0.4'), mp.mpf('1.7'), mp.mpf('2.6'), mp.mpf(1)]:
    for Rv, ph in [(mp.mpf('1.3'), mp.mpf('0.2')), (mp.mpf('0.7'), mp.mpf('1.5'))]:
        sv, bv = Rv * mp.cos(ph), Rv * mp.sin(ph)
        h = ph / mp.sin(ph) if alv == 1 else mp.sin((alv - 1) * ph) / ((alv - 1) * mp.sin(ph))
        worst = max(worst, abs(m_closed(alv, sv, bv) - Rv**(-alv) * h))
check("Prop symbol polar form", worst < mp.mpf(10)**-25, "R^-alpha h_alpha(phi) equals closed form (incl. alpha=1: phi/sin phi)")
lim1 = (mp.log(mp.mpf('0.8') + 0.6j) - mp.log(mp.mpf('0.8') - 0.6j)) / (2j * 0.6)
check("Prop symbol alpha=1", abs(m_closed(mp.mpf(1) + mp.mpf(10)**-12, mp.mpf('0.8'), mp.mpf('0.6')) - mp.atan(mp.mpf('0.6') / mp.mpf('0.8')) / mp.mpf('0.6')) < mp.mpf(10)**-10 and abs(lim1 - mp.atan(0.75) / 0.6) < 1e-15, "alpha->1 limit is arctan(b/s)/b")

# ---------------------------------------------------------------- Thm lorentz
ph = sp.symbols('varphi', positive=True)
h = sp.sin((al - 1) * ph) / ((al - 1) * sp.sin(ph))
hs = sp.series(h, ph, 0, 8).removeO()
c2 = sp.factor(hs.coeff(ph, 2))
c4 = sp.factor(hs.coeff(ph, 4))
c6 = sp.factor(hs.coeff(ph, 6))
check("Thm lorentz leading defect", z(c2 - al * (2 - al) / 6), "phi^2 coefficient is alpha(2-alpha)/6")
c4_expected = al * (2 - al) * (4 + 6 * al - 3 * al**2) / 360
check("Thm lorentz next term", z(c4 - c4_expected), "phi^4 coefficient is alpha(2-alpha)(4+6alpha-3alpha^2)/360 = (1-beta^2)(7-3beta^2)/360, beta=alpha-1")
print("     phi^4 coeff:", c4, "| phi^6 coeff:", c6)
alr = sp.Symbol('alpha_r', real=True)
check("Thm lorentz equivalence", set(sp.solve(alr * (2 - alr), alr)) == {0, 2} and sp.simplify(h.subs(al, 2)) == 1 and sp.simplify(h.subs(al, 0)) == 1, "phi^2 coefficient vanishes only at alpha=0,2; h_2 = h_0 = 1 identically")
h1 = sp.series(ph / sp.sin(ph), ph, 0, 6).removeO()
check("Thm lorentz alpha=1", z(h1 - (1 + ph**2 / 6 + sp.Rational(7, 360) * ph**4)) and z(c2.subs(al, 1) - sp.Rational(1, 6)) and z(c4.subs(al, 1) - sp.Rational(7, 360)), "h_1 = phi/sin phi = 1 + phi^2/6 + 7 phi^4/360; not constant; expansion continuous at alpha=1")
check("Thm lorentz symmetry", z(h - h.subs(al, 2 - al)), "h_alpha = h_(2-alpha)")
check("Thm lorentz alpha->0", z(sp.simplify(closed.subs(al, 0)) - 1), "closed form at alpha=0 is 1 (identity)")
okp, okn = True, True
for i in range(1, 400):
    av = mp.mpf(i) / 40  # 0.025 .. 9.975
    if av in (1, 2):
        continue
    for jx in range(1, 60):
        pv = mp.pi / 2 * jx / 60
        hv = mp.sin((av - 1) * pv) / ((av - 1) * mp.sin(pv))
        if av < 2 and not hv > 1:
            okp = False
        if av > 2 and not hv < 1:
            okn = False
check("Thm lorentz defect sign", okp and okn, "h>1 for 0<alpha<2 and h<1 for 2<alpha<10 on a 400 x 59 grid of (alpha, phi) in (0, pi/2)")

# ---------------------------------------------------------------- Cor green
c, t = sp.symbols('c t', positive=True)
check("Cor green G3", z((t * sp.DiracDelta(r - c * t) / (4 * pi * c**2 * t**2)) / sp.DiracDelta(r - c * t) - 1 / (4 * pi * c**2 * t)), "t nu_t in d=3 is delta(|x|-ct)/(4 pi c^2 t)")
check("Cor green G2 G1", z(t / (2 * pi * c * t * sp.sqrt(c**2 * t**2 - r**2)) - 1 / (2 * pi * c * sp.sqrt(c**2 * t**2 - r**2))) and z(t / (2 * c * t) - 1 / (2 * c)), "t nu_t in d=2 and d=1 match the printed forms")
# standard Kirchhoff form delta(t - r/c)/(4 pi c^2 r): delta(t-r/c) = c delta(r-ct); on the shell r = ct
check("Cor green Kirchhoff", z(c / (4 * pi * c**2 * (c * t)) - 1 / (4 * pi * c**2 * t)), "equals delta(t-r/c)/(4 pi c^2 r), the retarded Green's function of d_t^2 - c^2 Laplacian")
kk = sp.symbols('kappa', positive=True)
tp = sp.symbols('tp', positive=True)
g = lambda u_: u_**2  # source g(t) e^{ik.x}, zero past, t>0
conv = sp.integrate(sp.sin(c * kk * tp) / (c * kk) * g(t - tp), (tp, 0, t))
check("Cor green box(G*f)=f", z(sp.diff(conv, t, 2) + c**2 * kk**2 * conv - g(t)), "(d_t^2 + c^2 k^2) of G*f returns f for f = t^2 e^{ik.x}, any d")

# ---------------------------------------------------------------- Prop nulllegs
u, v = sp.symbols('u v', positive=True)
tt, xx = sp.symbols('t x', real=True)
U, V, pp_, qq_ = sp.symbols('U V p_ q_')
okop = True
for Ftest in [sp.exp(pp_ * U + qq_ * V), U**3 * V**2 * sp.sin(U * V), sp.log(1 + U**2) * sp.cosh(V) + U * V]:
    sub = {U: c * tt - xx, V: c * tt + xx}
    expr = Ftest.subs(sub)
    box = sp.diff(expr, tt, 2) - c**2 * sp.diff(expr, xx, 2)
    okop = okop and sp.simplify(box - 4 * c**2 * sp.diff(Ftest, U, V).subs(sub)) == 0
check("Prop nulllegs operator", okop, "box = 4 c^2 d_u d_v on exp(pu+qv) (all p,q) and two other test functions")
jac = sp.Matrix([[c, -1], [c, 1]]).det()
check("Prop nulllegs Jacobian", jac == 2 * c, "du dv = 2c dt dx")
# direct (t,x) convolution against G1 = theta/(2c) versus the null double integral, at a rational point
cv, tv, xv = sp.Integer(2), sp.Integer(1), sp.Rational(1, 4)
uv_, vv_ = cv * tv - xv, cv * tv + xv
ta, tb = uv_ / (2 * cv), vv_ / (2 * cv)
t1, x1 = sp.symbols('t1 x1', real=True)
okk = True
for p, q in [(0, 0), (1, 2), (3, 1)]:
    f = (cv * t1 - x1)**p * (cv * t1 + x1)**q  # times theta(u')theta(v')
    lo1, hi1 = -cv * t1, cv * t1
    lo2 = xv - cv * (tv - t1)
    hi2 = xv + cv * (tv - t1)
    first, second = min(ta, tb), max(ta, tb)
    # ta < tb here (x > 0): for t1<ta full width; ta<t1<tb lower bound from the target cone; t1>tb both
    tot = sp.integrate(sp.integrate(f, (x1, lo1, hi1)), (t1, 0, ta)) \
        + sp.integrate(sp.integrate(f, (x1, lo2, hi1)), (t1, ta, tb)) \
        + sp.integrate(sp.integrate(f, (x1, lo2, hi2)), (t1, tb, tv))
    direct = tot / (2 * cv)
    up, vp = sp.symbols('up vp', positive=True)
    null = sp.integrate(sp.integrate(up**p * vp**q, (up, 0, uv_)), (vp, 0, vv_)) / (4 * cv**2)
    okk = okk and sp.simplify(direct - null) == 0
check("Prop nulllegs integral", okk, "direct (t,x) convolution with theta/(2c) equals (4c^2)^-1 int int du' dv' for three test sources at c=2, t=1, x=1/4")

# ---------------------------------------------------------------- Prop descent
rr, kap = sp.symbols('r kappa', positive=True)
okd = True
for f1, f1hat in [(sp.exp(-rr**2 / 2), sp.sqrt(2 * pi) * sp.exp(-kap**2 / 2)), (sp.exp(-rr), 2 / (1 + kap**2))]:
    D = -sp.diff(f1, rr) / (2 * pi * rr)
    f3hat = (4 * pi / kap) * sp.integrate(rr * D * sp.sin(kap * rr), (rr, 0, sp.oo))
    okd = okd and z(f3hat - f1hat)
check("Prop descent", okd, "3D FT of -(2 pi r)^-1 f1'(r) equals 1D FT of f1 for f1 = exp(-x^2/2) and exp(-|x|)")
# descent of G1 = theta(ct-|x|)/(2c): f1' = -delta(r-ct)/(2c)
check("Prop descent G", z(-(-1 / (2 * c)) / (2 * pi * rr) - 1 / (4 * pi * c * rr)), "D G1 = delta(r-ct)/(4 pi c r) = G3")

# ---------------------------------------------------------------- Prop checker (1) and Thm pathsum
mu = sp.symbols('mu', positive=True)
up, vp = sp.symbols('up vp', positive=True)
Gn = [1 / (2 * c)]  # coefficient functions of theta(u)theta(v)
for n in range(1, 7):
    prev = Gn[-1].subs({u: up, v: vp})
    Gn.append(sp.integrate(sp.integrate(prev, (up, 0, u)), (vp, 0, v)) / (4 * c**2))
okc = all(z(Gn[n] - (u * v / (4 * c**2))**n / (2 * c * sp.factorial(n)**2)) for n in range(7))
check("Prop checker (1) terms", okc, "G^{*(n+1)} = theta theta/(2c) (uv/4c^2)^n/(n!)^2 for n=0..6 by iterated null integration")
check("Thm pathsum (3) telescoping", all(z(4 * c**2 * sp.diff(Gn[n], u, v) - Gn[n - 1]) for n in range(1, 7)), "box G^{*(n+1)} = G^{*n} inside the cone, n=1..6")
zz = sp.symbols('z', positive=True)
j0 = sp.series(sp.besselj(0, zz), zz, 0, 14).removeO()
i0 = sp.series(sp.besseli(0, zz), zz, 0, 14).removeO()
arg = mu * sp.sqrt(u * v) / c
sJ = sum((-mu**2)**n * Gn[n] for n in range(7))
sI = sum((mu**2)**n * Gn[n] for n in range(7))
check("Prop checker (1) J0", z(sp.expand(sJ - j0.subs(zz, arg) / (2 * c))), "sum (-mu^2)^n G^{*(n+1)} matches J0(mu sqrt(t^2-x^2/c^2))/(2c) through order mu^12")
check("Prop checker (1) I0", z(sp.expand(sI - i0.subs(zz, arg) / (2 * c))), "absolute sum matches I0/(2c) through order mu^12")
check("Prop checker (1) argument", z((u * v).subs({u: c * t - x, v: c * t + x}) / c**2 - (t**2 - x**2 / c**2)), "uv/c^2 = t^2 - x^2/c^2")
# J0 form solves KG inside the cone: use Bessel's equation
w_ = sp.Function('w')
J0f = sp.besselj(0, mu * sp.sqrt(u * v) / c)
res = sp.simplify(4 * c**2 * sp.diff(J0f, u, v) + mu**2 * J0f)
resn = [abs(sp.N(res.subs({u: sp.Rational(3, 7), v: sp.Rational(11, 5), c: sp.Rational(3, 2), mu: sp.Rational(5, 3)}), 25)),
        abs(sp.N(res.subs({u: sp.Rational(9, 2), v: sp.Rational(1, 5), c: 2, mu: 3}), 25))]
check("Prop checker (1) KG", max(resn) < 1e-20, "(box + mu^2) J0(mu sqrt(uv)/c) = 0 inside the cone (evaluated at two rational points, residual %s)" % sp.N(max(resn), 3))
check("Prop checker (1) source", z(4 * c**2 * (1 / (2 * c)) / (2 * c) - 1), "4c^2 d_u d_v [theta(u)theta(v)/(2c)] = 2c delta(u)delta(v) = delta(t)delta(x)")
# spatial integrals: int dx G^{*(n+1)} = t^(2n+1)/(2n+1)!  (k=0 symbol s^-(2n+2))
oks = True
for n in range(0, 5):
    val = sp.integrate(Gn[n].subs({u: c * t - yy, v: c * t + yy}), (yy, -c * t, c * t))
    oks = oks and z(val - t**(2 * n + 1) / sp.factorial(2 * n + 1))
check("Thm pathsum (4)/(1) mass", oks, "int dx G^{*(n+1)}(t,x) = t^(2n+1)/(2n+1)! for n=0..4 (matches k=0 symbol and prod tau_i weights)")
# Riemann-Liouville bound
Fb = sp.Integer(1)
okb = True
for n in range(1, 6):
    Fb = sp.integrate(tau * Fb.subs(T, T - tau), (tau, 0, T))
    okb = okb and z(Fb - T**(2 * n) / sp.factorial(2 * n))
check("Thm pathsum (1) bound", okb, "n-fold iterate of int_0^T tau F(T-tau) on the constant 1 is T^(2n)/(2n)!, n=1..5 (bound is attained by S = const)")
check("Thm pathsum (1) cosh", z(sp.series(sp.cosh(mu * T), T, 0, 12).removeO() - sum(mu**(2 * n) * T**(2 * n) / sp.factorial(2 * n) for n in range(6))), "sum mu^(2n) T^(2n)/(2n)! is cosh(mu T) (series to T^10)")
Dd = sp.symbols('D', positive=True)
okg = all(z((Dd + mu**2) * sum((-mu**2)**n * Dd**(-n - 1) for n in range(Nn)) - (1 - (-mu**2 / Dd)**Nn)) for Nn in range(1, 9))
check("Thm pathsum (3) symbol", okg, "(D+mu^2) sum_{n<N} (-mu^2)^n D^-(n+1) = 1 - (-mu^2/D)^N, N=1..8: sum is 1/(D+mu^2) for |D|>mu^2 (symbol series converges only there; the kernel series converges everywhere)")
check("Thm KG (1) resummation", z(sp.Rational(1, 2) - sp.Rational(1, 2) * mu**2 / (Dd + mu**2) - 1 / (2 + 2 * mu**2 / Dd)), "symbol of J = S/2 - (mu^2/2) G_mu*S equals 1/(2 + tau_c^-2/D)")

# ---------------------------------------------------------------- Prop descent3
w = c**2 * t**2 - rr**2
ok3 = True
G3n = {}
for n in range(1, 7):
    g1 = (w / (4 * c**2))**n / (2 * c * sp.factorial(n)**2)
    g3 = -sp.diff(g1, rr) / (2 * pi * rr)
    G3n[n] = sp.simplify(g3)
    ok3 = ok3 and z(g3 - n * w**(n - 1) / (2 * pi * c * (4 * c**2)**n * sp.factorial(n)**2))
check("Prop descent3 terms", ok3, "D applied to Prop checker(1): G^{*(n+1)} = theta n w^(n-1)/(2 pi c (4c^2)^n (n!)^2), n=1..6")
# independent: spatial FT of these obeys (d_t^2 + c^2 k^2) F_n = F_(n-1), F_0 = sin(ckt)/(ck), and F_n = n-fold time convolution
F = {0: sp.sin(c * kap * t) / (c * kap)}
okF = True
for n in range(1, 4):
    F[n] = sp.simplify((4 * pi / kap) * sp.integrate(rr * G3n[n] * sp.sin(kap * rr), (rr, 0, c * t)))
    okF = okF and z(sp.diff(F[n], t, 2) + c**2 * kap**2 * F[n] - F[n - 1]) and z(F[n].subs(t, 0)) and z(sp.diff(F[n], t).subs(t, 0))
check("Prop descent3 / Thm pathsum (4) symbol", okF, "3D FT of the n=1,2,3 terms solves (d_t^2+c^2k^2)F_n = F_(n-1) with zero data: symbol (s^2+c^2k^2)^-(n+1)")
F1d = sp.simplify(sp.integrate(sp.cos(kap * yy) * Gn[1].subs({u: c * t - yy, v: c * t + yy}), (yy, -c * t, c * t)))
check("Prop descent G^{*2}", z(F1d - F[1]), "1D FT of G1^{*2} equals 3D FT of G3^{*2} (same function of |k|)")
conv2 = sp.integrate(sp.sin(c * kap * tp) / (c * kap) * sp.sin(c * kap * (t - tp)) / (c * kap), (tp, 0, t))
check("Thm pathsum (4) n=1", z(conv2 - F[1]), "two-leg history integral (time convolution of tau sinc) equals FT of theta/(8 pi c^3)")
check("Prop descent3 mass", all(z(sp.integrate(4 * pi * rr**2 * G3n[n], (rr, 0, c * t)) - t**(2 * n + 1) / sp.factorial(2 * n + 1)) for n in range(1, 5)), "int d^3x G^{*(n+1)} = t^(2n+1)/(2n+1)!, n=1..4")
j1 = sp.series(sp.besselj(1, zz), zz, 0, 12).removeO()
i1 = sp.series(sp.besseli(1, zz), zz, 0, 12).removeO()
W = sp.symbols('W', positive=True)  # sqrt(w)
tailJ = sum((-mu**2)**n * n * W**(2 * n - 2) / (2 * pi * c * (4 * c**2)**n * sp.factorial(n)**2) for n in range(1, 7))
tailI = sum((mu**2)**n * n * W**(2 * n - 2) / (2 * pi * c * (4 * c**2)**n * sp.factorial(n)**2) for n in range(1, 7))
check("Prop descent3 J1 tail", z(sp.expand(tailJ - (-mu / (4 * pi * c**2)) * sp.expand(j1.subs(zz, mu * W / c) / W))), "sum_{n>=1} (-mu^2)^n G^{*(n+1)} = -mu J1(mu sqrt(w)/c)/(4 pi c^2 sqrt(w)), through mu^12")
check("Prop descent3 I1 bound", z(sp.expand(tailI - (mu / (4 * pi * c**2)) * sp.expand(i1.subs(zz, mu * W / c) / W))), "absolute tail sum is mu I1/(4 pi c^2 sqrt(w)), through mu^12")
J0r = sp.besselj(0, mu * sp.sqrt(w) / c) / (2 * c)
dd = sp.simplify(-sp.diff(J0r, rr) / (2 * pi * rr) + mu * sp.besselj(1, mu * sp.sqrt(w) / c) / (4 * pi * c**2 * sp.sqrt(w)))
check("Prop descent3 closed form", abs(sp.N(dd.subs({c: 2, t: 1, rr: sp.Rational(1, 3), mu: sp.Rational(7, 5)}), 20)) < 1e-15, "-(2 pi r)^-1 d_r [J0/(2c)] = -mu J1/(4 pi c^2 sqrt w) inside the cone; shell term from -d_r theta(ct-r) = +delta(r-ct)")

# ---------------------------------------------------------------- Thm KG
ss, E, hb, mm, kq = sp.symbols('s E hbar m k', positive=True)
Dsym = sp.symbols('Dsym')
solD = sp.solve(sp.Eq(2 + tc**-2 / Dsym, 0), Dsym)
check("Thm KG (3) root", solD == [-1 / (2 * tc**2)], "2 + tau_c^-2/(s^2+c^2k^2) = 0 iff s^2+c^2k^2 = -1/(2 tau_c^2) = -mu^2")
mu_def = 1 / (sp.sqrt(2) * tc)
check("Thm KG mass", z(mu_def * tc - 1 / sp.sqrt(2)) and z(mu_def**2 - 1 / (2 * tc**2)), "mu tau_c = 1/sqrt 2")
sE = -I * E / hb
disp = sp.expand(sE**2 + c**2 * kq**2 + mu**2)
check("Thm KG (3) dispersion", z(disp * hb**2 * (-1) - (E**2 - hb**2 * c**2 * kq**2 - hb**2 * mu**2)), "at s = -iE/hbar: E^2 = hbar^2 c^2 k^2 + (hbar mu)^2, so m c^2 = hbar mu (massive sign)")
# box sign convention: box = d_t^2 - c^2 Laplacian acting on e^{st+ikx}
ex = sp.exp(s * tt + I * kq * xx)
check("Thm KG box convention", z((sp.diff(ex, tt, 2) - c**2 * sp.diff(ex, xx, 2)) / ex - (s**2 + c**2 * kq**2)), "symbol of box = d_t^2 - c^2 d_x^2 is s^2 + c^2 k^2")
solT = sp.solve(sp.Eq(2 - tc**-2 / Dsym, 0), Dsym)
check("Remark phase tachyon", solT == [1 / (2 * tc**2)], "factor -1 on the memory: s^2+c^2k^2 = +mu^2, i.e. E^2 = hbar^2c^2k^2 - m^2c^4, growth rate s = sqrt(mu^2-c^2k^2) for c|k|<mu")
Ssym = sp.Function('S')(tt)
check("Thm KG (2) symbol", z((Dd + mu**2) * (sp.Rational(1, 2) - sp.Rational(1, 2) * mu**2 / (Dd + mu**2)) - Dd / 2), "symbol identity (D+mu^2) J^ = D S^/2")

# ---------------------------------------------------------------- Thm near2
om, bb, de = sp.symbols('omega b delta', positive=True)
alp = sp.symbols('alpha_', positive=True)
# on real timelike omega > b > 0, s = -i omega + 0: s -+ ib = (omega +- b) e^{-i pi/2}
Fd = alp + tc**(-alp) * sp.exp(I * pi * alp / 2) * ((om - bb)**(1 - alp) - (om + bb)**(1 - alp)) / (2 * bb)
check("Thm near2 symbol phase", z(Fd.subs(alp, 2) - (2 - tc**-2 / (om**2 - bb**2))), "alpha + (alpha-1) m_alpha on omega>c|k| is alpha + tau_c^-alpha e^{i pi alpha/2}[(omega-b)^(1-alpha)-(omega+b)^(1-alpha)]/(2b); phase e^{i alpha pi/2}; at alpha=2 it is 2 - tau_c^-2/(omega^2-b^2)")
subs0 = {tc: 1 / (sp.sqrt(2) * mu), om: sp.sqrt(bb**2 + mu**2)}
Fw = sp.diff(Fd, om).subs(alp, 2)
Fa = sp.diff(Fd, alp).subs(alp, 2)
om0 = sp.sqrt(bb**2 + mu**2)
check("Thm near2 dF/domega", z(Fw.subs(subs0) - 4 * om0 / mu**2), "omega-derivative of the symbol of T at the root is 4 omega/mu^2")
w1 = sp.simplify((-Fa / Fw).subs(subs0))
# real/imaginary parts with omega - b = P > 0 made explicit: mu^2 = P(P+2b), tau_c^-2 = 2 mu^2
Pp = sp.symbols('P', positive=True)
subsP = {tc: 1 / sp.sqrt(2 * Pp * (Pp + 2 * bb)), om: Pp + bb}
w1P = sp.expand_log(sp.expand((-Fa / Fw).subs(subsP)), force=True)
im1 = sp.simplify(sp.im(w1P))
re1 = sp.simplify(sp.re(w1P))
check("Thm near2 rate", z(im1 - pi * Pp * (Pp + 2 * bb) / (4 * (Pp + bb))), "first-order root shift: Im omega = (pi/4) delta mu^2/omega (delta = alpha-2; decay below 2, growth above)")
print("     Re d omega/d delta =", re1)
check("Thm near2 real shift", abs(sp.N(re1.subs({Pp: sp.Rational(1, 2), bb: sp.Rational(3, 4)}))) > 0.1, "Re omega also shifts at first order in delta (the paper's proof keeps only the phase change -i pi delta); it does not affect the stated Im rate")
# exact roots, 30 digits, from the definition-based closed form with principal branches
mp.mp.dps = 40


def Fnum(wv, alv, bv, tcv):
    sv = -1j * wv
    mval = ((sv - 1j * bv)**(1 - alv) - (sv + 1j * bv)**(1 - alv)) / (2j * bv * (alv - 1)) * tcv**(-alv)
    return alv + (alv - 1) * mval


bv, tcv = mp.mpf('0.75'), 1 / mp.sqrt(2)
w0 = mp.sqrt(bv**2 + 1)
w1n = complex(sp.N(w1.subs({bb: sp.Rational(3, 4), mu: 1}), 30))
# second-order coefficient, symbolic
Fww = sp.diff(Fd, om, 2).subs(alp, 2)
Faw = sp.diff(Fd, om, alp).subs(alp, 2)
Faa = sp.diff(Fd, alp, 2).subs(alp, 2)
w2 = (-(Faa / 2 + Faw * (-Fa / Fw) + Fww * (-Fa / Fw)**2 / 2) / Fw).subs(subs0)
w2n = sp.N(w2.subs({bb: sp.Rational(3, 4), mu: 1}), 30)
w1mp = mp.mpc(sp.N(sp.re(w1.subs({bb: sp.Rational(3, 4), mu: 1})), 35), sp.N(sp.im(w1.subs({bb: sp.Rational(3, 4), mu: 1})), 35))
w2mp = mp.mpc(sp.N(sp.re(w2n), 30), sp.N(sp.im(w2n), 30))
okr, ok2 = True, True
rows = []
for dv in [mp.mpf('1e-2'), mp.mpf('1e-3'), mp.mpf('1e-4'), mp.mpf('1e-6'), mp.mpf('-1e-6'), mp.mpf('-1e-4'), mp.mpf('-1e-3'), mp.mpf('-1e-2')]:
    root = mp.findroot(lambda wv: Fnum(wv, 2 + dv, bv, tcv), w0 + w1mp * dv)
    pred = mp.pi / 4 * dv / w0
    ratio = mp.im(root) / pred
    rows.append((dv, root, ratio))
    okr = okr and abs(ratio - 1) < 5 * abs(dv) + mp.mpf(10)**-20
    ok2 = ok2 and abs(root - (w0 + w1mp * dv + w2mp * dv**2)) < 50 * abs(dv)**3
    # negative-frequency partner
    rootm = mp.findroot(lambda wv: Fnum(wv, 2 + dv, bv, tcv), -w0 + (-mp.conj(w1mp)) * dv)
    okr = okr and abs(rootm + mp.conj(root)) < mp.mpf(10)**-25
for dv, root, ratio in rows:
    print("     delta=%s  omega=%s  Im/pred=%s" % (mp.nstr(dv, 3), mp.nstr(root, 15), mp.nstr(ratio, 12)))
check("Thm near2 exact roots", okr, "exact dispersion roots (b=0.75, mu=1): Im omega/[(pi/4) delta mu^2/omega0] = 1 + O(delta) for delta = +-1e-2..1e-6; negative-frequency root is -conj(omega), same Im")
check("Thm near2 second order", ok2, "roots agree with omega0 + w1 delta + w2 delta^2 to O(delta^3); w1 = %s, w2 = %s" % (mp.nstr(w1mp, 12), mp.nstr(w2mp, 12)))
print("     relative correction to the rate: Im(w2)/Im(w1) =", mp.nstr(mp.im(w2mp) / mp.im(w1mp), 12), "per unit delta at b=0.75, mu=1")
mp.mp.dps = 30

# ---------------------------------------------------------------- Prop checker (2),(3)
okv = True
aS = sp.symbols('aS0:5', positive=True)
A = sp.symbols('A', positive=True)
for n in range(0, 5):
    # volume of {a_0..a_n > 0, sum = A}: integrate a_1..a_n over the simplex
    vol = sp.Integer(1)
    rem = A
    for j in range(n):
        pass
    # iterative formula: V_n(A) = int_0^A V_(n-1)(A - a) da
    Vn = sp.Integer(1)
    for j in range(n):
        Vn = sp.integrate(Vn.subs(A, A - aS[j]), (aS[j], 0, A))
    okv = okv and z(Vn - A**n / sp.factorial(n))
check("Prop checker (2) leg volume", okv, "n+1 positive leg durations with fixed sum A have volume A^n/n!, n=0..4")
check("Prop checker (2) identity", all(sp.expand((I * mu)**(2 * n + 1) / (2 * I * c * mu) - (-mu**2)**n / (2 * c)) == 0 for n in range(0, 9)), "(i mu)^(2n+1)/(2 i c mu) = (-mu^2)^n/(2c), n=0..8")
Phi = lambda n: (I * mu)**(2 * n + 1) * (u * v / (4 * c**2))**n / sp.factorial(n)**2
check("Prop checker (2) term by term", all(z((-mu**2)**n * Gn[n] - Phi(n) / (2 * I * c * mu)) for n in range(7)), "(-mu^2)^n G^{*(n+1)} = Phi_(2n+1)/(2 i c mu) with Phi = (i mu)^(2n+1) (v/2c)^n (u/2c)^n/(n!)^2, n=0..6")
# n=0 with the endpoint delta done explicitly: one corner at t1, x = c(2 t1 - t)
t1s = sp.symbols('t1s', real=True)
dens0 = sp.integrate(I * mu * sp.DiracDelta(xx - c * (2 * t1s - tt)), (t1s, -sp.oo, sp.oo))
check("Prop checker (2) normalization", z(dens0 - I * mu / (2 * c)), "one-corner density in x is i mu/(2c): the divisor 2 i c mu = (Jacobian 2c of x vs corner time) x (one corner i mu); Phi as printed is the leg-duration volume, not a density in x")

# exact corner counting on small lattices (brute force over all 2^N paths)
from math import comb
from collections import Counter
okl, okboth = True, True
ncls = 0
for N in range(2, 17):
    cnt = Counter()
    for bits in range(1 << N):
        steps = [(bits >> i) & 1 for i in range(N)]  # 1 = right
        NR = sum(steps)
        corners = sum(1 for i in range(N - 1) if steps[i] != steps[i + 1])
        cnt[(NR, corners, steps[0], steps[-1])] += 1
    for NR in range(0, N + 1):
        NL = N - NR
        for n in range(0, N):
            got = cnt.get((NR, 2 * n + 1, 1, 0), 0)
            exp = comb(NR - 1, n) * comb(NL - 1, n) if NR >= 1 and NL >= 1 else 0
            ncls += 1
            if got != exp:
                okl = False
            got2 = cnt.get((NR, 2 * n + 1, 0, 1), 0)
            tot_odd = sum(val for key, val in cnt.items() if key[0] == NR and key[1] == 2 * n + 1)
            if got2 != exp or tot_odd != 2 * exp:
                okboth = False
check("Prop checker (3) corner count", okl, "paths with N_R right, N_L left steps, first step right, last left, 2n+1 corners: C(N_R-1,n) C(N_L-1,n); all 2^N paths, N=2..16 (%d classes)" % ncls)
check("Prop checker (3) orientation", okboth, "the mirror class (first left, last right) has the same count, so ALL corner-odd paths number 2 C C: the paper's L_eps is one fixed (first,last) orientation, not every corner-odd path")
# continuum limit, symbolic
eps = sp.symbols('epsilon', positive=True)
okL, okO = True, True
for n in range(0, 6):
    Nsym = v / (2 * eps)
    binom = sp.ff(Nsym - 1, n) / sp.factorial(n)
    e = sp.expand(eps**n * binom)
    okL = okL and z(e.subs(eps, 0) - (v / 2)**n / sp.factorial(n))
    okO = okO and z(sp.diff(e, eps).subs(eps, 0) - (-(v / 2)**n / sp.factorial(n) * n * (n + 1) / v))
    # domination: C(N-1,n) <= N^n/n!
check("Prop checker (3) limit", okL, "eps^n C(v/(2 eps)-1, n) -> (v/2)^n/n! (c=1), n=0..5, exact polynomial in eps")
check("Prop checker (3) O(eps)", okO, "first correction is -eps n(n+1)/v relative, so the error of L_eps/eps is O(eps)")
okdom = all(comb(Nn - 1, n) * sp.factorial(n) <= Nn**n for Nn in range(1, 40) for n in range(0, Nn))
check("Prop checker (3) domination", okdom, "C(N-1,n) <= N^n/n! for N=1..39")


def L_over_eps(tv, xv, epsv, muv=mp.mpf(1)):
    NR = int(round((tv + xv) / (2 * epsv)))
    NL = int(round((tv - xv) / (2 * epsv)))
    tot = mp.mpc(0)
    for n in range(0, min(NR, NL)):
        tot += (1j * epsv * muv)**(2 * n + 1) * mp.binomial(NR - 1, n) * mp.binomial(NL - 1, n)
    return tot / epsv


tv, xv = mp.mpf(3), mp.mpf('0.75')
uvv, vvv = tv - xv, tv + xv
lim = 1j * mp.besselj(0, mp.sqrt(tv**2 - xv**2))
pred1 = sum((1j)**(2 * n + 1) * (uvv * vvv / 4)**n / mp.factorial(n)**2 * (-(n * (n + 1)) * (1 / uvv + 1 / vvv)) for n in range(0, 60))
errs = []
for j in range(3, 11):
    epsv = mp.mpf(1) / 2**j * mp.mpf(1) / 8
    errs.append((epsv, (L_over_eps(tv, xv, epsv) - lim) / epsv))
okn = abs(errs[-1][1] - pred1) < 1e-3 * abs(pred1) + 1e-3 and abs(mp.re(L_over_eps(tv, xv, errs[-1][0]))) < mp.mpf(10)**-25
check("Prop checker (3) numeric", okn, "L_eps/eps -> i mu J0 at (t,x)=(3,0.75): (L/eps - limit)/eps = %s at eps=%s, predicted coefficient %s" % (mp.nstr(errs[-1][1], 8), mp.nstr(errs[-1][0], 3), mp.nstr(pred1, 8)))
check("Prop checker (2) sum", abs(sum((1j)**(2 * n + 1) * (uvv * vvv / 4)**n / mp.factorial(n)**2 for n in range(60)) / (2j) - mp.besselj(0, mp.sqrt(uvv * vvv)) / 2) < mp.mpf(10)**-25, "(2 i c mu)^-1 sum Phi_(2n+1) = J0/(2c) = G_mu (c=mu=1, 30 digits)")

# ---------------------------------------------------------------- Prop pair
wq, ta_, tb_ = sp.symbols('omega_k t_a t_b', positive=True)
M = lambda q: sp.Matrix([[sp.cos(wq * q), sp.sin(wq * q) / wq], [-wq * sp.sin(wq * q), sp.cos(wq * q)]])
check("Prop pair group law", sp.simplify(M(ta_ + tb_) - M(ta_) * M(tb_)) == sp.zeros(2, 2) and M(0) == sp.eye(2), "M_k(t+t') = M_k(t) M_k(t'), M_k(0) = 1")
gen = sp.Matrix([[0, 1], [-wq**2, 0]])
check("Prop pair flow", sp.simplify(sp.diff(M(ta_), ta_) - gen * M(ta_)) == sp.zeros(2, 2) and sp.simplify(M(ta_).det()) == 1, "d/dt M = [[0,1],[-omega^2,0]] M, det M = 1: flow of J'' = -omega^2 J on (J, dJ/dt)")

# ---------------------------------------------------------------- Thm unitary
tq = sp.symbols('t_q', real=True)
wr = sp.symbols('w_r', positive=True)
Asym = sp.Matrix(2, 2, sp.symbols('a11 a12 a21 a22'))
Bsym = sp.Matrix(2, 2, sp.symbols('b11 b12 b21 b22'))
Jp = Asym * sp.exp(-I * wr * tq)
Qp = sp.simplify(Jp * Jp.H - Asym * Asym.H)
check("Thm unitary (1)", sp.simplify(sp.Abs(sp.exp(-I * wr * tq)) - 1) == 0, "|e^{-i omega t}| = 1 for real omega: each mode multiplier is unimodular, generator hbar omega(k) = sqrt(m^2c^4 + hbar^2c^2k^2)")
check("Thm unitary (2)", Qp == sp.zeros(2, 2), "per mode J J^dagger = A A^dagger for J = A e^{-i omega t}, A in M_2(C): closure Q, Tr Q and P_m Q constant on one sign of frequency")
Jm = Asym * sp.exp(-I * wr * tq) + Bsym * sp.exp(I * wr * tq)
cross = sp.simplify(Jm * Jm.H - (Asym * Asym.H + Bsym * Bsym.H + Asym * Bsym.H * sp.exp(-2 * I * wr * tq) + Bsym * Asym.H * sp.exp(2 * I * wr * tq)))
check("Thm unitary (3) cross term", cross == sp.zeros(2, 2), "mixed mode: J J^dagger = AA^dagger + BB^dagger + A B^dagger e^{-2 i omega t} + B A^dagger e^{+2 i omega t}; trace cross term 2 Re tr(A B^dagger e^{-2 i omega t})")
E11 = sp.Matrix([[1, 0], [0, 0]])
E22 = sp.Matrix([[0, 0], [0, 1]])
Jc = E11 * sp.exp(-I * wr * tq) + E22 * sp.exp(I * wr * tq)
const_mixed = sp.simplify(Jc * Jc.H - sp.eye(2)) == sp.zeros(2, 2)
print("     counterexample to the universal reading of (3): A = E11, B = E22 (same k) gives J J^dagger = identity for all t:", const_mixed)
if not const_mixed:
    FAILS.append("counterexample evaluation")
note("Thm unitary (3): 'on a solution with both signs of frequency ||J||^2 is not constant' is false as a universal statement. ||J(t)||^2 varies iff int tr(a_+(k)^dagger a_-(k)) e^{2 i omega t} dk is not identically 0. Smallest example: J^ = E11 e^{-i omega t} + E22 e^{+i omega t} (or scalar a_+, a_- with disjoint k-supports): Q(t) constant.")
ap, am = sp.symbols('a_p a_m')
Js = ap * sp.exp(-I * wr * tq) + am * sp.exp(I * wr * tq)
Jsc = sp.conjugate(ap) * sp.exp(I * wr * tq) + sp.conjugate(am) * sp.exp(-I * wr * tq)
qd = sp.expand(I * (Jsc * sp.diff(Js, tq) - sp.diff(Jsc, tq) * Js) / 2)
check("Thm unitary (4) charge", z(qd - wr * (ap * sp.conjugate(ap) - am * sp.conjugate(am))), "(i/2)(J* dJ/dt - dJ*/dt J) per mode = omega(|a_+|^2 - |a_-|^2): cross terms cancel, time independent; times hbar/(mc^2) gives the stated density")
yv = sp.symbols('y_v', positive=True)  # y = sqrt(1+x) >= 1, x = hbar^2 k^2/(m^2 c^2)
check("Thm unitary (4) inequality", z(sp.factor((1 + (yv**2 - 1) / 2) - yv) - (yv - 1)**2 / 2), "1 <= sqrt(1+x) <= 1 + x/2: upper gap is (sqrt(1+x)-1)^2/2 >= 0, so 0 <= q - ||J||^2 <= hbar^2/(2m^2c^2) ||grad J||^2")

# ---------------------------------------------------------------- Thm NR, Cor feynman
xq = sp.symbols('x_q', positive=True)
low = sp.factor(((yv - 1 - (yv**2 - 1) / 2) + (yv**2 - 1)**2 / 8))
check("Thm NR inequality", z((yv - 1 - (yv**2 - 1) / 2) + (yv - 1)**2 / 2) and z(low - (yv - 1)**3 * (yv + 3) / 8), "sqrt(1+x)-1-x/2 = -(y-1)^2/2 <= 0 and + x^2/8 = (y-1)^3 (y+3)/8 >= 0 with y = sqrt(1+x) >= 1")
cinv = sp.symbols('cinv', positive=True)  # 1/c
omega_c = sp.sqrt(mm**2 / cinv**4 + hb**2 * kq**2 / cinv**2) / hb
expn = sp.series(sp.sqrt(1 + hb**2 * kq**2 * cinv**2 / mm**2), cinv, 0, 7).removeO() * mm / (cinv**2 * hb) - mm / (cinv**2 * hb)
expn = sp.expand(expn)
target = hb * kq**2 / (2 * mm) - hb**3 * kq**4 * cinv**2 / (8 * mm**3) + hb**5 * kq**6 * cinv**4 / (16 * mm**5)
check("Thm NR expansion", z(expn - target), "omega - mc^2/hbar = hbar k^2/2m - hbar^3 k^4/(8 m^3 c^2) + hbar^5 k^6/(16 m^5 c^4) + O(c^-6): error is exactly order 1/c^2, phase error coefficient t hbar^3 k^4/(8 m^3) (constant 1/8 is sharp)")
check("Thm NR bound constant", z(mm / cinv**2 * (hb**2 * kq**2 * cinv**2 / mm**2)**2 / 8 / hb - hb**3 * kq**4 * cinv**2 / (8 * mm**3)), "(t/hbar) m c^2 x^2/8 = t hbar^3 k^4/(8 m^3 c^2), matching |t| hbar^3/(8m^3c^2) ||Delta^2 psi||")
om_s = sp.symbols('om_s', positive=True)
posfreq = sp.expand((sp.sin(om_s * t) / om_s).rewrite(sp.exp))
check("Cor feynman positive-frequency part", z(posfreq.coeff(sp.exp(-I * om_s * t)) - I / (2 * om_s)), "e^{-i omega t} part of sin(omega t)/omega is (i/2 omega) e^{-i omega t}")
check("Cor feynman multiplier", z(2 * mm / (cinv**2 * I * hb) * I / (2 * om_s) - mm / (cinv**2 * hb * om_s)), "(2mc^2/(i hbar)) (i/2omega) = mc^2/(hbar omega)")
check("Cor feynman amplitude bound", z(sp.factor((yv**2 - 1) / 2 - (1 - 1 / yv)) - (yv - 1) * (yv**2 + yv - 2) / (2 * yv)) and z((yv**2 + yv - 2) - (yv - 1) * (yv + 2)), "x/2 - (1 - 1/sqrt(1+x)) = (y-1)^2 (y+2)/(2y) >= 0 and 1-1/y >= 0: 0 <= 1 - mc^2/(hbar omega) <= hbar^2k^2/(2m^2c^2)")
amp = sp.series(1 - 1 / sp.sqrt(1 + hb**2 * kq**2 * cinv**2 / mm**2), cinv, 0, 4).removeO()
check("Cor feynman amplitude order", z(amp - hb**2 * kq**2 * cinv**2 / (2 * mm**2)), "amplitude error is hbar^2k^2/(2m^2c^2) + O(c^-4): order 1/c^2, constant sharp")
# Feynman kernel: Schrodinger equation, and composition by continuation t -> -i/p
okK = True
xs = sp.symbols('x1:4', real=True)
tpos = sp.symbols('t_pos', positive=True)
for d in (1, 2, 3):
    r2 = sum(xi**2 for xi in xs[:d])
    K = (mm / (2 * pi * I * hb * tpos))**sp.Rational(d, 2) * sp.exp(I * mm * r2 / (2 * hb * tpos))
    resid = I * hb * sp.diff(K, tpos) + hb**2 / (2 * mm) * sum(sp.diff(K, xi, 2) for xi in xs[:d])
    okK = okK and sp.simplify(resid / K) == 0
check("Cor feynman kernel", okK, "K_F(t;x) solves i hbar d_t K = -(hbar^2/2m) Laplacian K for d=1,2,3")
p_, q_, Aa = sp.symbols('p q A_a', positive=True)
yv2 = sp.symbols('y2', real=True)
Kp = lambda pp, xval: sp.sqrt(Aa * pp / pi) * sp.exp(-Aa * pp * xval**2)
comp = sp.integrate(Kp(p_, xx - yv2) * Kp(q_, yv2), (yv2, -sp.oo, sp.oo))
check("Cor feynman composition", sp.simplify(comp - Kp(p_ * q_ / (p_ + q_), xx)) == 0 and z(sp.integrate(Kp(p_, yv2), (yv2, -sp.oo, sp.oo)) - 1), "with A = m/(2 hbar), p = 1/(i t): K_p * K_q = K_(pq/(p+q)), i.e. times add, and int K = 1 (Gaussian identity, continued to imaginary time parameter)")
ksq = sp.symbols('ksq', positive=True)
check("Cor feynman group law", z(sp.exp(-I * hb * ksq * ta_ / (2 * mm)) * sp.exp(-I * hb * ksq * tb_ / (2 * mm)) - sp.exp(-I * hb * ksq * (ta_ + tb_) / (2 * mm))) and z(sp.exp(-I * (om_s) * ta_) * sp.exp(-I * om_s * tb_) - sp.exp(-I * om_s * (ta_ + tb_))), "multipliers: K_F(t)K_F(t') = K_F(t+t') and the relativistic group law, mode by mode")
note("Cor feynman: the rescaled positive-frequency object has multiplier (mc^2/hbar omega) e^{-i(omega - mc^2/hbar)t}; the prefactor is not 1, so this family is NOT itself a group for finite c (U(t)U(t') has prefactor squared). The group law holds for e^{-i(H-mc^2)t/hbar} of Thm unitary(1), which is what the corollary cites; only the limit of [G_mu]^+ composes.")
check("Cor feynman finite-c non-group", sp.simplify((1 / yv) * (1 / yv) - 1 / yv) != 0, "(mc^2/hbar omega)^2 != mc^2/hbar omega for k != 0")

print()
print("elapsed %.1f s" % (time.time() - T0))
for nline in NOTES:
    print("NOTE-SUMMARY " + nline)
if FAILS:
    print("FAILED CHECKS: %d" % len(FAILS))
    for f in FAILS:
        print("  " + f)
    sys.exit(1)
print("ALL CHECKS PASSED")
sys.exit(0)
