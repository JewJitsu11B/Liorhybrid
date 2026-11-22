# Agent E: Thm rotation, formal expansion derived independently (sympy, exact).
# The last block is a high-precision numerical sanity check of MY series (mpmath); it is
# labelled as such and is not part of the exact derivation.
import sys
import sympy as sp
from sympy import symbols, I, pi, exp, log, sqrt, series, expand, simplify, Rational as R, zeta

def check(label, ok, note=""):
    print(("PASS  " if ok else "FAIL  ") + label + ("   | " + note if note else "")); sys.stdout.flush()

# ---- 1. the k=0 term and the paper's displayed expansion of F --------------------------
u, e, L, g = symbols('u epsilon L ghat')        # u = i phi, L = ln(i phi)
lam = symbols('lam', positive=True)             # bookkeeping: u ~ lam, eps ~ lam^2
k0 = (1 - exp(-u))*u**(-2)*exp(e*L)             # (1-e^{-u}) u^{-2+eps}
k0s = expand(series(k0.subs({u: lam*u, e: lam**2*e}), lam, 0, 4).removeO())
print("k=0 term, graded (u~lam, eps~lam^2):")
for p in range(-1, 4):
    print("   lam^%d:" % p, sp.factor(k0s.coeff(lam, p)))
bullet = expand(k0s.coeff(lam, -1)/lam + k0s.coeff(lam, 0))
check("Thm rotation bullet 1: (1-e^{-u})u^{-2+eps} = u^{-1+eps} - u^eps/2 + O(u)",
      simplify(expand(series(((1-exp(-u))*u**(-2) - 1/u + R(1, 2)), u, 0, 1).removeO())) == 0)
S00 = 2*zeta(2)*(2*pi*I)**(-2)                  # sum_{k!=0} (2 pi i k)^{-2}
check("Thm rotation bullet 2: sum_{k!=0}(2 pi i k)^-2 = -1/12, so k!=0 terms = -u/12 + ... = O(phi)",
      simplify(S00 + R(1, 12)) == 0)
# F = e^u - 1 + eps + eps g [k0 + (1-e^{-u})(-1/12 + ...)]
Fser = (exp(u) - 1 + e) + e*g*(k0 + (1 - exp(-u))*(-R(1, 12)))
Fg = expand(series(Fser.subs({u: lam*u, e: lam**2*e}), lam, 0, 5).removeO())
print("F graded:")
for p in range(1, 5):
    print("   lam^%d:" % p, sp.collect(expand(Fg.coeff(lam, p)), [g, L]))
phi = symbols('phi')
paperF = I*phi - phi**2/2 + e + e*g/(I*phi) - e*g/2 + e**2*g*L/(I*phi)
mine_to_paper = expand((Fg.coeff(lam, 1) + Fg.coeff(lam, 2)).subs(u, I*phi) + e**2*g*L/(I*phi))
check("Thm rotation: displayed F expansion (all terms shown) matches", simplify(mine_to_paper - paperF) == 0)
o3 = expand(Fg.coeff(lam, 3).subs(u, I*phi))
rest3 = simplify(o3 - e**2*g*L/(I*phi))
print("   order eps^{3/2} terms of F other than the log term:", sp.factor(rest3))
check("Thm rotation: non-log eps^{3/2} terms are i*(eps*g*phi/12 - phi^3/6): imaginary for real phi",
      simplify(rest3 - I*(e*g*phi/12 - phi**3/6)) == 0)
check("Thm rotation: F'(phi_0) = 2i for the leading part i phi + eps g/(i phi)",
      simplify(sp.diff(I*phi + e*g/(I*phi), phi).subs(phi, sqrt(e*g)) - 2*I) == 0)

# ---- 2. solve order by order: phi = t sqrt(g) (1 + d1 t + d2 t^2 + d3 t^3), eps = t^2 ----
t = symbols('t', positive=True)
gp = symbols('ghat', positive=True)
sg = sqrt(gp)
d1, d2, d3 = symbols('d1 d2 d3')
dl = d1*t + d2*t**2 + d3*t**3
uu = I*t*sg*(1 + dl)
ueps = exp(t**2*(L + log(1 + dl)))              # (i phi)^eps, L = ln(i phi_0)
# G = u*F, exact k=0 term, k!=0 terms to the order needed: (1-e^{-u})(-1/12 + O(t^2 ln, t^2))
G = uu*(exp(uu) - 1 + t**2) + t**2*gp*ueps*(1 - exp(-uu))/uu + t**2*gp*uu*(1 - exp(-uu))*(-R(1, 12))
Gs = expand(series(G, t, 0, 6).removeO())
sol = {}
check("leading order t^2 of u*F vanishes at phi_0 = sqrt(eps*ghat)", simplify(Gs.coeff(t, 2)) == 0)
for p, dv in ((3, d1), (4, d2), (5, d3)):
    eq = Gs.coeff(t, p).subs(sol)
    sol[dv] = sp.factor(sp.solve(eq, dv)[0])
    print("  ", dv, "=", sol[dv])
phis = expand(t*sg*(1 + dl.subs(sol)))
ell = symbols('ell', real=True)                  # ell = ln(phi_0) = ln sqrt(eps*ghat), real
ph = expand(phis.subs(L, ell + I*pi/2))
re_ = sp.re(ph); im_ = sp.im(ph)
print("Re phi =", sp.collect(expand(re_), t))
print("Im phi =", sp.collect(expand(im_), t))
phi1_paper = I*t**2*(1 - gp)/2
phi2_paper = R(1, 2)*t**3*sg*(ell + I*pi/2)
c2 = expand(ph.coeff(t, 2)); c3 = expand(ph.coeff(t, 3)); c4 = expand(ph.coeff(t, 4))
check("Thm rotation phi_1 = i eps (1-ghat)/2", simplify(c2*t**2 - phi1_paper) == 0)
check("Thm rotation phi_2: log part (1/2) eps^{3/2} sqrt(g) (ln phi_0 + i pi/2)",
      simplify(sp.im(c3) - pi*sg/4) == 0 and simplify(sp.diff(c3, ell) - sg/2) == 0)
print("   full eps^{3/2} coefficient:", sp.collect(sp.factor(c3), ell))
print("   real non-log part of eps^{3/2} coefficient:", sp.factor(simplify(sp.re(c3) - sg*ell/2)))
check("Thm rotation: Im phi = (1/2)eps(1-g) + (pi/4) eps^{3/2} sqrt(g) through order eps^{3/2}",
      simplify(im_.coeff(t, 2) - (1 - gp)/2) == 0 and simplify(im_.coeff(t, 3) - pi*sg/4) == 0)
print("   eps^2 coefficient of Im phi (paper: O(eps^2 ln eps)):", sp.collect(sp.factor(sp.im(c4)), ell))
print("   eps^2 coefficient of Re phi:", sp.factor(sp.re(c4)))
im4 = sp.im(c4)
print("   at ghat=1: Im phi eps^2 coeff =", sp.simplify(im4.subs(gp, 1)), "; Re:", sp.simplify(sp.re(c4).subs(gp, 1)))

# ---- 3. numerical sanity check of the series above (not exact; mpmath, 40 digits) --------
import mpmath as mp
mp.mp.dps = 40
def Wcont(phi, al):
    s_ = al + 1
    a1 = 1 + phi/(2*mp.pi); a2 = 1 - phi/(2*mp.pi)
    tot = (1j*phi)**(-s_)
    tot += (2*mp.pi)**(-s_)*(mp.exp(-1j*mp.pi*s_/2)*mp.zeta(s_, a1) + mp.exp(1j*mp.pi*s_/2)*mp.zeta(s_, a2))
    return (1 - mp.exp(-1j*phi))*tot
def Fex(phi, eps, gh):
    return mp.exp(1j*phi) - (1 - eps) + eps*gh*Wcont(phi, 1 - eps)
# check the continuation against direct summation at |z|>1 (phi = 0.3 - 0.2i -> |z| = e^0.2)
al0 = mp.mpf('0.7'); p0 = mp.mpc('0.3', '-0.2'); z0 = mp.exp(1j*p0)
direct = mp.nsum(lambda j: (j**al0 - (j-1)**al0)/mp.gamma(al0+1)*z0**(-j), [1, mp.inf])
print("sanity: W continuation vs direct sum at |z|=e^0.2:", mp.nstr(abs(direct - Wcont(p0, al0)), 5))
fser = sp.lambdify((t, gp, ell), ph, 'mpmath')
print("eps      ghat  Re phi(exact)            Im phi(exact)            |exact-series|/eps^(5/2)   Im/Re/eps")
for gh in (1, mp.mpf('0.9'), mp.mpf('0.5')):
    for ee in ('1e-2', '1e-3', '1e-4', '1e-5', '1e-6', '1e-7', '1e-9'):
        eps = mp.mpf(ee); tt = mp.sqrt(eps)
        guess = mp.sqrt(eps*gh) + 1j*(eps*(1-gh)/2 + mp.pi/4*eps**1.5*mp.sqrt(gh))
        root = mp.findroot(lambda p: Fex(p, eps, gh), guess)
        sv = fser(tt, gh, mp.log(mp.sqrt(eps*gh)))
        print(ee, mp.nstr(gh, 3), mp.nstr(root.real, 12), mp.nstr(root.imag, 8),
              mp.nstr(abs(root - sv)/eps**2.5, 6), mp.nstr(root.imag/root.real/eps, 6))
