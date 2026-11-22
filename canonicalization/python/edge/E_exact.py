# Agent E: exact algebra checks (sympy / Fractions, exact arithmetic only).
# Independent of reference/cal_edge_transducer.py (not imported, not read).
import sys, random
from fractions import Fraction as Fr
import sympy as sp
from sympy import (symbols, Symbol, Rational as R, sqrt, exp, log, cos, sin, cosh, sinh,
                   tanh, acos, acosh, gamma, pi, I, oo, simplify, expand, series, Matrix,
                   eye, zeros, conjugate, integrate, limit, diff, factor, together, cancel)

RES = []
def check(label, ok, note=""):
    RES.append((label, bool(ok)))
    print(("PASS  " if ok else "FAIL  ") + label + ("   | " + note if note else ""))
    sys.stdout.flush()

def Z(e):
    return simplify(e) == 0

a = symbols('alpha', positive=True)
g = symbols('ghat', positive=True)
z = symbols('z')

# ---------------------------------------------------------------- Def memory
tau = symbols('tau', positive=True)
check("Def memory: d/dtau [tau^a/Gamma(a+1)] = tau^(a-1)/Gamma(a)",
      Z(sp.gammasimp(diff(tau**a/gamma(a+1), tau) - tau**(a-1)/gamma(a))))
ok = True
for jj in range(1, 6):
    for aa in (R(1, 3), R(1, 2), R(9, 10)):
        val = integrate(tau**(aa-1)/gamma(aa), (tau, jj-1, jj))
        tgt = (R(jj)**aa - R(jj-1)**aa)/gamma(aa+1)
        ok &= Z(val - tgt)
check("Def memory: integral over [j-1,j] equals w_j, j=1..5, alpha in {1/3,1/2,9/10}", ok)

# ---------------------------------------------------------------- Lemma W first line
x = symbols('x')
N = 60
pw = lambda j: sp.Integer(0) if j == 0 else sp.Integer(j)**a   # 0^alpha = 0 for alpha>0
lhs = sum((pw(j) - pw(j-1))*x**j for j in range(1, N+1))
rhs = expand((1-x)*sum(pw(j)*x**j for j in range(1, N+1)))
ok = all(Z(lhs.coeff(x, j) - rhs.coeff(x, j)) for j in range(1, N+1))
check("Lemma W proof line 1: coefficientwise to order x^60, symbolic alpha", ok)

# ---------------------------------------------------------------- Lemma modes
Nw = 7
w = symbols('w1:%d' % (Nw+1))
n = symbols('n', integer=True)
Wz = sum(w[j-1]*z**(-j) for j in range(1, Nw+1))
F = z - a + (1-a)*g*Wz
resid = z**(n+1) - a*z**n + (1-a)*g*sum(w[j-1]*z**(n-j) for j in range(1, Nw+1))
check("Lemma modes (1): residual of z^n in eq. update = z^n F(z) (generic w_1..w_7)",
      Z(expand(sp.powsimp(expand(resid/z**n), force=True) - F)) and
      all(Z(expand(resid.subs(n, nv) - z**nv*F)) for nv in (-3, 0, 5)))

# (3) zero past: z-transform = z J0 / F(z).  x = 1/z:  Jhat = J0 / (1 - a x + (1-a) g x W(x))
J0 = symbols('J0')
Js = [J0]
for nn in range(0, Nw):
    Js.append(expand(a*Js[nn] - (1-a)*g*sum(w[j-1]*Js[nn-j] for j in range(1, nn+1))))
Wx = sum(w[j-1]*x**j for j in range(1, Nw+1))
ser = series(J0/(1 - a*x + (1-a)*g*x*Wx), x, 0, Nw+1).removeO()
ok = all(Z(expand(ser.coeff(x, k) - Js[k])) for k in range(0, Nw+1))
check("Lemma modes (3): series of z J0/F(z) in 1/z reproduces zero-past J_0..J_7", ok)

# (4) split of the memory sum at j = n, history depth M (formal, generic weights)
M = 12
ww = symbols('v1:%d' % (M+1))
c, h = symbols('c h')          # h = e^{i phi_h}
ok = True
for nn in range(0, 7):
    hist = sum(ww[j-1]*c*h**(nn-j) for j in range(nn+1, M+1))        # terms with n-j<0
    claim = c*h**nn*(sum(ww[j-1]*h**(-j) for j in range(1, M+1))
                     - sum(ww[j-1]*h**(-j) for j in range(1, nn+1)))
    ok &= Z(expand(hist - claim))
    ok &= all((nn-j < 0) == (j > nn) for j in range(1, M+1))
check("Lemma modes (4): history part = c h^n (W(h) - sum_{j<=n} w_j h^-j), n=0..6", ok)

# ---------------------------------------------------------------- Lemma flat
Ff = z - a + (1-a)*g/(z-1)
check("Lemma flat: (z-1)F = z^2-(1+a)z+a+(1-a)ghat",
      Z(expand((z-1)*Ff - (z**2 - (1+a)*z + a + (1-a)*g))))
ph = symbols('phi', real=True)
pol1 = lambda zz, al: zz**2 - (1+al)*zz + 1
al_c = 2*cos(ph) - 1
check("Lemma flat (1): e^{+-i phi}, cos phi=(1+a)/2, are the roots at ghat=1",
      Z(sp.expand(pol1(exp(I*ph), al_c), complex=True)) and Z(sp.expand(pol1(exp(-I*ph), al_c), complex=True)))
check("Lemma flat (1): (1+a)/2 in (1/2,1) for a in (0,1), so phi real, |z|=1", True, "by inspection")
check("Lemma flat (2): alpha=1 gives (z-1)^2", Z(expand(pol1(z, 1) - (z-1)**2)),
      "note: F itself is z-1 at alpha=1 (simple root); the double root is of (z-1)F")
s = symbols('s', positive=True)
al_h = 2*cosh(s) - 1
check("Lemma flat (3) / Thm feedback (2) flat roots: e^{+-s}, cosh s=(1+a)/2",
      Z(pol1(exp(s), al_h).rewrite(exp)) and Z(pol1(exp(-s), al_h).rewrite(exp)))
e = symbols('epsilon', positive=True)
t = symbols('t', positive=True)
phi_ser = series(acos(1 - t**2/2), t, 0, 8).removeO()
s_ser = series(acosh(1 + t**2/2), t, 0, 8).removeO()
print("      phi(alpha<1), t=sqrt(1-alpha):", phi_ser)
print("      s(alpha>1),   t=sqrt(alpha-1):", s_ser)
check("Lemma flat: phi = sqrt(eps)(1 + eps/24 + 3 eps^2/640 + ...)",
      Z(phi_ser - (t + t**3/24 + 3*t**5/640 + 5*t**7/7168)))
check("Lemma flat: s = sqrt(d)(1 - d/24 + 3 d^2/640 - ...), d=alpha-1",
      Z(s_ser - (t - t**3/24 + 3*t**5/640 - 5*t**7/7168)))

# ---------------------------------------------------------------- Cor dispersion
k, r2 = symbols('k r2', positive=True)
c4 = symbols('c4')
gk = 1 - r2*k**2/6 + c4*k**4
check("Cor dispersion: 1 - sqrt(ghat) = <r^2>k^2/12 + O(k^4)",
      Z(series(1 - sqrt(gk), k, 0, 4).removeO() - r2*k**2/12))
print("      k^4 coefficient of 1-sqrt(ghat):", sp.factor(series(1 - sqrt(gk), k, 0, 6).removeO().coeff(k, 4)))
check("Cor dispersion: Gaussian closed form 1 - exp(-<r^2>k^2/12)",
      Z(sp.powsimp(sqrt(exp(-r2*k**2/6)) - exp(-r2*k**2/12), force=True)))
print("      Gaussian lag series:", series(1 - exp(-r2*k**2/12), k, 0, 6))
# Table mass at leading order in eps: (1-sqrt(ghat))/k^2
rows = [(R(6, 100), R(1, 10), "0.005000"), (R(6, 10), R(1, 10), "0.049988"), (R(6, 10), R(2, 10), "0.049950"),
        (6, R(5, 100), "0.499688"), (6, R(2, 10), "0.495034")]
for rr, kk, tab in rows:
    v = (1 - exp(-rr*kk**2/12))/kk**2
    print("      Table mass  <r^2>=%s k=%s  eps->0 exact value %s   paper(eps=1e-5) %s" % (rr, kk, sp.N(v, 8), tab))

# ---------------------------------------------------------------- Thm unitary
sg = symbols('sg', positive=True)        # sqrt(ghat)
gg = sg**2
imphi = R(1, 2)*e*(1-gg) + pi/4*e**R(3, 2)*sg     # Thm rotation, Im phi
rephi = sqrt(e)*sg                                 # leading Re phi
rest = sqrt(e)                                     # Re phi(0)
tt = symbols('T', positive=True)                   # rest radians; n = T/rest
expo = I*rephi*tt/rest - imphi*tt/rest - I*tt      # exponent with the rest rotation e^{iT} removed
paper1 = -I*(1-sg)*tt - (R(1, 2)*sqrt(e)*(1-gg) + pi/4*e*sg)*tt
check("Thm unitary: first displayed exponent follows from Thm rotation / sqrt(eps)", Z(expand(expo - paper1)))
decay = R(1, 2)*sqrt(e)*(1-gg) + pi/4*e*sg
extra = decay - pi/4*e                             # extra decay over the k=0 (uniform) decay
ratio = cancel(extra/(1-sg))
check("Thm unitary (2): (decay_k - decay_0)/(1-sqrt g) = (1/2)sqrt(eps)(1+sqrt g) - (pi/4)eps exactly",
      Z(ratio - (R(1, 2)*sqrt(e)*(1+sg) - pi/4*e)))
# long-wavelength form: sg = 1 - K, K = (hbar/2m) k^2
K = symbols('K', positive=True)
lw = series(expand(paper1.subs(sg, 1-K)), K, 0, 2).removeO()
paper2 = -(I + sqrt(e))*K*tt - pi/4*e*tt
dlt = simplify(expand(lw - paper2))
print("      first display to O(K) minus second display:", dlt)
check("Thm unitary: second display equals first to O(k^2) up to the term +(pi/4) eps K t",
      Z(dlt - pi/4*e*K*tt), "paper drops -(pi/4)eps in the diffusive coefficient; exact O(K) coefficient is -(i+sqrt(eps)-(pi/4)eps)")
gv = exp(-R(6, 100)*R(1, 100)/6)
print("      Table diffusion predicted, eps=1e-2,k=0.1,<r^2>=0.06:",
      sp.N(R(1, 2)*sqrt(R(1, 100))*(1+sqrt(gv)) - pi/4*R(1, 100), 8), " paper says 0.0921")

# ---------------------------------------------------------------- Prop born (1)
k2 = symbols('k2', positive=True)   # (hbar/2m) k^2 = K
amp_rate = series((R(1, 2)*sqrt(e)*(1-(1-K)**2) + pi/4*e*(1-K)), K, 0, 2).removeO()
print("      amplitude decay per radian to O(K):", sp.collect(expand(amp_rate), K))
check("Prop born (1): |amp|^2 rate = 2[(pi/4)eps + sqrt(eps) K] up to -(pi/2) eps K",
      Z(expand(2*amp_rate - 2*(pi/4*e + sqrt(e)*K) + pi/2*e*K)),
      "leading order in k^2 and eps; <k^2> average is the initial rate only")

# ---------------------------------------------------------------- Thm feedback
Jn, Wsum, b, lam = symbols('J_n Wsum b lambda')
check("Thm feedback (1): alpha=1 in eq. update gives J_{n+1}=J_n",
      Z((a*Jn - (1-a)*g*Wsum).subs(a, 1) - Jn))
# (3) differencing with flat weights, ghat=1: S_n = sum_{j>=1} J_{n-j},  S_n - S_{n-1} = J_{n-1}
Jp, Jm, S1 = symbols('J_np1 J_nm1 S_nm1')
upd_n = Jp - (a*Jn - (1-a)*(S1 + Jm))      # update at n  (S_n = S_{n-1} + J_{n-1})
upd_m = Jn - (a*Jm - (1-a)*S1)             # update at n-1
check("Thm feedback (3): difference of consecutive updates = J_{n+1}-(1+a)J_n+J_{n-1}",
      Z(expand(upd_n - upd_m - (Jp - (1+a)*Jn + Jm))))
G = lambda m: exp(-s*abs(m))/(2*sinh(s))
L = lambda m: G(m+1) - (1+al_h)*G(m) + G(m-1)
ok = all(Z(L(m).rewrite(exp)) for m in (-4, -3, -2, -1, 1, 2, 3, 4))
check("Thm feedback (3): G=e^{-s|n-n0|}/(2 sinh s) solves the recurrence for n != n0", ok)
jump = simplify(L(0).rewrite(exp))
print("      jump at n0: G(1)-(1+a)G(0)+G(-1) =", jump)
check("Thm feedback (3): jump at n0 equals +1 (unit source)", jump == 1,
      "actual value is -1: Green's function of the displayed operator is -e^{-s|n-n0|}/(2 sinh s)")
# exact rational instance in the ORIGINAL update rule: e^s = 2, alpha = 3/2, J_n = 2^{-|n|}
al = Fr(3, 2)
def Jx(m): return Fr(1, 2**abs(m))
def Rn(nn, depth=40):
    # sum_{j>=1} J_{n-j}: explicit terms down to index -depth, exact geometric tail below
    tot = sum(Jx(nn-j) for j in range(1, nn+depth+1))      # indices n-1 .. -depth
    tot += Fr(1, 2**depth)                                  # sum_{i<-depth} 2^{i} = 2^{-depth}
    return Jx(nn+1) - (al*Jx(nn) - (1-al)*tot)
vals = [Rn(nn) for nn in range(-6, 7)]
print("      update-rule residual of J_n=2^-|n| (alpha=3/2), n=-6..6:", [str(v) for v in vals])
check("Thm feedback (3): in eq. update, e^{-s|n|} needs a source -2 sinh(s) at EVERY n>=0 (step, not impulse)",
      all(v == 0 for v in vals[:6]) and all(v == Fr(-3, 2) for v in vals[6:]),
      "2 sinh s = 3/2 here")
# bounded impulse response in eq. update: H_n = -(G(n) - G(n-1)), G = 2^{-|n|}/(3/2)
def Hx(m): return -(Jx(m) - Jx(m-1))/Fr(3, 2)
def RH(nn, depth=40):
    tot = sum(Hx(nn-j) for j in range(1, nn+depth+1))
    tot += -(Fr(1, 2**depth) - Fr(1, 2**(depth+1)))/Fr(3, 2)*1   # tail: sum_{i<-depth} H_i
    return Hx(nn+1) - (al*Hx(nn) - (1-al)*tot)
vh = [RH(nn) for nn in range(-6, 7)]
print("      residual of H_n=-(G_n-G_{n-1}), n=-6..6:", [str(v) for v in vh])
print("      H_n, n=-3..3:", [str(Hx(m)) for m in range(-3, 4)])
check("Thm feedback (3): bounded response to a unit impulse S_n=delta_{n,0} in J_{n+1}=T_n+S_n is H_n=-(G_n-G_{n-1})",
      all(v == 0 for i, v in enumerate(vh) if i != 6) and vh[6] == 1,
      "two-sided, nonzero before the source; amplitudes differ in size and sign on the two sides")
# (4)
Wl = symbols('W_lambda')
Fb = lambda zz, Wv: zz - (1-b)*(a*1 + (a-1)*Wv)
T_n = a*Jn - (1-a)*Wl*Jn           # mode at z=1: memory sum = W_lambda(1) J
check("Thm feedback (4): F(1) = 1-(1-b)(a+(a-1)W_lambda(1))",
      Z(expand((Jn - (1-b)*T_n)/Jn - (1 - (1-b)*(a + (a-1)*Wl)))))

# ---------------------------------------------------------------- Thm normalizable
xr, yr = symbols('x_r y_r', real=True)
zc = xr + I*yr
check("Thm normalizable: |z-a|^2 - (|z|-a)^2 = 2a(|z|-Re z) >= 0",
      Z(expand(sp.Abs(zc - a)**2 - (sp.Abs(zc) - a)**2 - 2*a*(sp.Abs(zc) - xr), complex=True)),
      "equality iff z >= 0 real; with 1-a=|z|-a this forces z=1")
check("Thm normalizable: F(1) = 2(1-a) when sum w = 1, ghat = 1", Z((1 - a + (1-a)*1*1) - 2*(1-a)))

def schur_inside(cf):
    """cf[k] = coefficient of z^k (rational, real). True iff all roots in the open unit disc.
    Schur-Cohn reduction in exact integer arithmetic (denominators cleared, content removed)."""
    from math import gcd, lcm
    den = 1
    for v in cf:
        den = lcm(den, Fr(v).denominator)
    cf = [int(Fr(v)*den) for v in cf]
    while len(cf) > 1:
        nn = len(cf) - 1
        c0, cn = cf[0], cf[nn]
        if cn == 0:
            raise ValueError
        if abs(c0) >= abs(cn):
            return False
        cf = [cn*cf[kk+1] - c0*cf[nn-kk-1] for kk in range(nn)]
        if len(cf) <= 12:
            gg_ = 0
            for v in cf:
                gg_ = gcd(gg_, v)
            if gg_ > 1:
                cf = [v//gg_ for v in cf]
    return True

def poly_for(al_, g_, wl):
    Nn = len(wl)       # z^N F(z) = z^{N+1} - a z^N + (1-a) g sum w_j z^{N-j}
    cf = [Fr(0)]*(Nn+2)
    cf[Nn+1] = Fr(1); cf[Nn] = -al_
    for j, wj in enumerate(wl, start=1):
        cf[Nn-j] += (1-al_)*g_*wj
    return cf

# self-test of the Schur-Cohn routine on known cases
assert schur_inside([Fr(1, 4), Fr(-1), Fr(1)])            # (z-1/2)^2
assert not schur_inside([Fr(1), Fr(-2), Fr(1)])           # (z-1)^2
assert not schur_inside([Fr(-3, 4), Fr(-1), Fr(1)])       # roots 3/2, -1/2
assert not schur_inside([Fr(1), Fr(0), Fr(1)])            # +-i on the circle
assert schur_inside([Fr(1, 2), Fr(0), Fr(1)])

random.seed(20261122)
bad = 0; trials = 0
edge_al = [Fr(1, 10**6), Fr(1, 100), Fr(1, 2), Fr(99, 100), Fr(999999, 10**6)]
for it in range(4000):
    Nn = random.randint(1, 10)
    raw = [Fr(random.randint(0, 50)) if random.random() < 0.7 else Fr(0) for _ in range(Nn)]
    if sum(raw) == 0:
        raw[random.randrange(Nn)] = Fr(1)
    tot = sum(raw); wl = [v/tot for v in raw]
    al_ = random.choice(edge_al) if random.random() < 0.4 else Fr(random.randint(1, 999), 1000)
    g_ = random.choice([Fr(0), Fr(1), Fr(1), Fr(random.randint(0, 100), 100)])
    trials += 1
    if not schur_inside(poly_for(al_, g_, wl)):
        bad += 1; print("      COUNTEREXAMPLE", al_, g_, wl)
# structured: Dirac at lag m, two-point kernels, alternating supports
for m in range(1, 16):
    for al_ in edge_al:
        wl = [Fr(0)]*m; wl[m-1] = Fr(1)
        trials += 1
        if not schur_inside(poly_for(al_, Fr(1), wl)):
            bad += 1; print("      COUNTEREXAMPLE Dirac lag", m, al_)
check("Thm normalizable: exact Schur-Cohn test, %d rational kernels, no root with |z|>=1" % trials, bad == 0)
# hypotheses are needed:
check("Thm normalizable sharpness: total weight M=1/(1-a) (Dirac, a=1/2, M=2) puts roots ON the circle",
      (not schur_inside(poly_for(Fr(1, 2), Fr(1), [Fr(2)]))) and schur_inside(poly_for(Fr(1, 2), Fr(1), [Fr(199, 100)])),
      "z^2 - z/2 + 1 = 0, |z|=1")
check("Thm normalizable sharpness: signed weights with sum 1 can fail (w=(3,-2), a=1/2 -> root z=-1.186)",
      not schur_inside(poly_for(Fr(1, 2), Fr(1), [Fr(3), Fr(-2)])))
# exact Table normalizable rows (closed forms)
aa = R(99, 100)
zD = (aa + sqrt(aa**2 - 4*(1-aa)))/2
print("      Table normalizable, Dirac eps=1e-2: largest |z| =", sp.N(zD, 8), " paper 0.97979")
for wdt in (10, 100):
    q = exp(-R(1, wdt))          # w_j = (1-q) q^{j-1}, infinite kernel: W = (1-q)/(z-q)
    rts = sp.solve((z-aa)*(z-q) + (1-aa)*(1-q), z)
    print("      Table normalizable, exponential width %d (infinite kernel, q=e^(-1/%d)): |z| =" % (wdt, wdt),
          [sp.N(sp.Abs(rv), 8) for rv in rts], " paper", "0.97677" if wdt == 10 else "0.99205")

# ---------------------------------------------------------------- Prop accumulator
ss = symbols('s_', positive=True)
al01 = R(1, 3)
ok = True
for al01 in (R(1, 3), R(1, 2), R(4, 5)):
    lap = integrate(exp(-ss*tau)*sin(pi*al01)/pi*ss**(-al01), (ss, 0, oo))
    ratio_ = sp.gammasimp(sp.simplify(lap*gamma(al01)/tau**(al01-1)))
    print("      a =", al01, " integral =", lap, "  ratio to tau^(a-1)/Gamma(a):", ratio_)
    ok &= (ratio_ == 1)
check("Prop accumulator: Laplace representation of tau^(a-1)/Gamma(a), a in {1/3,1/2,4/5}", ok)
asym = symbols('a_', positive=True)
check("Prop accumulator: Gamma(a)Gamma(1-a)=pi/sin(pi a) closes the general case",
      sp.gammasimp(sin(pi*asym)/pi*gamma(1-asym)*gamma(asym)) == 1)
dl, Sx = symbols('delta S', positive=True)
ok = True
for al01 in (R(1, 3), R(1, 2), R(4, 5), R(99, 100)):
    fr = integrate(ss**(-al01), (ss, 0, dl))/integrate(ss**(-al01), (ss, 0, Sx))
    ok &= Z(sp.powsimp(fr - (dl/Sx)**(1-al01), force=True))
check("Prop accumulator: mass fraction in [0,delta] of [0,S] is (delta/S)^(1-a)", ok)
ok = all(limit((sp.Integer(j)**asym - sp.Integer(j-1)**asym)/gamma(asym+1), asym, 1, '-') == 1 for j in range(2, 8))
ok &= limit(1/gamma(asym+1), asym, 1, '-') == 1
check("Prop accumulator: w_j -> 1 as a -> 1-, j=1..7", ok)

# ---------------------------------------------------------------- Prop carrier
re_ = symbols('a0:4', real=True); im_ = symbols('b0:4', real=True)
J = Matrix(2, 2, [re_[i] + I*im_[i] for i in range(4)])
Q = (J*J.H).applyfunc(expand)
vec = Matrix([J[0, 0], J[0, 1], J[1, 0], J[1, 1]])      # |i>|j>, index 2i+j
rho = vec*vec.H
TrR = Matrix(2, 2, lambda i, kk: sum(rho[2*i+j, 2*kk+j] for j in range(2)))
check("Prop carrier (1): Tr_right |J>><<J| = J J^dagger", (TrR - Q).applyfunc(expand) == zeros(2, 2))
TrL = Matrix(2, 2, lambda j, l: sum(rho[2*i+j, 2*i+l] for i in range(2)))
check("Prop carrier (1) converse: Tr_left gives (J^dagger J)^T, not J J^dagger",
      (TrL - (J.H*J).T).applyfunc(expand) == zeros(2, 2))
detJ = J.det()
absdet2 = expand(detJ*conjugate(detJ))
check("Prop carrier (3): det(J J^dagger) = |det J|^2", Z(expand(Q.det()) - absdet2))
check("Prop carrier (3): tr(J J^dagger) = sum |J_ij|^2", Z(Q.trace() - sum(re_[i]**2 + im_[i]**2 for i in range(4))))
s1 = Matrix([[0, 1], [1, 0]]); s2 = Matrix([[0, -I], [I, 0]]); s3 = Matrix([[1, 0], [0, -1]])
x0 = Q.trace()/2
xv = [expand((Q*sm).trace()/2) for sm in (s1, s2, s3)]
check("Prop carrier (4): Q = x0 I + x.sigma with real x",
      (Q - (x0*eye(2) + xv[0]*s1 + xv[1]*s2 + xv[2]*s3)).applyfunc(expand) == zeros(2, 2)
      and all(sp.im(v) == 0 for v in xv))
xsq = expand(sum(v**2 for v in xv))
check("Prop carrier (4): (2|det J|/tr Q)^2 = 1 - |x|^2/x0^2", Z(expand(4*absdet2 - (2*x0)**2 + 4*xsq)))
check("Prop carrier (4): eigenvalues x0 +- |x|; 4 p+ p- = 1 - |x|^2/x0^2",
      Z(expand(Q.charpoly().as_expr().subs(Q.charpoly().gens[0], z) - (z**2 - 2*x0*z + x0**2 - xsq))))
u1, u2, v1, v2 = symbols('u1 u2 v1 v2')
Jprod = Matrix([[u1*v1, u1*v2], [u2*v1, u2*v2]])
check("Prop carrier (2): product (rank-one) J has det 0", Z(Jprod.det()),
      "converse needs J != 0: J=0 has det 0 and rank 0")

# ---------------------------------------------------------------- Prop lorentz
gs = symbols('g0:4')
Gm = Matrix(2, 2, gs)
check("Prop lorentz (1): det(GJ) = det G det J", Z(expand((Gm*J).det() - Gm.det()*detJ)))
eta, eta2, rho_ = symbols('eta eta2 rho', positive=True)
th, vp = symbols('theta varphi', real=True)
nv = (sin(th)*cos(vp), sin(th)*sin(vp), cos(th))
ns = nv[0]*s1 + nv[1]*s2 + nv[2]*s3
check("Prop lorentz: (n.sigma)^2 = I", (ns*ns - eye(2)).applyfunc(simplify) == zeros(2, 2))
B = lambda et: cosh(et/2)*eye(2) + sinh(et/2)*ns          # e^{eta n.sigma/2}
Qb = (B(eta)*(rho_*eye(2))*B(eta).H).applyfunc(lambda v: simplify(v.rewrite(exp)))
tgt = (rho_*(cosh(eta)*eye(2) + sinh(eta)*ns)).applyfunc(lambda v: simplify(v.rewrite(exp)))
check("Prop lorentz (3): boost of Q=rho I is rho(cosh eta + sinh eta n.sigma)",
      (Qb - tgt).applyfunc(simplify) == zeros(2, 2))
check("Prop lorentz (1): det Q invariant (rho^2), tr Q = 2 rho cosh eta changes",
      Z(simplify(Qb.det() - rho_**2)) and Z(simplify(Qb.trace() - 2*rho_*cosh(eta).rewrite(exp))))
check("Prop lorentz (3): C = sqrt(1-tanh^2) = sech eta", Z(simplify((1 - tanh(eta)**2 - 1/cosh(eta)**2).rewrite(exp))))
pp = (1 + tanh(eta))/2; pm = (1 - tanh(eta))/2
check("Prop lorentz (3): p+- = e^{+-eta}/(2 cosh eta), p+/p- = e^{2 eta}",
      Z(simplify((pp - exp(eta)/(2*cosh(eta))).rewrite(exp))) and Z(simplify((pm - exp(-eta)/(2*cosh(eta))).rewrite(exp)))
      and Z(simplify((pp/pm - exp(2*eta)).rewrite(exp))))
check("Prop lorentz (3): collinear boosts add rapidities",
      (B(eta)*B(eta2) - B(eta+eta2)).applyfunc(lambda v: simplify(v.rewrite(exp))) == zeros(2, 2))
# rotation: tr and det of Q unchanged under unitary G
ca, cb = symbols('ca cb')   # generic SU(2) parametrised by two complex numbers with |ca|^2+|cb|^2=1
Uu = Matrix([[re_[0]+I*im_[0], re_[1]+I*im_[1]], [-(re_[1]-I*im_[1]), re_[0]-I*im_[0]]])
nrm = re_[0]**2 + im_[0]**2 + re_[1]**2 + im_[1]**2
check("Prop lorentz (2): G in SU(2) (up to norm) gives G G^dagger = |G|^2 I, so tr Q and C unchanged",
      ((Uu*Uu.H).applyfunc(expand) - nrm*eye(2)).applyfunc(expand) == zeros(2, 2))
print("      Table readout: sech(1.6) =", sp.N(1/cosh(R(16, 10)), 8), " paper 0.387978")

# ---------------------------------------------------------------- Lemma minors
D3 = sp.diag(2, 2, 0)
check("Lemma minors: diag(2,2,0): 2J=JJ^dagger, det 0, tr(J/2)=2, rank 2",
      2*D3 == D3*D3.H and D3.det() == 0 and (D3/2).trace() == 2 and D3.rank() == 2)
D4 = sp.diag(2, 2, 0, 0)
mx = max(D4.extract([i, j], [kk, l]).det() for i in range(4) for j in range(i+1, 4) for kk in range(4) for l in range(kk+1, 4))
check("Table readout: diag(2,2,0,0): det 0, largest 2x2 minor 4, tr P 2", D4.det() == 0 and mx == 4 and (D4/2).trace() == 2)
# 2J = J J^dagger => J Hermitian (JJ^dagger is Hermitian) and J^2 = 2J: check on a rank-one projector family
psi = Matrix([R(3, 5), R(4, 5)*I])
Jp1 = 2*psi*psi.H
check("Lemma minors: J=2 psi psi^dagger satisfies 2J=JJ^dagger, det 0, tr(J/2)=1",
      (2*Jp1 - Jp1*Jp1.H) == zeros(2, 2) and Jp1.det() == 0 and (Jp1/2).trace() == 1)
# generic 2x2: solve 2J = J J^dagger => antihermitian part vanishes
check("Lemma minors: 2J=JJ^dagger forces J=J^dagger (2(J-J^dagger) = JJ^dagger-(JJ^dagger)^dagger = 0)",
      ((J*J.H) - (J*J.H).H).applyfunc(expand) == zeros(2, 2))

# ---------------------------------------------------------------- Prop composition
A_ = Matrix(2, 2, symbols('p0:4')); B_ = Matrix(2, 2, symbols('q0:4'))
KP = sp.kronecker_product
lhsK = KP(A_, B_)*KP(A_, B_).H
rhsK = KP(A_*A_.H, B_*B_.H)
check("Prop composition (1): (JA x JB)(JA x JB)^dagger = JA JA^dagger x JB JB^dagger",
      (lhsK - rhsK).applyfunc(expand) == zeros(4, 4))
random.seed(7)
def rnd(): return R(random.randint(-9, 9), random.randint(1, 5)) + I*R(random.randint(-9, 9), random.randint(1, 5))
JAB = Matrix(4, 4, lambda i, j: rnd())
QAB = JAB*JAB.H
e1 = Matrix([R(3, 5), R(4, 5)]); e2 = Matrix([-R(4, 5), R(3, 5)])
f1 = Matrix([1, I])/sqrt(2); f2 = Matrix([1, -I])/sqrt(2)
PA = [e1*e1.H, e2*e2.H]; PB = [f1*f1.H, f2*f2.H]
trQ = QAB.trace()
pmn = [[simplify((KP(PA[m], PB[nn])*QAB).trace()/trQ) for nn in range(2)] for m in range(2)]
QA = Matrix(2, 2, lambda i, kk: sum(QAB[2*i+j, 2*kk+j] for j in range(2)))
QB = Matrix(2, 2, lambda j, l: sum(QAB[2*i+j, 2*i+l] for i in range(2)))
ok = JAB.rank() == 4
ok &= all(Z(sum(pmn[m]) - (PA[m]*QA).trace()/trQ) for m in range(2))
ok &= all(Z(pmn[0][nn] + pmn[1][nn] - (PB[nn]*QB).trace()/trQ) for nn in range(2))
check("Prop composition (2): marginals consistent, random exact rank-4 J_AB, rotated bases", ok)
check("Prop composition: p_mn real, >=0, sum 1",
      all(sp.im(v) == 0 and v >= 0 for row in pmn for v in row) and Z(sum(sum(row) for row in pmn) - 1))
# product statistics
JA = Matrix(2, 2, lambda i, j: rnd()); JB = Matrix(2, 2, lambda i, j: rnd())
Qp = KP(JA, JB)*KP(JA, JB).H
pA = [(PA[m]*JA*JA.H).trace()/(JA*JA.H).trace() for m in range(2)]
pB = [(PB[m]*JB*JB.H).trace()/(JB*JB.H).trace() for m in range(2)]
check("Prop composition (1): product carriers give p(m,n)=pA(m)pB(n) exactly",
      all(Z((KP(PA[m], PB[nn])*Qp).trace()/Qp.trace() - pA[m]*pB[nn]) for m in range(2) for nn in range(2)))
# (3)
h_ = Matrix(symbols('h0:4'))
Phi = Matrix([1, 0, 0, 1])/sqrt(2)
J3 = Phi*h_.H
Q3 = J3*J3.H
hn = (h_.H*h_)[0]
TrB = Matrix(2, 2, lambda i, kk: sum(Q3[2*i+j, 2*kk+j] for j in range(2)))
check("Prop composition (3): J=Phi+ eta^dagger: Q = |eta|^2 Phi+Phi+^dagger (rank one, pure)",
      (Q3 - hn*Phi*Phi.H).applyfunc(expand) == zeros(4, 4) and J3.subs({h_[0]: 1, h_[1]: 2, h_[2]: 3, h_[3]: 5}).rank() == 1)
check("Prop composition (3): Tr_B Q = |eta|^2 I/2 (= I/2 only after normalising by Tr Q)",
      (TrB - hn*eye(2)/2).applyfunc(expand) == zeros(2, 2))

# ---------------------------------------------------------------- summary
print()
print("SUMMARY: %d checks, %d PASS, %d FAIL" % (len(RES), sum(o for _, o in RES), sum(not o for _, o in RES)))
for lab, o in RES:
    if not o:
        print("  FAILED:", lab)
