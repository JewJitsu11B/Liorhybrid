# Agent A: Lemma weyl, Def weyl, Lemma W (eq. jonquiere).
import mpmath as mp
from A_common import *

mp.mp.dps = 45
I = mp.mpc(0, 1)
nd = lambda x: mp.nstr(x, 6)

print("=== Lemma weyl: int_0^inf s^(alpha-1) e^{i omega s} ds vs Gamma(alpha)(-i omega)^(-alpha) ===")
for alpha in [mp.mpf('0.1'), mp.mpf('0.3'), mp.mpf('0.5'), mp.mpf('0.9'), mp.mpf('0.99')]:
    for omega in [mp.mpf(1), mp.mpf(-1), mp.mpf('2.5'), mp.mpf('-0.4')]:
        aw = abs(omega)
        # head [0, T0] by tanh-sinh, tail by quadosc on cos and sin separately
        T0 = 2 * mp.pi / aw
        # substitution s = u^(1/alpha) removes the endpoint singularity
        head = mp.quad(lambda u: mp.exp(I * omega * u ** (1 / alpha)), mp.linspace(0, T0 ** alpha, 9)) / alpha
        tc = mp.quadosc(lambda s: s ** (alpha - 1) * mp.cos(aw * s), [T0, mp.inf], omega=aw)
        ts = mp.quadosc(lambda s: s ** (alpha - 1) * mp.sin(aw * s), [T0, mp.inf], omega=aw)
        val = head + tc + mp.sign(omega) * I * ts
        claim = mp.gamma(alpha) * (-I * omega) ** (-alpha)
        # Def weyl symbol/phase: (-iE)^(-alpha) = |E|^-alpha exp(i sgn(E) alpha pi/2)
        sym = abs(omega) ** (-alpha) * mp.exp(I * mp.sign(omega) * alpha * mp.pi / 2)
        print("alpha=%s omega=%s  |quad-claim|=%s  |principal power - Def weyl phase form|=%s" % (
            mp.nstr(alpha, 3), mp.nstr(omega, 3), nd(abs(val - claim)), nd(abs((-I * omega) ** (-alpha) - sym))))

print()
print("=== Lemma weyl at alpha >= 1 (and the endpoint alpha <= 0): partial integrals int_0^T ===")
def partial(alpha, omega, T):
    # exact partial integral by piecewise quadrature over half-periods
    n = int(mp.ceil(T * abs(omega) / mp.pi))
    pts = [mp.mpf(k) * T / n for k in range(n + 1)]
    return mp.quad(lambda s: s ** (alpha - 1) * mp.exp(I * omega * s), pts)
mp.mp.dps = 20
for alpha in [mp.mpf('0.5'), mp.mpf(1), mp.mpf('1.5')]:
    claim = mp.gamma(alpha) * (-I * 1) ** (-alpha)
    print("alpha=%s  formula value %s" % (mp.nstr(alpha, 3), mp.nstr(claim, 8)))
    for T in [10, 10.5 , 100, 100.5, 400, 400.5]:
        p = partial(alpha, mp.mpf(1), mp.mpf(T))
        print("   T=%-6s partial=%s   |partial - formula|=%s" % (T, mp.nstr(p, 8), nd(abs(p - claim))))
# Abel-regularized value at alpha >= 1 still equals the formula
for alpha in [mp.mpf(1), mp.mpf('1.5')]:
    for d in [mp.mpf('1e-2'), mp.mpf('1e-4')]:
        ab = mp.gamma(alpha) * (d - I) ** (-alpha)
        print("alpha=%s Abel damping %s: |Abel - formula| = %s" % (mp.nstr(alpha, 3), mp.nstr(d, 2), nd(abs(ab - mp.gamma(alpha) * (-I) ** (-alpha)))))
# near s=0 for alpha=0: divergence
print("alpha=0: int_delta^1 s^-1 cos(s) ds for delta=1e-3,1e-6,1e-9:",
      [mp.nstr(mp.quad(lambda s: mp.cos(s) / s, [d, mp.sqrt(d), 1]), 8) for d in [mp.mpf('1e-3'), mp.mpf('1e-6'), mp.mpf('1e-9')]])

mp.mp.dps = 45
print()
print("=== Lemma W (a): eq. jonquiere continued to z = r e^{i theta}, |z|>1, vs direct summation ===")
worst = 0
for alpha in [mp.mpf('0.2'), mp.mpf('0.5'), mp.mpf('0.9'), mp.mpf('0.999')]:
    for r in [mp.mpf('1.05'), mp.mpf('1.5'), mp.mpf(3)]:
        for th in [mp.mpf('0.3'), mp.mpf(2), mp.mpf('-1.1'), mp.pi, mp.mpf(0), mp.mpf('5.5'), mp.mpf('-7')]:
            z = r * mp.exp(I * th)
            phi = th - I * mp.log(r)          # z = e^{i phi}
            d = abs(W_jonq(phi, alpha) - W_direct(z, alpha))
            worst = max(worst, d)
print("worst |jonquiere - direct| over 84 points:", nd(worst))
# Hurwitz reduction vs term-by-term principal powers (tail ~ K^-alpha, so only a coarse check)
phi = mp.mpf('0.7') - I * mp.mpf('0.2'); alpha = mp.mpf('0.5')
for K in [10**3, 10**4]:
    print("term-by-term k-sum |k|<=%d vs Hurwitz form: diff %s (expected tail ~ %s)" % (
        K, nd(abs(W_jonq_ksum(phi, alpha, K) - W_jonq(phi, alpha))),
        nd(abs(1 - mp.exp(-I * phi)) * 2 * (2 * mp.pi) ** (-alpha - 1) * K ** (-alpha) / alpha)))

print()
print("=== Lemma W (b): unit circle, vs (i) Levin-accelerated series, (ii) Bose-type integral ===")
def W_bose(x, alpha):
    # Li_{-a}(x) = (1/Gamma(1-a)) int_0^inf t^-a x e^-t/(1 - x e^-t)^2 dt  (0<a<1, x not in [1,inf))
    g = lambda t: x * mp.exp(-t) / (1 - x * mp.exp(-t)) ** 2
    b = 1 - alpha   # t = u^(1/b) removes the t^-alpha endpoint singularity on [0,1]
    li = mp.quad(lambda u: g(u ** (1 / b)), mp.linspace(0, 1, 9)) / b
    li += mp.quad(lambda t: t ** (-alpha) * g(t), [1, 2, 4, 8, 16, 32, 64, 128, 250])
    li /= mp.gamma(1 - alpha)
    return (1 - x) * li / mp.gamma(1 + alpha)
worst_b = 0; worst_l = 0
for alpha in [mp.mpf('0.2'), mp.mpf('0.5'), mp.mpf('0.9')]:
    for th in [mp.mpf('0.3'), mp.mpf(1), mp.mpf(2), mp.mpf(3), mp.mpf('-1.3'), mp.mpf(4)]:
        x = mp.exp(-I * th)
        wj = W_jonq(th, alpha)
        db = abs(wj - W_bose(x, alpha)); worst_b = max(worst_b, db)
        lev = mp.nsum(lambda j: w(j, alpha) * mp.exp(-I * th * j), [1, mp.inf], method='levin')
        dl = abs(wj - lev); worst_l = max(worst_l, dl)
print("worst |jonquiere - Bose integral| on unit circle:", nd(worst_b))
print("worst |jonquiere - Levin-accelerated series| on unit circle:", nd(worst_l))
# inside the unit circle (decaying side), continuation: Bose integral is analytic for z off (0,1]
worst_in = 0
for alpha in [mp.mpf('0.5'), mp.mpf('0.9')]:
    for (re, im) in [('0.3', '0.2'), ('2.5', '0.5'), ('-1', '0.1'), ('0.05', '0.01'), ('-0.05', '0.01'), ('-2', '1.5')]:
        phi = mp.mpf(re) + I * mp.mpf(im)
        d = abs(W_jonq(phi, alpha) - W_bose(mp.exp(-I * phi), alpha)); worst_in = max(worst_in, d)
print("worst |jonquiere(principal) - Bose integral| for Im phi>0 (|z|<1), both signs of Re phi:", nd(worst_in))
# the branch cut: jump of the principal-branch formula across Re phi = 0 at Im phi > 0
alpha = mp.mpf('0.5'); y = mp.mpf('0.3'); e = mp.mpf('1e-30')
jump = W_jonq(e + I * y, alpha) - W_jonq(-e + I * y, alpha)
print("jump across Re phi=0 at Im phi=0.3 (z=e^-0.3 in (0,1)), alpha=0.5:", mp.nstr(jump, 10))
jump2 = W_jonq(e - I * y, alpha) - W_jonq(-e - I * y, alpha)
print("jump across Re phi=0 at Im phi=-0.3 (z>1):", nd(abs(jump2)))

print()
print("=== Lemma W (c): vs mpmath.polylog: W = (1-x) Li_{-alpha}(x)/Gamma(1+alpha), x=1/z ===")
worst_c = 0; fails = 0
for alpha in [mp.mpf('0.2'), mp.mpf('0.5'), mp.mpf('0.9')]:
    for (re, im) in [('0.3', '-0.4'), ('2', '-0.1'), ('1', '0'), ('3', '0'), ('-2', '0'), ('1', '-2'), ('0.5', '0.3'), ('-0.5', '0.3')]:
        phi = mp.mpf(re) + I * mp.mpf(im)
        x = mp.exp(-I * phi)
        try:
            pl = (1 - x) * mp.polylog(-alpha, x) / mp.gamma(1 + alpha)
            d = abs(W_jonq(phi, alpha) - pl); worst_c = max(worst_c, d)
            if d > mp.mpf(10) ** (-30):
                print("  alpha=%s phi=%s |x|=%s diff=%s" % (mp.nstr(alpha, 2), mp.nstr(phi, 4), mp.nstr(abs(x), 4), nd(d)))
        except Exception as ex:
            fails += 1
            print("  polylog unavailable at alpha=%s phi=%s |x|=%s: %s" % (mp.nstr(alpha, 2), mp.nstr(phi, 4), mp.nstr(abs(x), 4), type(ex).__name__))
print("worst |jonquiere - polylog| where polylog evaluated:", nd(worst_c), " polylog failures:", fails)

print()
print("=== Lemma W: non-principal alternative (shows the convention matters) ===")
# take the k=0 power with arg(i phi) in (-2pi,0] instead of principal, at Re phi<0 on the circle
alpha = mp.mpf('0.5'); th = mp.mpf('-1.3')
s = alpha + 1
wrong_t0 = mp.exp(-s * (mp.log(abs(th)) + I * (mp.pi / 2)))   # arg(i*th) taken as +pi/2 instead of -pi/2
right_t0 = (I * th) ** (-s)
z = mp.mpf('1.0000001') * mp.exp(I * th)
print("phi=-1.3: W with arg(i phi)=+pi/2 for the k=0 term differs from the true W by",
      nd(abs((1 - mp.exp(-I * th)) * (wrong_t0 - right_t0))), "; principal gives", nd(abs(W_jonq(th, alpha) - W_bose(mp.exp(-I * th), alpha))))
# alpha > 1 (used in Thm feedback): same formula vs direct sum
for alpha in [mp.mpf('1.01'), mp.mpf('1.001'), mp.mpf('1.7')]:
    z = mp.exp(mp.mpf('0.1'))
    print("alpha=%s, z=e^0.1: |jonquiere - direct| = %s" % (mp.nstr(alpha, 5), nd(abs(W_of_z(z, alpha) - W_direct(z, alpha)))))
