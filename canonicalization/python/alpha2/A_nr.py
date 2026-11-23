# Agent A: Thm KG (1), Prop pair, Thm unitary, Thm NR, Cor feynman, Table NR and the Sec. 9 bullet numbers.
# Independent code (mpmath, 40 digits). hbar = m = 1 in the packet tests.
import random
import mpmath as mp
from A_common import check, info, finish, table_check

mp.mp.dps = 40
TOL = mp.mpf(10) ** (-30)
GL = dict(method='gauss-legendre')

# ------------------------------------------------------------------ Thm KG (1): J = S/2 - (mu^2/2) G_mu*S solves 2J + K_2 J = S (one Fourier mode)
tauc, c, k = mp.mpf('0.9'), mp.mpf('1.3'), mp.mpf('0.7')
b = c * k
S = lambda t: t * t * mp.exp(-t) if t > 0 else mp.mpf(0)


def Jsol(t, mu):
    om = mp.sqrt(b * b + mu * mu)
    return S(t) / 2 - mu * mu / 2 * mp.quad(lambda tp: mp.sin(om * (t - tp)) / om * S(tp), mp.linspace(0, t, 5), **GL)


def residual(t, mu):
    mem = tauc ** (-2) * mp.quad(lambda tau: mp.sin(b * tau) / b * Jsol(t - tau, mu), mp.linspace(0, t, 5), **GL)
    return 2 * Jsol(t, mu) + mem - S(t)


mu_ok = 1 / (mp.sqrt(2) * tauc)
resids = [residual(mp.mpf(t), mu_ok) for t in ('0.5', '2', '5')]
worst = max(abs(r) for r in resids)
check("Thm KG (1) solution formula", worst < TOL, "2J + K_2 J - S for J = S/2 - (mu^2/2) G_mu*S, mode k=0.7, c=1.3, tau_c=0.9, mu = 1/(sqrt2 tau_c): residuals at t=0.5,2,5: %s (S there: %s)" % (", ".join(mp.nstr(r, 3) for r in resids), ", ".join(mp.nstr(S(mp.mpf(t)), 6) for t in ('0.5', '2', '5'))))
bad = abs(residual(mp.mpf(2), 1 / tauc))
check("Thm KG mass constant is sharp", bad > mp.mpf('1e-3'), "same formula with mu tau_c = 1 instead of 1/sqrt2 leaves residual %s" % mp.nstr(bad, 6))

# ------------------------------------------------------------------ Prop pair
random.seed(12345)


def M(om, t):
    return mp.matrix([[mp.cos(om * t), mp.sin(om * t) / om], [-om * mp.sin(om * t), mp.cos(om * t)]])


worst = mp.mpf(0)
t1, t2 = mp.mpf('0.7'), mp.mpf('1.9')
for _ in range(200):
    kk = mp.mpf(random.uniform(-20, 20))
    om = mp.sqrt(kk * kk + 1)
    worst = max(worst, mp.norm(M(om, t1 + t2) - M(om, t2) * M(om, t1)))
check("Prop pair composition", worst < mp.mpf(10) ** (-34), "max ||M(t1+t2) - M(t2)M(t1)|| over 200 random modes at 40 digits: %s (paper prints 1.1e-14, which is double-precision rounding noise and not reproducible digit for digit)" % mp.nstr(worst, 3))

# ------------------------------------------------------------------ packet psi ~ exp(-x^2/8 + i x): |psi^(k)|^2 = (2/sqrt pi) exp(-4 (k-1)^2)
rho = lambda q: 2 / mp.sqrt(mp.pi) * mp.exp(-4 * (q - 1) ** 2)
KGRID = [mp.mpf(x) for x in range(-12, 15)]


def avg(f):
    return mp.quad(lambda q: rho(q) * f(q), KGRID, **GL)


check("packet normalization", abs(avg(lambda q: 1) - 1) < TOL, "int |psi^|^2 dk = 1")
t = mp.mpf(10)
n4 = mp.sqrt(avg(lambda q: q ** 8))
n2 = mp.sqrt(avg(lambda q: q ** 4))
C0 = t / 8 * n4
table_check("Sec9 bullet predicted limit (t/8)||k^4 psi^||", "3.672", C0)


def err_nr(cc):
    def f(q):
        th = t * (mp.sqrt(cc ** 4 + cc * cc * q * q) - cc * cc - q * q / 2)
        return 4 * mp.sin(th / 2) ** 2
    return mp.sqrt(avg(f))


tab = {2: ('6.058e-1', '2.423'), 4: ('2.076e-1', '3.321'), 8: ('5.608e-2', '3.589'), 16: ('1.426e-2', '3.651'), 32: ('3.581e-3', '3.667')}
for cc, (pe, pc2) in tab.items():
    e = err_nr(mp.mpf(cc))
    table_check("Table NR c=%d error" % cc, pe, e)
    table_check("Table NR c=%d error*c^2" % cc, pc2, e * cc * cc)
    check("Thm NR bound c=%d" % cc, e <= C0 / (cc * cc), "error %s <= t ||k^4 psi^||/(8 c^2) = %s" % (mp.nstr(e, 8), mp.nstr(C0 / (cc * cc), 8)))
info("Table NR norm drift column", "both multipliers have modulus exactly 1, so the drift is identically 0; the printed 0 and 2.2e-16 are double-precision rounding and are not a mathematical quantity to reproduce")

# order and coefficient
C1 = t / 16 * avg(lambda q: q ** 10) / n4
print("INFO Thm NR error*c^2 for growing c; predicted C0 - C1/c^2 with C0 = (t/8)||k^4 psi^|| = %s, C1 = (t/16)<k^10>/||k^4 psi^|| = %s" % (mp.nstr(C0, 12), mp.nstr(C1, 12)))
ok = True
last = None
for j in range(1, 13):
    cc = mp.mpf(2) ** j
    e = err_nr(cc)
    c1est = (C0 - e * cc * cc) * cc * cc
    print("INFO   c=%6d  err=%s  err*c^2=%s  (C0-err*c^2)*c^2=%s" % (int(cc), mp.nstr(e, 10), mp.nstr(e * cc * cc, 12), mp.nstr(c1est, 10)))
    last = c1est
    if not e * cc * cc <= C0:
        ok = False
check("Thm NR order O(1/c^2)", ok and abs(last - C1) / C1 < mp.mpf('1e-4'),
      "error*c^2 increases to C0 = %s (the bound of Thm NR is attained in the limit, so the order 1/c^2 is exact, not better); next term -C1/c^2 with C1 = %s (measured %s at c=4096)" % (mp.nstr(C0, 10), mp.nstr(C1, 8), mp.nstr(last, 8)))

# Cor feynman: multiplier (c^2/omega) exp(-i(omega-c^2)t) against exp(-i k^2 t/2)
okc = True
for cc in (2, 4, 8, 16, 32, 256):
    cc = mp.mpf(cc)
    def f(q):
        om = mp.sqrt(cc ** 4 + cc * cc * q * q)
        return abs(cc * cc / om * mp.exp(-1j * (om - cc * cc) * t) - mp.exp(-1j * q * q * t / 2)) ** 2
    e = mp.sqrt(avg(f))
    bound = n2 / (2 * cc * cc) + t * n4 / (8 * cc * cc)
    lim = mp.sqrt((n2 / 2) ** 2 + C0 ** 2)
    info("Cor feynman c=%d" % int(cc), "error %s, bound %s, error*c^2 %s (limit sqrt((||k^2 psi^||/2)^2 + C0^2) = %s)" % (mp.nstr(e, 8), mp.nstr(bound, 8), mp.nstr(e * cc * cc, 8), mp.nstr(lim, 8)))
    if not e <= bound:
        okc = False
check("Cor feynman error bound", okc and abs(e * cc * cc - lim) < mp.mpf('1e-3'), "error <= ||Delta psi||/(2c^2) + t ||Delta^2 psi||/(8c^2) for c = 2..256, and error*c^2 tends to the stated limit")

# Cor feynman: [G_mu]^+ multiplier identity
cc, q = mp.mpf('3.0'), mp.mpf('1.7')
om = mp.sqrt(cc ** 4 + cc * cc * q * q)
tt = mp.mpf('0.83')
pos_part = (mp.sin(om * tt) / om - mp.exp(1j * om * tt) / (2j * om))  # remove the e^{+i omega t} part
lhs = 2 * cc * cc / 1j * mp.exp(1j * cc * cc * tt) * pos_part
rhs = cc * cc / om * mp.exp(-1j * (om - cc * cc) * tt)
check("Cor feynman positive-frequency part", abs(pos_part - 1j / (2 * om) * mp.exp(-1j * om * tt)) < TOL and abs(lhs - rhs) < TOL, "e^{-i omega t} part of sin(omega t)/omega is (i/2omega) e^{-i omega t}; (2mc^2/i hbar) e^{imc^2 t/hbar} [G^_mu]^+ = (mc^2/hbar omega) e^{-i(omega - mc^2/hbar)t}")

# Cor feynman: composition law of K_F in d = 1 (Fresnel integral on the rotated contour y = y0 + e^{i pi/4} sigma)
def KF(tm, x, y, m=mp.mpf('1.3'), hb=mp.mpf('0.7')):
    return mp.sqrt(m / (2j * mp.pi * hb * tm)) * mp.exp(1j * m * (x - y) ** 2 / (2 * hb * tm))


ta, tb, xa, xz = mp.mpf('0.6'), mp.mpf('1.1'), mp.mpf('0.4'), mp.mpf('-1.2')
y0 = (xa * tb + xz * ta) / (ta + tb)
rot = mp.exp(1j * mp.pi / 4)
comp = rot * mp.quad(lambda sg: KF(ta, xa, y0 + rot * sg) * KF(tb, y0 + rot * sg, xz), mp.linspace(-30, 30, 61), **GL)
check("Cor feynman composition law", abs(comp - KF(ta + tb, xa, xz)) < TOL, "int K_F(t;x,y) K_F(t';y,z) dy = K_F(t+t';x,z) including the prefactor (m/2 pi i hbar t)^(1/2), principal branch: diff %s" % mp.nstr(abs(comp - KF(ta + tb, xa, xz)), 3))

# ------------------------------------------------------------------ Thm unitary
cc = mp.mpf(1)
omf = lambda q: mp.sqrt(cc ** 4 + cc * cc * q * q)  # hbar = m = 1
ap = lambda q: mp.sqrt(rho(q))                       # a_+ (real, positive)
am = lambda q: mp.mpf('0.4') * mp.sqrt(rho(q)) * mp.exp(1j * mp.mpf('0.7') * q)  # a_- (my choice)


def norms(tm, mixed):
    def Jh(q):
        v = ap(q) * mp.exp(-1j * omf(q) * tm)
        return v + (am(q) * mp.exp(1j * omf(q) * tm) if mixed else 0)
    def dJh(q):
        v = -1j * omf(q) * ap(q) * mp.exp(-1j * omf(q) * tm)
        return v + (1j * omf(q) * am(q) * mp.exp(1j * omf(q) * tm) if mixed else 0)
    n2_ = mp.quad(lambda q: abs(Jh(q)) ** 2, KGRID, **GL)
    # q(J) = (i hbar / 2 m c^2) int (conj(J) dJ - conj(dJ) J) = -(1/c^2) int Im(conj(J) dJ)
    qch = -mp.quad(lambda q: mp.im(mp.conj(Jh(q)) * dJh(q)), KGRID, **GL) / (cc * cc)
    return n2_, qch


times = [mp.mpf(x) for x in ('0', '1.3', '3.3', '7.1', '10')]
pos = [norms(tm, False) for tm in times]
mix = [norms(tm, True) for tm in times]
sp_pos = max(abs(p[0] - pos[0][0]) for p in pos)
sp_mix = max(p[0] for p in mix) - min(p[0] for p in mix)
sp_q = max(abs(p[1] - mix[0][1]) for p in mix)
check("Thm unitary (2) trace conserved on positive frequency", sp_pos < TOL, "spread of ||J(t)||^2 over 5 times in [0,10]: %s" % mp.nstr(sp_pos, 3))
check("Thm unitary (3) trace not conserved on mixed frequency", sp_mix > mp.mpf('1e-3'), "||J(t)||^2 at t=0,1.3,3.3,7.1,10 with a_- = 0.4 e^{0.7 i k} a_+: %s" % ", ".join(mp.nstr(p[0], 8) for p in mix))
qexp = mp.quad(lambda q: omf(q) / (cc * cc) * (abs(ap(q)) ** 2 - abs(am(q)) ** 2), KGRID, **GL)
check("Thm unitary (4) charge conserved on mixed frequency", sp_q < TOL and abs(mix[0][1] - qexp) < TOL, "q(J) constant to %s and equal to int (hbar omega/mc^2)(|a+|^2-|a-|^2) = %s" % (mp.nstr(sp_q, 3), mp.nstr(qexp, 12)))
okq = True
for cv in ('0.5', '1', '2', '8'):
    cv = mp.mpf(cv)
    qq = avg(lambda q: mp.sqrt(cv ** 4 + cv * cv * q * q) / (cv * cv))
    gap = qq - 1
    bnd = avg(lambda q: q * q) / (2 * cv * cv)
    if not (0 <= gap <= bnd):
        okq = False
    info("Thm unitary (4) c=%s" % mp.nstr(cv, 3), "q - ||J||^2 = %s, bound hbar^2||grad J||^2/(2m^2c^2) = %s" % (mp.nstr(gap, 10), mp.nstr(bnd, 10)))
check("Thm unitary (4) inequality", okq, "0 <= q(J) - ||J||^2 <= hbar^2 ||grad J||^2/(2 m^2 c^2) on the positive-frequency packet, c = 0.5, 1, 2, 8")
qneg = -avg(lambda q: omf(q))
check("Thm unitary (4) sign on negative frequency", qneg < 0, "q = %s < 0" % mp.nstr(qneg, 8))
info("Sec9 bullet trace readout", "the printed spreads 2.5e-16 (positive frequency) and 4.4e-16 (charge) are double-precision zeros; the printed 1.6e-1 for 'a mixed solution' cannot be reproduced from the text because the mixed state (a_+, a_-) and the definition of 'relative spread' are not stated")

finish("A_nr")
