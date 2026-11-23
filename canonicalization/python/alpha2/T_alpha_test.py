"""Test of the order-alpha light-cone law at one given alpha (default 1.9965701).
Units hbar = c = 1, tau_c = 1/sqrt 2, so mu = 1 at alpha = 2. Reuses the functions of A_near2-style code,
restated here so the file stands alone."""
import math
import sys
import numpy as np
import mpmath as mp

mp.mp.dps = 50
ALPHA = sys.argv[1] if len(sys.argv) > 1 else "1.9965701"
LONG = (len(sys.argv) > 2 and sys.argv[2] == "long")
TAUC = 1 / mp.sqrt(2)
SQ2 = math.sqrt(2.0)
a = mp.mpf(ALPHA)
delta = a - 2


def D(om, alpha, k):
    s = -1j * mp.mpmathify(om)
    if k == 0:
        return alpha + (alpha - 1) * (TAUC * s) ** (-alpha)
    b = mp.mpf(k)
    return alpha + TAUC ** (-alpha) * ((s - 1j * b) ** (1 - alpha) - (s + 1j * b) ** (1 - alpha)) / (2j * b)


def root(alpha, k):
    k = mp.mpf(k)
    om0 = mp.sqrt(k * k + 1)
    g = om0 + 1j * mp.pi / 4 * (alpha - 2) / om0
    f = lambda w: D(w, alpha, k)
    r = mp.findroot(f, (g, g * (1 + mp.mpf("1e-3")), g * (1 - mp.mpf("1e-3")) + mp.mpf("1e-4")), solver="muller",
                    tol=mp.mpf(10) ** (-45), maxsteps=200)
    assert abs(f(r)) < mp.mpf(10) ** (-40)
    return r


print("alpha = %s   delta = alpha - 2 = %s" % (ALPHA, mp.nstr(delta, 10)))

# Lorentz defect
h = lambda ph: mp.sin((a - 1) * ph) / ((a - 1) * mp.sin(ph))
grid = [mp.pi / 2 * j / 2000 for j in range(1, 2001)]
dmax = max(abs(h(p) - 1) for p in grid)
print("Lorentz defect: max|h-1| on (0, pi/2] = %s ; leading coefficient alpha(2-alpha)/6 = %s"
      % (mp.nstr(dmax, 6), mp.nstr(a * (2 - a) / 6, 6)))

# roots
print("\n k      Re omega        sqrt(k^2+1)     Im omega          (pi/4) delta mu^2/|omega|   ratio      loss per radian   Q = Re/(2|Im|)")
for ks in ["0", "0.1", "0.5", "1", "2", "5"]:
    k = mp.mpf(ks)
    r = root(a, k)
    w0 = mp.sqrt(k * k + 1)
    form = mp.pi / 4 * delta / w0
    print(" %-5s  %-14s  %-14s  %-16s  %-26s  %-9s  %-16s  %s" % (
        ks, mp.nstr(mp.re(r), 11), mp.nstr(w0, 11), mp.nstr(mp.im(r), 9), mp.nstr(form, 9),
        mp.nstr(mp.im(r) / form, 7), mp.nstr(-mp.im(r) / mp.re(r), 6), mp.nstr(mp.re(r) / (2 * abs(mp.im(r))), 6)))
r0 = root(a, 0)
rho = ((a - 1) / a) ** (1 / a) / TAUC
print("\nk = 0 exact: omega = i rho exp(-i pi/alpha), rho = %s ; Im omega = rho cos(pi/alpha) = %s (root: %s)"
      % (mp.nstr(rho, 12), mp.nstr(rho * mp.cos(mp.pi / a), 12), mp.nstr(mp.im(r0), 12)))
print("rest rate Re omega(0) = %s  (1 at alpha = 2): shift %s" % (mp.nstr(mp.re(r0), 12), mp.nstr(mp.re(r0) - 1, 6)))
print("uniform loss per radian at rest, |Im|/Re = %s ; (pi/4)|delta| = %s" % (mp.nstr(-mp.im(r0) / mp.re(r0), 9), mp.nstr(mp.pi / 4 * abs(delta), 9)))
print("e-folding time of the amplitude at rest: %s rest radians = %s rest cycles"
      % (mp.nstr(mp.re(r0) / abs(mp.im(r0)), 7), mp.nstr(mp.re(r0) / abs(mp.im(r0)) / (2 * mp.pi), 7)))


# time-stepped run at k = 1 and the tail
def source(t):
    return np.where((t > 0) & (t < 2), np.sin(np.pi * t / 2) ** 2, 0.0)


def volterra(alpha, k, hh, T):
    n = int(round(T / hh))
    t = hh * np.arange(n + 1)
    ker = np.zeros(n + 1)
    ker[1:] = t[1:] ** (alpha - 1) * np.sinc(k * t[1:] / np.pi) * SQ2 ** alpha / math.gamma(alpha)
    S = source(t)
    J = np.zeros(n + 1)
    J[0] = S[0] / alpha
    for i in range(1, n + 1):
        integ = hh * (np.dot(ker[1:i], J[i - 1:0:-1]) + 0.5 * ker[i] * J[0])
        J[i] = (S[i] - (alpha - 1) * integ) / alpha
    return t, J


def peaks(t, J, t0, t1):
    ab = np.abs(J)
    idx = np.where((ab[1:-1] > ab[:-2]) & (ab[1:-1] >= ab[2:]) & (t[1:-1] >= t0) & (t[1:-1] <= t1))[0] + 1
    return t[idx], ab[idx]


al = float(a)
kk = 1.0
r1 = root(a, 1)
gam = -float(mp.im(r1))
T = 16000.0 if LONG else 1500.0
hh = 0.05
t, J = volterra(al, kk, hh, T)
tp, ap = peaks(t, J, 200.0, min(T, 1500.0))
fit = np.polyfit(tp, np.log(ap), 1)
A_pole = math.exp(fit[1])
print("\ntime-stepped run, k = 1, step %.2f, to t = %.0f" % (hh, T))
print("measured decay rate on [200, 1500]: %.6e   exact root: %.6e   ratio %.5f" % (-fit[0], gam, -fit[0] / gam))
mp.mp.dps = 30
Stil = mp.quad(lambda x: mp.sin(mp.pi * x / 2) ** 2 * mp.exp(1j * kk * x), [0, 1, 2])
coef = complex(Stil * (-2j * kk) * mp.mpf(2) ** (-mp.mpf(al) / 2) / mp.gamma(1 - mp.mpf(al)))
A_tail = 2 * abs(coef)
print("pole amplitude %.4e ; tail amplitude 2|C| = %.4e  (tail = 2|C| t^-alpha; 1/Gamma(1-alpha) = %.4e, linear in delta)"
      % (A_pole, A_tail, float(1 / mp.gamma(1 - mp.mpf(al)))))
f = lambda x: math.log(A_pole) - gam * x - (math.log(A_tail) - al * math.log(x))
lo, hi = 10.0, 1e7
for _ in range(200):
    mid = 0.5 * (lo + hi)
    if f(mid) > 0:
        lo = mid
    else:
        hi = mid
tc = 0.5 * (lo + hi)
print("predicted crossover (pole envelope = tail envelope): t = %.0f = %.0f rest cycles ; amplitude there %.2e of the initial pole amplitude"
      % (tc, tc * float(mp.re(r0)) / (2 * math.pi), math.exp(-gam * tc)))
if LONG:
    for (t0, t1) in [(2000, 4000), (6000, 8000), (9000, 11000), (13000, 16000)]:
        tq, aq = peaks(t, J, float(t0), float(t1))
        e1 = np.polyfit(tq, np.log(aq), 1)[0]
        p1 = np.polyfit(np.log(tq), np.log(aq), 1)[0]
        print("window [%d, %d]: local exponential rate %.4e (pole %.4e) ; local power-law exponent %.3f ; envelope at end %.3e ; tail formula there %.3e"
              % (t0, t1, -e1, gam, -p1, aq[-1], A_tail * tq[-1] ** (-al)))
