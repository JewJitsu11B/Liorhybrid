"""Agent N: probes of the author's script (imported unchanged from scripts/ at the repository root)
to pin down where the paper's description and the script differ."""
import sys, os, math
import numpy as np
from math import comb
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(here, '..', '..', '..', 'scripts')))
import cal_feynman_alpha2 as A
from N_sim import fit_damped, root_near

FAILS = []


def check(ok, label, text):
    print(("PASS " if ok else "FAIL ") + label + " " + text, flush=True)
    if not ok:
        FAILS.append(label)


# (a) checkerboard recursion of the script vs the binomial formula of Prop checker(3)
T, eps = 8.0, 0.5
x, R, L = A.checkerboard(1.0, T, eps)
N = int(round(T / eps)); worst = 0
for xi, Li in zip(x, L):
    NR = int(round((T + xi) / (2 * eps))); NL = int(round((T - xi) / (2 * eps)))
    f = sum((1j * eps) ** (2 * n + 1) * comb(NR - 1, n) * comb(NL - 1, n) for n in range(0, N))
    worst = max(worst, abs(f - Li))
check(worst < 1e-12, "Prop checker(3) lattice formula vs author's recursion", "max |L_recursion - sum_n (i eps mu)^(2n+1) C(N_R-1,n) C(N_L-1,n)| = %.1e (N=16)" % worst)

# (b) Table 6 row t=8: where does 2.46e-06 come from?
mu, Lg, Ng = 1.0, 12.0, 1201
u, total, target, rec = A.null_leg_path_sum(mu, Lg, Ng, 60)
du = u[1] - u[0]
tail1 = total - 0.5
dU = np.gradient(tail1, du, axis=0); dV = np.gradient(tail1, du, axis=1)
from scipy.special import j1
for t in (4.0, 8.0):
    s_idx = int(round(2 * t / du)); rows = []
    for i in range(1, s_idx):
        j = s_idx - i
        if j >= Ng or i >= j - 2:
            continue
        uu, vv = i * du, j * du; r = 0.5 * (vv - uu)
        if r < 0.25 or uu < 0.25:
            continue
        w = uu * vv
        ref = -mu * j1(mu * math.sqrt(w)) / (4 * math.pi * math.sqrt(w))
        rows.append((abs(-(dV[i, j] - dU[i, j]) / (2 * math.pi * r) - ref), j, r))
    rows.sort(reverse=True)
    interior = max(e for e, j, r in rows if j < Ng - 1)
    print("INFO Table 6 t=%g: %d points, r range %.2f..%.2f; max err %.2e at j=%d (grid edge j=%d, one-sided np.gradient); max over interior points %.2e"
          % (t, len(rows), min(r for _, _, r in rows), max(r for _, _, r in rows), rows[0][0], rows[0][1], Ng - 1, interior))
    if t == 8.0:
        check(rows[0][1] == Ng - 1 and interior < 0.1 * rows[0][0], "Table 6 row t=8 max error is a grid-edge one-sided difference",
              "edge point error %.2e vs interior max %.2e; r covered only up to %.2f of 7.75" % (rows[0][0], interior, max(r for _, _, r in rows)))

# (c) time-stepped run of Table 3 caption: samples vs envelope vs rate
tauc = 1 / math.sqrt(2)
for a in (1.95, 2.05):
    t, J = A.time_step(a, 1.0, tauc)
    J = J.real; h = 0.02
    env = lambda tc, hw=3.0: np.abs(J[int((tc - hw) / h):int((tc + hw) / h) + 1]).max()
    g, om = fit_damped(J[int(40 / h):int((120 if a < 2 else 200) / h)], 40, h)
    rt = root_near(a, 1.0, tauc, math.sqrt(2))
    samp = (abs(J[int(50 / 0.02)]), abs(J[int(200 / 0.02)]))
    print("INFO author time_step alpha=%.2f: samples |J(50)|=%.2e |J(200)|=%.2e (rate from samples %.4e); envelopes %.3e, %.3e (rate from envelopes %.4e); fitted rate %.4e; Im root %.4e"
          % (a, samp[0], samp[1], math.log(samp[1] / samp[0]) / 150, env(50), env(200) if a > 2 else env(197), math.log((env(200) if a > 2 else env(197)) / env(50)) / 150 if a > 2 else math.log(env(197) / env(50)) / 147, g, rt.imag))
    check(abs(g / rt.imag - 1) < 0.02, "Table 3 caption time-stepped alpha=%.2f (author's stepper, fitted rate)" % a, "%.4e vs Im root %.4e" % (g, rt.imag))
# author's stepper, longer run below 2: late-time tail
t, J = A.time_step(1.90, 1.0, tauc, T=600.0)
J = J.real; h = 0.02
ev = [np.abs(J[int((tc - 8) / h):int((tc + 8) / h)]).max() for tc in (100, 300, 400, 500, 590)]
pw = [-math.log(ev[i + 1] / ev[i]) / math.log(tt2 / tt1) for i, (tt1, tt2) in enumerate(((100, 300), (300, 400), (400, 500), (500, 590)))]
print("INFO author time_step alpha=1.90 to T=600: envelopes at t=100,300,400,500,590: %s ; local power-law exponents %s ; pure pole decay from t=100 would give %.1e at t=590"
      % (["%.3e" % v for v in ev], ["%.2f" % v for v in pw], ev[0] * math.exp(-5.4354e-2 * 490)))
check(abs(pw[-1] - 1.9) < 0.15, "Thm near2 late-time tail t^-alpha in the author's own stepper (extra)", "exponent %.2f at alpha=1.9" % pw[-1])

# (d) Table 3 caption: growth rate at alpha=2.01, k=5
r5 = root_near(2.01, 5.0, tauc, math.sqrt(26))
r5a = A.newton_c(lambda E: A.F_cal(2.01, E, 5.0, tauc), math.sqrt(26) + 0.001j)
form = (math.pi / 4) * 0.01 * (r5.real ** 2 - 25) / r5.real
print("INFO alpha=2.01 k=5: my root %.6f%+.4ei ; author's F_cal/newton_c root %.6f%+.4ei ; first-order formula %.4e ; paper caption 1.5e-3"
      % (r5.real, r5.imag, r5a.real, r5a.imag, form))
check(abs(r5.imag - r5a.imag) < 1e-9, "Table 3 caption growth rate alpha=2.01 k=5 (two root finders agree)", "Im = %.4e (paper prints 1.5e-3; script prints no such number)" % r5.imag)
print("failed: %s" % (FAILS if FAILS else "none"))
sys.exit(1 if FAILS else 0)
