# Agent A: Thm rotation, Table modes, Cor dispersion / Table mass, Thm unitary / Table diffusion.
import mpmath as mp
from A_common import *

mp.mp.dps = 80
I = mp.mpc(0, 1)
ns = mp.nstr
pi4 = mp.pi / 4

print("=== Thm rotation: co-rotating root of the exact F ===")
roots = {}
for g in [mp.mpf(1), mp.mpf('0.9'), mp.mpf('0.5')]:
    print("ghat =", ns(g, 3))
    for e in range(2, 13):
        eps = mp.mpf(10) ** (-e)
        r = corot_root(eps, g)
        res = abs(F_phi(r, 1 - eps, g))
        roots[(g, e)] = r
        pp = paper_phi(eps, g)
        print("  eps=1e-%02d  Re phi=%s  Im phi=%s  |F|=%s  Re/sqrt(eps g)-1=%s  (Im-paper Im)/paper Im=%s" % (
            e, ns(r.real, 18), ns(r.imag, 18), ns(res, 3),
            ns(r.real / mp.sqrt(eps * g) - 1, 8), ns((r.imag - pp.imag) / pp.imag, 8)))

print()
print("=== Coefficient extraction, ghat = 1 ===")
print("Im phi/eps^1.5 -> pi/4 ? ; fit Im/eps^1.5 = c0 + sqrt(eps)(c1 ln eps + c2)")
mp.mp.dps = 140
def im_ratio(e, g=mp.mpf(1)):
    eps = mp.mpf(10) ** (-e)
    r = corot_root(eps, g)
    return eps, r
for e in [7, 12, 20, 30, 40]:
    eps, r = im_ratio(e)
    A = r.imag / eps ** 1.5
    print("  eps=1e-%d  Im/eps^1.5 = %s   minus pi/4 = %s   (that)/(sqrt(eps) ln eps) = %s" % (
        e, ns(A, 20), ns(A - pi4, 8), ns((A - pi4) / (mp.sqrt(eps) * mp.log(eps)), 12)))
# three-point solve for c0, c1, c2 with no assumption on c0
def solve3(es, g=mp.mpf(1), lead=None):
    rows = []; rhs = []
    for e in es:
        eps, r = im_ratio(e, g)
        y = (r.imag - (lead(eps) if lead else 0)) / (eps ** 1.5 * mp.sqrt(g))
        rows.append([1, mp.sqrt(eps) * mp.log(eps), mp.sqrt(eps)]); rhs.append(y)
    return mp.lu_solve(mp.matrix(rows), mp.matrix(rhs))
c = solve3([30, 35, 40])
print("  3-point fit (eps=1e-30,1e-35,1e-40): c0=%s (pi/4=%s, c0-pi/4=%s)  c1=%s  c2=%s" % (
    ns(c[0], 22), ns(pi4, 22), ns(c[0] - pi4, 5), ns(c[1], 12), ns(c[2], 12)))
print("  candidates: c1 = -1/4 ; c2 = ? ; (pi/4-related:", ns(pi4, 8), ns(mp.pi / 8, 8), ns(mp.pi ** 2 / 16, 8), ")")

print()
print("Re phi/sqrt(eps) - 1 = eps (a ln eps + b): prediction from my own expansion a=1/4, b=1/24")
for (e1, e2) in [(10, 12), (20, 25), (35, 40)]:
    x1, r1 = im_ratio(e1); x2, r2 = im_ratio(e2)
    y1 = (r1.real / mp.sqrt(x1) - 1) / x1; y2 = (r2.real / mp.sqrt(x2) - 1) / x2
    a = (y1 - y2) / (mp.log(x1) - mp.log(x2)); b = y1 - a * mp.log(x1)
    print("  eps=1e-%d,1e-%d: a=%s  b=%s (1/24=%s)" % (e1, e2, ns(a, 12), ns(b, 12), ns(mp.mpf(1) / 24, 12)))

print()
print("=== Coefficient extraction, ghat < 1 ===")
for g in [mp.mpf('0.9'), mp.mpf('0.5')]:
    print("ghat =", ns(g, 3))
    for e in [6, 12, 20, 30, 40]:
        eps, r = im_ratio(e, g)
        lead = r.imag / eps
        nxt = (r.imag - eps * (1 - g) / 2) / (eps ** 1.5 * mp.sqrt(g))
        print("  eps=1e-%d  Im/eps=%s ((1-g)/2=%s)   [Im - eps(1-g)/2]/(eps^1.5 sqrt g)=%s  minus pi/4: %s  /(sqrt(eps)ln eps): %s" % (
            e, ns(lead, 14), ns((1 - g) / 2, 4), ns(nxt, 16), ns(nxt - pi4, 6), ns((nxt - pi4) / (mp.sqrt(eps) * mp.log(eps)), 10)))
    c = solve3([30, 35, 40], g, lead=lambda eps: eps * (1 - g) / 2)
    print("  3-point fit: c0=%s (c0-pi/4=%s)  c1=%s  c2=%s" % (ns(c[0], 20), ns(c[0] - pi4, 5), ns(c[1], 12), ns(c[2], 12)))
    pred_b = g / 24 + (1 - g) / 4 - (1 - g) ** 2 / (8 * g)
    x1, r1 = im_ratio(35, g); x2, r2 = im_ratio(40, g)
    y1 = (r1.real / mp.sqrt(x1 * g) - 1) / x1; y2 = (r2.real / mp.sqrt(x2 * g) - 1) / x2
    a = (y1 - y2) / (mp.log(x1) - mp.log(x2)); b = y1 - a * mp.log(x1)
    print("  Re phi/sqrt(eps g) - 1 = eps(a ln eps + b): a=%s  b=%s ; my expansion: a=1/4, b=(1/4)ln g + g/24 + (1-g)/4 - (1-g)^2/(8g) = %s" % (
        ns(a, 12), ns(b, 12), ns(pred_b + mp.log(g) / 4, 12)))

mp.mp.dps = 60
print()
print("=== Table modes (k=0, ghat=1) ===")
paper_modes = {2: ('0.09897219', '0.989722', '8.922e-4', '9.015e-3', '0.90146'),
               3: ('0.03157031', '0.998341', '2.652e-5', '8.401e-4', '0.84005'),
               4: ('0.00999775', '0.999775', '8.082e-7', '8.084e-5', '0.80843'),
               5: ('0.00316219', '0.999972', '2.512e-8', '7.945e-6', '0.79450'),
               6: ('0.00100000', '0.999997', '7.888e-10', '7.889e-7', '0.78885'),
               7: ('0.00031623', '1.000000', '2.488e-11', '7.867e-8', '0.78667')}
for e in range(2, 8):
    eps = mp.mpf(10) ** (-e)
    r = corot_root(eps, 1)
    mine = (mp.nstr(r.real, 8, min_fixed=-20), mp.nstr(r.real / mp.sqrt(eps), 7), mp.nstr(r.imag, 5), mp.nstr(r.imag / r.real, 5),
            mp.nstr(r.imag / r.real / eps, 6))
    print("  row eps=1e-%d  mine: rot=%.8f ratio=%.6f decay=%.4e perrad=%.4e /eps=%.5f" % (
        e, r.real, r.real / mp.sqrt(eps), r.imag, r.imag / r.real, r.imag / r.real / eps))
    print("              paper: rot=%s ratio=%s decay=%s perrad=%s /eps=%s" % paper_modes[e])
print("  remark '0.78667 at 1e-7 tends to pi/4': decay per radian / eps at smaller eps:")
for e in [7, 8, 10, 12, 16, 20]:
    eps = mp.mpf(10) ** (-e)
    r = corot_root(eps, 1)
    q = r.imag / r.real / eps
    print("    eps=1e-%d: %s   minus pi/4: %s" % (e, ns(q, 12), ns(q - pi4, 5)))

print()
print("=== Table mass (Cor dispersion), eps=1e-5, ghat = exp(-<r^2> k^2/6) ===")
eps = mp.mpf('1e-5')
r0 = corot_root(eps, 1)
paper_mass = [('0.06', '0.10', '0.005000', '1.0000'), ('0.60', '0.10', '0.049988', '0.9998'), ('0.60', '0.20', '0.049950', '0.9990'),
              ('6.00', '0.05', '0.499688', '0.9994'), ('6.00', '0.20', '0.495034', '0.9901')]
for (r2, k, pm, pr) in paper_mass:
    r2m = mp.mpf(r2); km = mp.mpf(k)
    g = mp.exp(-r2m * km ** 2 / 6)
    rk = corot_root(eps, g)
    meas = (r0.real - rk.real) / (r0.real * km ** 2)
    lead = (1 - mp.sqrt(g)) / km ** 2
    print("  <r2>=%s k=%s  mine=%.6f ratio=%.4f | paper=%s ratio=%s | leading-order (1-sqrt g)/k^2=%.6f" % (
        r2, k, meas, meas / (r2m / 12), pm, pr, lead))

print()
print("=== Table diffusion (Thm unitary), <r^2>=0.06 ===")
paper_diff = {2: ('0.09187', '0.09185', '0.09163'), 3: ('0.03081', '0.03080', '0.03073'),
              4: ('0.00992', '0.00992', '0.00990'), 6: ('0.00100', '0.00100', '0.00100')}
r2m = mp.mpf('0.06')
for e in [2, 3, 4, 6]:
    eps = mp.mpf(10) ** (-e)
    r0 = corot_root(eps, 1)
    out = []; pred = []
    for k in ['0.1', '0.3', '1']:
        km = mp.mpf(k)
        g = mp.exp(-r2m * km ** 2 / 6)
        rk = corot_root(eps, g)
        out.append((rk.imag - r0.imag) / (r0.real - rk.real))
        pred.append(mp.sqrt(eps) * (1 + mp.sqrt(g)) / 2 - pi4 * eps)
    print("  eps=1e-%d  mine: %s | paper: %s | predicted formula: %s" % (
        e, ' '.join('%.5f' % x for x in out), ' '.join(paper_diff[e]), ' '.join('%.5f' % x for x in pred)))

print()
print("=== Thm unitary (1): does every mode decay? Im phi > 0 on a grid ===")
mn = None
for e in [2, 4, 8]:
    eps = mp.mpf(10) ** (-e)
    for g in ['1', '0.99', '0.9', '0.5', '0.1', '0.01']:
        r = corot_root(eps, mp.mpf(g))
        ok = abs(F_phi(r, 1 - eps, mp.mpf(g))) < mp.mpf(10) ** (-40)
        print("  eps=1e-%d ghat=%s  phi=%s + %s i  converged=%s  paper phi: %s + %s i" % (
            e, g, ns(r.real, 10), ns(r.imag, 10), ok, ns(paper_phi(eps, g).real, 10), ns(paper_phi(eps, g).imag, 10)))
