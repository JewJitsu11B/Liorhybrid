# Agent A: Prop symbol, Thm lorentz, Def profile, Table symbol, defect table, Table disp.
# Independent code (mpmath). tau_c = 1 unless stated.
import mpmath as mp
from A_common import check, info, finish, table_check

mp.mp.dps = 45
TOL = mp.mpf(10) ** (-30)


def sinc(z):
    return mp.sin(z) / z if z != 0 else mp.mpf(1)


def m_quad(alpha, s, b):
    """Direct quadrature of Def symbol, tau_c = 1."""
    alpha = mp.mpf(alpha)
    s = mp.mpmathify(s)
    b = mp.mpf(b)
    g = lambda t: mp.exp(-s * t) * sinc(b * t)
    T = 110 / mp.re(s)
    step = mp.pi / (abs(b) + abs(mp.im(s)) + 1)
    n = int(mp.ceil(T / step))
    pts = [k * step for k in range(n + 1)]
    a = pts[1]
    # first interval: substitute tau = x^(1/alpha) to remove the tau^(alpha-1) endpoint singularity
    first = mp.quad(lambda x: g(x ** (1 / alpha)), [0, a ** alpha]) / alpha
    rest = mp.quad(lambda t: t ** (alpha - 1) * g(t), pts[1:], method='gauss-legendre')
    return (first + rest) / mp.gamma(alpha)


def m_closed(alpha, s, b):
    alpha = mp.mpf(alpha)
    s = mp.mpmathify(s)
    b = mp.mpf(b)
    if alpha == 1:
        return mp.atan(b / s) / b
    return ((s - 1j * b) ** (1 - alpha) - (s + 1j * b) ** (1 - alpha)) / (2j * b * (alpha - 1))


def h(alpha, phi):
    alpha = mp.mpf(alpha)
    if alpha == 1:
        return phi / mp.sin(phi)
    return mp.sin((alpha - 1) * phi) / ((alpha - 1) * mp.sin(phi))


# ---------------------------------------------------------------- Def profile
kk, aa = mp.mpf('1.7'), mp.mpf('2.3')
d2 = mp.quad(lambda th: mp.besselj(0, kk * aa * mp.sin(th)) * mp.sin(th), [0, mp.pi / 2])  # r = a sin th
check("Def profile d=2", abs(d2 - sinc(kk * aa)) < TOL, "FT of 2D density by quadrature vs sinc: diff %s" % mp.nstr(d2 - sinc(kk * aa), 3))
d3 = mp.quad(lambda z: mp.cos(kk * aa * z) / 2, [-1, 1])
check("Def profile d=3", abs(d3 - sinc(kk * aa)) < TOL, "FT of uniform sphere vs sinc: diff %s" % mp.nstr(d3 - sinc(kk * aa), 3))
d1 = mp.quad(lambda y: mp.cos(kk * y) / (2 * aa), [-aa, aa])
check("Def profile d=1", abs(d1 - sinc(kk * aa)) < TOL, "FT of uniform segment vs sinc: diff %s" % mp.nstr(d1 - sinc(kk * aa), 3))

# ---------------------------------------------------------------- Prop symbol
alphas = ['0.05', '0.25', '0.5', '0.9', '1', '1.1', '1.5', '1.9', '2', '2.1', '2.5', '3', '4.5', '7']
sb = [(mp.mpf('0.8'), '0.6'), (mp.mpf('0.3'), '2.0'), (mp.mpf('2'), '0.1'),
      (mp.mpc('0.5', '1.5'), '1.0'), (mp.mpc('0.2', '-0.7'), '3')]
worst = mp.mpf(0)
for al in alphas:
    w_al = mp.mpf(0)
    for s, b in sb:
        q = m_quad(al, s, b)
        c = m_closed(al, s, b)
        rel = abs(q - c) / abs(c)
        w_al = max(w_al, rel)
        if mp.im(s) == 0:
            R = mp.sqrt(s ** 2 + mp.mpf(b) ** 2)
            phi = mp.atan2(mp.mpf(b), s)
            pol = R ** (-mp.mpf(al)) * h(al, phi)
            w_al = max(w_al, abs(pol - c) / abs(c))
    worst = max(worst, w_al)
    check("Prop symbol alpha=%s" % al, w_al < TOL,
          "quadrature vs closed form vs polar form over 5 (s,k), 2 with complex s: max rel diff %s" % mp.nstr(w_al, 3))

# alpha -> 1 limit of the general formula and the log form at complex s
s, b = mp.mpc('0.5', '1.5'), mp.mpf(1)
lim = m_closed(mp.mpf(1) + mp.mpf(10) ** (-15), s, b)
check("Prop symbol alpha=1 limit", abs(lim - m_closed(1, s, b)) < mp.mpf(10) ** (-13),
      "general formula at alpha=1+1e-15 vs arctan(b/s)/b at complex s: diff %s" % mp.nstr(abs(lim - m_closed(1, s, b)), 3))
logform = (mp.log(s + 1j * b) - mp.log(s - 1j * b)) / (2j * b)
check("Prop symbol alpha=1 log form", abs(logform - m_closed(1, s, b)) < TOL, "[log(s+ib)-log(s-ib)]/(2ib) = arctan(b/s)/b at complex s: diff %s" % mp.nstr(abs(logform - m_closed(1, s, b)), 3))
# k = 0
for al in ['0.5', '2', '3.3']:
    s = mp.mpf('0.7')
    q = m_quad(al, s, 0)
    check("Prop symbol k=0 alpha=%s" % al, abs(q - s ** (-mp.mpf(al))) / abs(q) < TOL, "quadrature vs s^-alpha: rel diff %s" % mp.nstr(abs(q - s ** (-mp.mpf(al))) / abs(q), 3))
# alpha -> 0+
c0 = m_closed(mp.mpf(10) ** (-20), mp.mpf('0.8'), '0.6')
check("Thm lorentz alpha->0+", abs(c0 - 1) < mp.mpf(10) ** (-18), "closed form at alpha=1e-20 is 1 + %s" % mp.nstr(c0 - 1, 3))

# Re s = 0 boundary value (Def symbol, carrier value). Conditionally convergent only for alpha < 2.
def boundary_quad(alpha, om, b):
    # int_0^inf tau^(alpha-2) e^{i om tau} sin(b tau) dtau / (b Gamma(alpha)), 1 < alpha < 2
    alpha = mp.mpf(alpha)
    p = alpha - 2
    def piece(a):  # int_0^inf tau^p e^{i a tau} dtau, -1 < p < 0
        sg = 1 if a > 0 else -1
        a = abs(a)
        P = 2 * mp.pi / a
        q = p + 1
        head = mp.quad(lambda x: mp.exp(1j * a * x ** (1 / q)), [0, P ** q]) / q
        c = mp.quadosc(lambda t: t ** p * mp.cos(a * t), [P, mp.inf], omega=a)
        sn = mp.quadosc(lambda t: t ** p * mp.sin(a * t), [P, mp.inf], omega=a)
        v = head + c + 1j * sn
        return v if sg > 0 else mp.conj(v)
    return (piece(om + b) - piece(om - b)) / (2j * b * mp.gamma(alpha))

mp.mp.dps = 25
for (al, om, b) in [('1.5', mp.mpf('1.3'), mp.mpf('0.6')), ('1.5', mp.mpf('0.3'), mp.mpf('0.6')), ('1.8', mp.mpf('-1.3'), mp.mpf('0.6'))]:
    q = boundary_quad(al, om, b)
    c = m_closed(al, -1j * om, b)
    check("Def symbol Re s=0 alpha=%s omega=%s" % (al, mp.nstr(om, 3)), abs(q - c) / abs(c) < mp.mpf(10) ** (-12),
          "oscillatory quadrature on the imaginary axis vs principal-branch closed form: rel diff %s (value %s)" % (mp.nstr(abs(q - c) / abs(c), 3), mp.nstr(c, 8)))
mp.mp.dps = 45
# phase claim used in Thm near2: on real timelike omega, arg m_alpha = sgn(omega) alpha pi/2
for al in ['1.5', '1.9', '2.1', '2.6']:
    for om in [mp.mpf('1.3'), mp.mpf('-1.3')]:
        c = m_closed(al, -1j * om, mp.mpf('0.6'))
        target = mp.exp(1j * mp.sign(om) * mp.mpf(al) * mp.pi / 2)
        check("Thm near2 phase alpha=%s omega=%s" % (al, mp.nstr(om, 3)), abs(c / abs(c) - target) < TOL,
              "m_alpha/|m_alpha| vs exp(i sgn(omega) alpha pi/2): diff %s" % mp.nstr(abs(c / abs(c) - target), 3))

# ---------------------------------------------------------------- Thm lorentz
# (a) by direct quadrature: same R, different angle
for al in ['0.5', '1', '1.5', '1.9', '2', '2.1', '3']:
    q1 = m_quad(al, mp.mpf('0.8'), '0.6')
    q2 = m_quad(al, mp.mpf('0.6'), '0.8')
    d = abs(q1 - q2)
    if al == '2':
        check("Thm lorentz alpha=2 invariance (quadrature)", d < TOL and abs(q1 - 1) < TOL, "m_2 at (0.8,0.6) and (0.6,0.8), both R=1: %s, diff %s" % (mp.nstr(q1, 12), mp.nstr(d, 3)))
    else:
        check("Thm lorentz alpha=%s non-invariance (quadrature)" % al, d > mp.mpf('1e-3'), "m at (0.8,0.6) vs (0.6,0.8), same R: differ by %s" % mp.nstr(d, 6))

# (b) defect tabulation
phis = [mp.mpf(x) for x in ['0.2', '0.6', '1.0', '1.4']] + [mp.pi / 2]
grid = [mp.pi / 2 * j / 400 for j in range(1, 401)]
print("INFO Thm lorentz defect table h_alpha(phi)-1 at phi = 0.2, 0.6, 1.0, 1.4, pi/2 and max over (0,pi/2]")
maxdef = {}
for al in ['0', '0.25', '0.5', '1', '1.5', '1.9', '1.99', '2', '2.01', '2.1', '2.5', '3', '4']:
    vals = [h(al, p) - 1 for p in phis]
    md = max(abs(h(al, p) - 1) for p in grid)
    maxdef[al] = md
    print("INFO   alpha=%-5s " % al + " ".join("%12s" % mp.nstr(v, 6) for v in vals) + "   max|.|=%s" % mp.nstr(md, 6))
# sign claim and zero set
ok_sign = True
minabs = (mp.inf, None)
for j in range(1, 800):
    al = mp.mpf(j) / 100  # 0.01 .. 7.99
    for p in grid[::20]:
        v = h(al, p) - 1 if al != 1 else p / mp.sin(p) - 1
        if al == 2:
            ok_sign = ok_sign and v == 0
            continue
        if (al < 2 and not v > 0) or (al > 2 and not v < 0):
            ok_sign = False
    if al != 2:
        md = max(abs(h(al, p) - 1) for p in grid[::20])
        if md < minabs[0]:
            minabs = (md, al)
check("Thm lorentz defect sign", ok_sign, "h_alpha(phi)-1 > 0 for 0<alpha<2 and < 0 for 2<alpha<8, all sampled phi in (0,pi/2]; alpha grid step 0.01")
check("Thm lorentz zero set", minabs[0] > mp.mpf('1e-3'), "smallest max-defect over alpha in (0,8) other than alpha=2 is %s at alpha=%s (neighbours of 2); zero only at alpha=2 (and 0)" % (mp.nstr(minabs[0], 4), mp.nstr(minabs[1], 4)))
check("Thm lorentz symmetry", all(abs(h(mp.mpf(a), p) - h(2 - mp.mpf(a), p)) < TOL for a in ['0.3', '1.7', '2.6'] for p in phis), "h_alpha = h_(2-alpha) at sampled points")

# (c) small-angle coefficients
mp.mp.dps = 70
phi = mp.mpf(10) ** (-8)
okc = True
worst2 = worst4 = mp.mpf(0)
for al in ['0.3', '1', '1.5', '2.5', '3', '5']:
    a = mp.mpf(al)
    c2 = a * (2 - a) / 6
    c4 = a * (2 - a) * (7 - 3 * (a - 1) ** 2) / 360
    hv = h(a, phi)
    e2 = abs((hv - 1) / phi ** 2 - c2)
    e4 = abs(((hv - 1) / phi ** 2 - c2) / phi ** 2 - c4)
    worst2, worst4 = max(worst2, e2), max(worst4, e4)
check("Thm lorentz leading coefficient", worst2 < mp.mpf(10) ** (-14), "(h-1)/phi^2 -> alpha(2-alpha)/6 at phi=1e-8: max deviation %s" % mp.nstr(worst2, 3))
check("Thm lorentz next term", worst4 < mp.mpf(10) ** (-12), "phi^4 coefficient = alpha(2-alpha)(7-3(alpha-1)^2)/360: max deviation %s" % mp.nstr(worst4, 3))
mp.mp.dps = 45

# ---------------------------------------------------------------- Table symbol
rows = {'1.5': ['1.054093', '1.054093', '1.0050', '1.0468', '1.1395', '1.3075'],
        '2.0': ['1.000000', '1.000000', '1.0000', '1.0000', '1.0000', '1.0000'],
        '2.5': ['0.913547', '0.913547', '0.9917', '0.9249', '0.7903', '0.5840'],
        '3.0': ['0.800000', '0.800000', '0.9801', '0.8253', '0.5403', '0.1700']}
for al, pr in rows.items():
    q = m_quad(al, mp.mpf('0.8'), '0.6')
    c = m_closed(al, mp.mpf('0.8'), '0.6')
    table_check("Table symbol alpha=%s numeric" % al, pr[0], mp.re(q))
    table_check("Table symbol alpha=%s closed" % al, pr[1], mp.re(c))
    for j, p in enumerate(['0.2', '0.6', '1.0', '1.4']):
        table_check("Table symbol alpha=%s h(%s)" % (al, p), pr[2 + j], h(al, mp.mpf(p)))

# defect row of Sec. 9
defrow = {'0': '0', '0.5': '0.41', '1.5': '0.41', '1.9': '0.097', '2': '0', '2.1': '0.10', '3': '1.0'}
for al, pr in defrow.items():
    md = max(abs(h(al, p) - 1) for p in grid)
    table_check("Sec9 defect row alpha=%s" % al, pr, md)

# ---------------------------------------------------------------- Table disp (Thm KG (3), Remark phase)
tauc = 1 / mp.sqrt(2)


def T2_symbol(E, k, sign=1):
    s = -1j * E
    if k == 0:
        m = (tauc * s) ** (-2)
    else:
        m = tauc ** (-2) * ((s - 1j * k) ** (-1) - (s + 1j * k) ** (-1)) / (2j * k)
    return 2 + sign * m


def bisect(f, lo, hi, n=180):
    flo = f(lo)
    assert flo * f(hi) < 0
    for _ in range(n):
        mid = (lo + hi) / 2
        fm = f(mid)
        if (fm < 0) == (flo < 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


disp = {'0.0': ('1.0000000000', '-1.00'), '0.5': ('1.1180339887', '-0.75'), '1.0': ('1.4142135624', '0.00'),
        '2.0': ('2.2360679775', '3.00'), '4.0': ('4.1231056256', '15.00')}
for ks, (pe, pt) in disp.items():
    k = mp.mpf(ks)
    E = bisect(lambda e: mp.re(T2_symbol(e, k)), k + mp.mpf('1e-9'), mp.mpf(50))
    table_check("Table disp k=%s root E" % ks, pe, E)
    table_check("Table disp k=%s sqrt(k^2+mu^2)" % ks, pe, mp.sqrt(k * k + 1))
    # added factor -1: solve for W = E^2 through s^2 = -W
    W = bisect(lambda w: mp.re(T2_symbol(mp.sqrt(mp.mpc(w)), k, -1)), k * k - mp.mpf(40), k * k - mp.mpf('1e-9'))
    table_check("Table disp k=%s E^2 with factor -1" % ks, pt, W)
# tachyon growth rate checked against the DEFINITION (quadrature symbol), real s > 0
for ks in ['0', '0.5']:
    k = mp.mpf(ks)
    sroot = mp.sqrt(1 - k * k)
    val = 2 - tauc ** (-2) * m_quad(2, sroot, k)
    check("Remark phase tachyon k=%s" % ks, abs(val) < TOL, "2 - K_2 symbol (by quadrature) vanishes at real s = sqrt(mu^2-k^2) = %s: residual %s" % (mp.nstr(sroot, 8), mp.nstr(abs(val), 3)))
    val2 = 2 + tauc ** (-2) * m_quad(2, sroot, k)
    check("Thm KG no growing mode k=%s" % ks, abs(val2) > 1, "2 + K_2 symbol at the same s is %s (no root)" % mp.nstr(val2, 8))

finish("A_symbol")
