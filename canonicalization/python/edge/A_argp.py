# Agent A: argument principle. Other roots of F on or outside the unit circle for 0<alpha<1?
import mpmath as mp
from A_common import *

I = mp.mpc(0, 1)


def winding(f, segments, maxstep=mp.mpf('0.4'), stats=None):
    """Total change of arg f along a path, in units of 2 pi (rounded if closed).
    segments: list of (a, b) parameter intervals, or (a, b, map) with path point map(u).
    Adaptive bisection keeps each phase step below maxstep."""
    total = mp.mpf(0)
    minabs = mp.inf
    for seg in segments:
        a, b = seg[0], seg[1]
        mapf = seg[2] if len(seg) > 2 else (lambda u: u)
        n0 = 64
        us = [a + (b - a) * mp.mpf(k) / n0 for k in range(n0 + 1)]
        vals = [f(mapf(u)) for u in us]
        stack = [(us[k], vals[k], us[k + 1], vals[k + 1], 0) for k in range(n0 - 1, -1, -1)]
        while stack:
            u0, v0, u1, v1, depth = stack.pop()
            minabs = min(minabs, abs(v0), abs(v1))
            d = mp.arg(v1 / v0)
            if abs(d) > maxstep and depth < 60:
                um = (u0 + u1) / 2
                vm = f(mapf(um))
                stack.append((um, vm, u1, v1, depth + 1))
                stack.append((u0, v0, um, vm, depth + 1))
            else:
                total += d
    if stats is not None:
        stats['minabs'] = minabs
        stats['raw'] = total / (2 * mp.pi)
    return int(mp.nint(total / (2 * mp.pi)))


def count_outside(alpha, ghat, delta=mp.mpf('1e-12'), Y=mp.mpf(6)):
    """Zeros of F(e^{i phi}) in -pi<Re phi<=pi, -Y<Im phi<0 (1<|z|<e^Y), by the argument principle.
    Boundary (counter-clockwise in phi): bottom Im=-Y left->right; top = unit circle right->left with a
    semicircular indentation of radius delta around phi=0 into Im phi<0. Vertical sides cancel by periodicity."""
    f = lambda p: F_phi(p, alpha, ghat)
    st = {}
    L = mp.log
    segs = [
        (-mp.pi, mp.pi, lambda u: u - I * Y),                       # big circle |z|=e^Y
        (L(mp.pi), L(delta), lambda u: mp.exp(u)),                  # unit circle, phi: pi -> delta (log-spaced)
        (mp.mpf(0), -mp.pi, lambda th: delta * mp.exp(I * th)),     # indentation through -i delta
        (L(delta), L(mp.pi), lambda u: -mp.exp(u)),                 # unit circle, phi: -delta -> -pi
    ]
    n = winding(f, segs, stats=st)
    return n, st


def count_inside(alpha, ghat, delta, Y2):
    """Zeros with delta<Re phi<2pi-delta, 0<Im phi<Y2 (e^-Y2<|z|<1, away from the cut z in (0,1))."""
    f = lambda p: F_phi(p, alpha, ghat)
    st = {}
    L = mp.log
    tp = 2 * mp.pi
    segs = [
        (L(delta), L(mp.pi), lambda u: mp.exp(u)),                    # bottom, phi: delta -> pi
        (L(mp.pi), L(delta), lambda u: tp - mp.exp(u)),               # bottom, phi: pi -> 2pi-delta
        (L(delta), L(Y2), lambda u: tp - delta + I * mp.exp(u)),      # right side, up (start slightly above 0)
        (tp - delta, delta, lambda u: u + I * Y2),                    # top, right -> left
        (L(Y2), L(delta), lambda u: delta + I * mp.exp(u)),           # left side, down
    ]
    # close the two tiny gaps (Im from 0 to delta) explicitly
    segs.insert(2, (mp.mpf(0), delta, lambda u: tp - delta + I * u))
    segs.append((delta, mp.mpf(0), lambda u: delta + I * u))
    n = winding(f, segs, stats=st)
    return n, st


if __name__ == '__main__':
    mp.mp.dps = 30
    ns = mp.nstr
    print("=== validation of the counter on a known case: flat weights, F=(z^2-(1+a)z+a+(1-a)g)/(z-1) ===")
    for (a, g) in [('0.9', '1'), ('0.9', '0.5'), ('1.2', '1')]:
        a_ = mp.mpf(a); g_ = mp.mpf(g)
        f = lambda th, R=mp.mpf('1.001'): (lambda z: z - a_ + (1 - a_) * g_ / (z - 1))(R * mp.exp(I * th))
        wn = winding(f, [(-mp.pi, mp.pi)])
        # zeros - poles inside |z|<1.001; pole at z=1 counts -1
        print("  alpha=%s ghat=%s: zeros-poles inside |z|<1.001 = %d  => zeros = %d (expected: 2, 2, 1)" % (a, g, wn, wn + 1))

    print()
    print("=== zeros of the exact F with 1 < |z| < e^6 (indented at z=1), 0<alpha<1 ===")
    tot = 0; cnt = 0
    for a in ['0.05', '0.3', '0.5', '0.7', '0.9', '0.99', '0.999', '0.9999', '0.999999']:
        for g in ['1', '0.9', '0.5', '0.1', '0.001', '0']:
            n, st = count_outside(mp.mpf(a), mp.mpf(g))
            tot += abs(n); cnt += 1
            print("  alpha=%-9s ghat=%-6s zeros outside: %d   (raw %s, min|F| on contour %s)" % (a, g, n, ns(st['raw'], 12), ns(st['minabs'], 4)))
    print("  cases: %d, total zeros found on/outside: %d" % (cnt, tot))

    print()
    print("=== min |F| on the unit circle itself (a root ON the circle would give 0) ===")
    for a in ['0.3', '0.9', '0.999', '0.999999']:
        for g in ['1', '0.5']:
            al = mp.mpf(a); gg = mp.mpf(g); eps = 1 - al
            best = (mp.inf, None)
            for k in range(-600, 601):
                # log-spaced both signs
                for sgn in (1, -1):
                    ph = sgn * mp.pi * mp.mpf(10) ** (mp.mpf(-abs(k)) / 60)
                    v = abs(F_phi(ph, al, gg))
                    if v < best[0]:
                        best = (v, ph)
            print("  alpha=%-9s ghat=%s  min|F(e^{i phi})| = %s at phi=%s   (sqrt(eps ghat)=%s; scale eps=%s)" % (
                a, g, ns(best[0], 6), ns(best[1], 6), ns(mp.sqrt(eps * gg), 6), ns(eps, 3)))

    print()
    print("=== zeros strictly inside: delta<Re phi<2pi-delta, 0<Im phi<Y2 (principal sheet, cut z in (0,1) excluded) ===")
    for a in ['0.3', '0.5', '0.9', '0.99', '0.999', '0.9999']:
        for g in ['1', '0.5']:
            al = mp.mpf(a)
            for Y2 in [mp.mpf('0.5'), mp.mpf(3), mp.mpf(8)]:
                n, st = count_inside(al, mp.mpf(g), mp.mpf('1e-9'), Y2)
                print("  alpha=%-7s ghat=%-4s e^-%s<|z|<1: zeros = %d (raw %s, min|F| %s)" % (a, g, ns(Y2, 3), n, ns(st['raw'], 10), ns(st['minabs'], 3)))
    print()
    print("=== locate the inside roots for alpha not close to 1 (does the conjugate pair persist?) ===")
    mp.mp.dps = 40
    prev = paper_phi(mp.mpf('1e-3'), 1)
    for a in ['0.999', '0.99', '0.95', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0.05']:
        al = mp.mpf(a)
        try:
            r = mp.findroot(lambda p: F_phi(p, al, 1), (prev, prev * (1 + mp.mpf('1e-6'))), solver='secant', tol=mp.mpf(10) ** (-30), verify=False)
            res = abs(F_phi(r, al, 1))
            print("  alpha=%-6s ghat=1: phi = %s + %s i   |z|=%s  |F|=%s   Re phi/sqrt(eps)=%s  Im phi/(Re phi*eps)=%s" % (
                a, ns(r.real, 12), ns(r.imag, 12), ns(mp.exp(-r.imag), 10), ns(res, 3), ns(r.real / mp.sqrt(1 - al), 8), ns(r.imag / r.real / (1 - al), 8)))
            prev = r
        except Exception as ex:
            print("  alpha=%s: root tracking failed: %s" % (a, ex))
