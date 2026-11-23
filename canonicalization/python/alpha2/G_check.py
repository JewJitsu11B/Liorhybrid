"""Checks written for the alpha = 2 canonical sheet, labels G1--G4. Exact sympy; exits non-zero on failure.

G1  a boost of rapidity eta acts on the null coordinates by u -> e^{eta} u, v -> e^{-eta} v
    (or the reverse), so uv and du dv are invariant
G2  the pair matrix M_k(t) has determinant 1 and trace 2 cos(omega t): elliptic in SL(2,R)
G3  the two independent forms of the next Lorentz-defect coefficient agree, and the
    phi^2 and phi^4 coefficients of h_alpha are as stated
G4  rotation of (s, c|k|) by an angle leaves R fixed and shifts phi: h_2 = 1 is the invariant case
"""
import sys
import sympy as sp

res = []


def check(label, text, ok):
    res.append(bool(ok))
    print(("PASS" if ok else "FAIL"), label, text)


c, t, x, eta = sp.symbols("c t x eta", real=True, positive=True)
ch, sh = sp.cosh(eta), sp.sinh(eta)
ct2 = ch * c * t - sh * x
x2 = ch * x - sh * c * t
u, v = c * t - x, c * t + x
u2, v2 = ct2 - x2, ct2 + x2
ok = sp.simplify((u2 / u).rewrite(sp.exp) - sp.exp(eta)) == 0 and sp.simplify((v2 / v).rewrite(sp.exp) - sp.exp(-eta)) == 0
ok &= sp.simplify((u2 * v2 - u * v).rewrite(sp.exp)) == 0
check("G1", "boost: u -> e^{eta} u, v -> e^{-eta} v; uv and du dv invariant", ok)

w = sp.symbols("omega", positive=True)
M = sp.Matrix([[sp.cos(w * t), sp.sin(w * t) / w], [-w * sp.sin(w * t), sp.cos(w * t)]])
ok = sp.simplify(M.det() - 1) == 0 and sp.simplify(M.trace() - 2 * sp.cos(w * t)) == 0
t2 = sp.symbols("t2", real=True)
ok &= sp.simplify(M.subs(t, t + t2) - M * M.subs(t, t2)) == sp.zeros(2, 2)
check("G2", "M_k(t): det 1, trace 2 cos(omega t), group law", ok)

al, ph = sp.symbols("alpha phi", real=True)
ok = sp.expand((4 + 6 * al - 3 * al ** 2) - (7 - 3 * (al - 1) ** 2)) == 0
h = sp.sin((al - 1) * ph) / ((al - 1) * sp.sin(ph))
ser = sp.series(h, ph, 0, 6).removeO()
ok &= sp.simplify(ser.coeff(ph, 2) - al * (2 - al) / 6) == 0
ok &= sp.simplify(ser.coeff(ph, 4) - al * (2 - al) * (4 + 6 * al - 3 * al ** 2) / 360) == 0
check("G3", "h_alpha = 1 + alpha(2-alpha) phi^2/6 + alpha(2-alpha)(4+6alpha-3alpha^2) phi^4/360", ok)

R, th = sp.symbols("R theta", positive=True)
s_, b_ = R * sp.cos(ph), R * sp.sin(ph)
s2 = sp.cos(th) * s_ - sp.sin(th) * b_
b2 = sp.sin(th) * s_ + sp.cos(th) * b_
ok = sp.simplify(s2 ** 2 + b2 ** 2 - R ** 2) == 0 and sp.simplify(s2 - R * sp.cos(ph + th)) == 0
ok &= sp.simplify(h.subs(al, 2) - 1) == 0
check("G4", "rotating (s, c|k|) keeps R and shifts phi; h_2 = 1", ok)

print("%d of %d passed" % (sum(res), len(res)))
sys.exit(0 if all(res) else 1)
