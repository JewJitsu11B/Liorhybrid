"""Naturality checks N1--N3 for the closure c(J) = J J^dagger.

Source: M_2(C) as a real 8-dimensional space, SL(2,C) acting by J -> G J.
Target: Herm(2), SL(2,C) acting by X -> G X G^dagger.
Equivariance is tested infinitesimally on the six real generators of sl(2,C)
(the group is connected), by matching polynomial coefficients exactly.

N1  real-linear equivariant maps M_2(C) -> Herm(2): dimension 0
N2  real-quadratic equivariant maps: dimension 4 (J -> J H J^dagger, H Hermitian)
N3  those also invariant under right U(2), J -> J U: dimension 1 (H = 1, the closure)
"""
import itertools
import sys

import sympy as sp

I = sp.I
s0 = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -I], [I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])
HERM = [s0, sx, sy, sz]
SL = [sx, sy, sz, I * sx, I * sy, I * sz]
U2 = [I * s0, I * sx, I * sy, I * sz]

x = sp.symbols("x0:8", real=True)
J = sp.Matrix([[x[0] + I * x[1], x[2] + I * x[3]], [x[4] + I * x[5], x[6] + I * x[7]]])


def coords(M):
    out = []
    for e in M:
        e = sp.expand(e)
        out += [sp.re(e), sp.im(e)]
    return out


def general(degree):
    monos = [sp.Mul(*c) for c in itertools.combinations_with_replacement(x, degree)]
    cs, Q = [], sp.zeros(2, 2)
    for k, H in enumerate(HERM):
        for j, m in enumerate(monos):
            c = sp.Symbol("c_%d_%d" % (k, j), real=True)
            cs.append(c)
            Q += c * m * H
    return Q, cs


def derivative(Q, V):
    v = coords(V)
    return sum((sp.diff(Q, x[i]) * v[i] for i in range(8)), sp.zeros(2, 2))


def equations(E):
    eqs = []
    for H in HERM:
        comp = sp.expand((E * H).trace() / 2)
        for part in (sp.re(comp), sp.im(comp)):
            part = sp.expand(part)
            if part != 0:
                eqs += [c for c in sp.Poly(part, *x).coeffs()]
    return eqs


def dimension(Q, cs, right_invariant):
    eqs = []
    for A in SL:
        eqs += equations(derivative(Q, A * J) - A * Q - Q * A.H)
    if right_invariant:
        for B in U2:
            eqs += equations(derivative(Q, J * B))
    M = sp.Matrix([[sp.expand(e).coeff(c) for c in cs] for e in eqs])
    return len(cs) - M.rank()


results = []
Q1, c1 = general(1)
results.append(("N1 linear equivariant maps: dimension 0", dimension(Q1, c1, False) == 0))
Q2, c2 = general(2)
results.append(("N2 quadratic equivariant maps: dimension 4", dimension(Q2, c2, False) == 4))
results.append(("N3 quadratic, right U(2) invariant: dimension 1", dimension(Q2, c2, True) == 1))

# the surviving map in N3 is the closure itself
closure = J * J.H
ok = all(sp.simplify(derivative(closure, A * J) - A * closure - closure * A.H) == sp.zeros(2, 2) for A in SL)
ok = ok and all(sp.simplify(derivative(closure, J * B)) == sp.zeros(2, 2) for B in U2)
results.append(("N3 the closure J J^dagger is that map", ok))

bad = 0
for name, good in results:
    print("PASS" if good else "FAIL", name)
    bad += not good
print("%d of %d passed" % (len(results) - bad, len(results)))
sys.exit(1 if bad else 0)
