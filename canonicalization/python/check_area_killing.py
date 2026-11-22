"""Area checks W1--W4: the area element of a 2-plane is the square root of a Killing norm.

Vectors u, v of Minkowski space are the Hermitian matrices U = u0 + u.sigma, V = v0 + v.sigma,
with eta(u, v) = (1/2) tr(U adj V) and det U = eta(u, u).  The plane u ^ v is the element
A = (1/4)(U adj V - V adj U) of sl(2,C).  h is the Gram matrix of (u, v).

W1  A is traceless and det A = (1/4) det h (so det A is real on planes)
W2  real Killing norm: 8 Re tr(A^2) = -4 det h, so -det h = kappa(A, A)/4 = -4 det A
W3  the map is equivariant: dU = BU + UB^dagger, dV likewise, gives dA = [B, A]
W4  a timelike plane is boost-like (A Hermitian, -det h > 0); a spacelike plane is
    rotation-like (A anti-Hermitian, -det h < 0)
"""
import sys

import sympy as sp

I = sp.I
s0 = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -I], [I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])
SIG = [s0, sx, sy, sz]
SL = [sx, sy, sz, I * sx, I * sy, I * sz]

u = sp.symbols("u0:4", real=True)
v = sp.symbols("v0:4", real=True)


def herm(w):
    return sum((w[k] * SIG[k] for k in range(4)), sp.zeros(2, 2))


def adj(M):
    return M.trace() * s0 - M


def eta(P, Q):
    return sp.expand((P * adj(Q)).trace() / 2)


def plane(P, Q):
    return (P * adj(Q) - Q * adj(P)) / 4


def zero(M):
    return sp.simplify(sp.expand(M)) == sp.zeros(*M.shape)


U, V = herm(u), herm(v)
A = plane(U, V)
gram = sp.expand(eta(U, U) * eta(V, V) - eta(U, V) ** 2)

results = []
results.append(("W1 A traceless, det A = det h / 4",
                sp.expand(A.trace()) == 0 and sp.expand(A.det() - gram / 4) == 0))
results.append(("W2 8 Re tr(A^2) = -4 det h",
                sp.expand(8 * sp.re(sp.expand((A * A).trace())) + 4 * gram) == 0))

ok = True
for B in SL:
    dU, dV = B * U + U * B.H, B * V + V * B.H
    dA = plane(dU, V) + plane(U, dV)
    ok = ok and zero(dA - (B * A - A * B))
results.append(("W3 equivariance dA = [B, A]", ok))

e = [herm([1 if k == j else 0 for k in range(4)]) for j in range(4)]
At, As = plane(e[0], e[1]), plane(e[1], e[2])
gt = eta(e[0], e[0]) * eta(e[1], e[1]) - eta(e[0], e[1]) ** 2
gs = eta(e[1], e[1]) * eta(e[2], e[2]) - eta(e[1], e[2]) ** 2
results.append(("W4 timelike plane boost-like, spacelike plane rotation-like",
                zero(At - At.H) and -gt > 0 and zero(As + As.H) and -gs < 0))

bad = 0
for name, good in results:
    print("PASS" if good else "FAIL", name)
    bad += not good
print("%d of %d passed" % (len(results) - bad, len(results)))
sys.exit(1 if bad else 0)
