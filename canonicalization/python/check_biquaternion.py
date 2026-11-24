"""Biquaternions, M2(C) and Cl(3,0) are one algebra. Labels Q1--Q9. Exact sympy; exits non-zero on failure.

Three algebras are built independently of one another:
  B   = quaternions with complex coefficients, from Hamilton's rules i^2 = j^2 = k^2 = ijk = -1
  M2  = complex 2x2 matrices
  Cl3 = Cl(3,0) on blades, from s_j s_k = -s_k s_j, s_j^2 = +1 (no matrices)
and the maps between them are checked to be algebra isomorphisms that carry each
involution, the norm, the scalar part, the units and the grades to their partners.

Q1  B -> M2 (1, i, j, k -> 1, e1, e2, e3 with e_j = -i sigma_j) is multiplicative and is the carrier
Q2  B -> M2 is a bijection (complex 4x4, determinant non-zero)
Q3  quaternion conjugate -> adjugate; coefficient conjugate -> (adjugate)^dagger; both -> dagger
Q4  quaternion norm = det; scalar coefficient = (1/2) trace
Q5  B -> Cl3 is multiplicative and a bijection (real 8x8, determinant non-zero)
Q6  i, j, k -> the unit bivectors s3 s2, s1 s3, s2 s1; the complex unit -> the pseudoscalar s1 s2 s3
Q7  grades: Re Z0 grade 0, Im Z_j grade 1, Re Z_j grade 2, Im Z0 grade 3
Q8  quaternion conjugate -> Clifford conjugate; coefficient conjugate -> grade involution; both -> reversion
Q9  C (x)_R H -> B, z (x) h -> z h, is multiplicative on the basis and a bijection (real 8x8)
"""
import itertools
import sys
import sympy as sp

I = sp.I
RES = []


def check(label, ok, note=""):
    RES.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + label + ("  " + note if note else ""))


def zero(x):
    return sp.simplify(sp.expand(x)) == 0


# ------------------------------------------------------------------ B: Hamilton's rules, complex coefficients
# a biquaternion is a 4-tuple (w, x, y, z) of complex numbers
def bmul(p, q):
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    return (a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2)


def bstar(p):          # quaternion conjugate
    return (p[0], -p[1], -p[2], -p[3])


def bcc(p):            # complex conjugate of the coefficients
    return tuple(sp.conjugate(t) for t in p)


ONE, BI, BJ, BK = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
assert bmul(BI, BI) == (-1, 0, 0, 0) and bmul(BI, BJ) == BK and bmul(bmul(BI, BJ), BK) == (-1, 0, 0, 0)

# ------------------------------------------------------------------ M2
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -I], [I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])
E = [sp.eye(2), -I*sx, -I*sy, -I*sz]


def toM(p):
    return sum((p[a]*E[a] for a in range(4)), sp.zeros(2))


# ------------------------------------------------------------------ Cl3 on blades
BL = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]


def blade_mul(a, b):
    w, s = list(a + b), 1
    changed = True
    while changed:
        changed = False
        for n in range(len(w) - 1):
            if w[n] > w[n+1]:
                w[n], w[n+1] = w[n+1], w[n]; s = -s; changed = True
            elif w[n] == w[n+1]:
                del w[n:n+2]; changed = True; break
    return tuple(w), s


def cmul(u, v):
    out = {b: 0 for b in BL}
    for a, x in u.items():
        for b, y in v.items():
            bl, s = blade_mul(a, b)
            out[bl] += s*x*y
    return {b: sp.expand(t) for b, t in out.items()}


def cvec(d):
    return {b: sp.sympify(d.get(b, 0)) for b in BL}


def ceq(u, v):
    return all(zero(u[b] - v[b]) for b in BL)


def crev(u):   return {b: (-1)**(len(b)*(len(b)-1)//2) * t for b, t in u.items()}
def cinv(u):   return {b: (-1)**len(b) * t for b, t in u.items()}
def cconj(u):  return crev(cinv(u))


# B -> Cl3: Z_a = a_a + i b_a ;  i, j, k -> s3 s2, s1 s3, s2 s1 = -(s2 s3), +(s1 s3), -(s1 s2) ;  i -> s1 s2 s3
def toC(p):
    a = [sp.re(t) for t in p]
    b = [sp.im(t) for t in p]
    return cvec({(): a[0], (1, 2, 3): b[0],
                 (2, 3): -a[1], (1, 3): a[2], (1, 2): -a[3],
                 (1,): b[1], (2,): b[2], (3,): b[3]})


# ------------------------------------------------------------------ symbols
ar = sp.symbols("a0:4", real=True); br = sp.symbols("b0:4", real=True)
cr = sp.symbols("c0:4", real=True); dr = sp.symbols("d0:4", real=True)
P = tuple(ar[n] + I*br[n] for n in range(4))
Q = tuple(cr[n] + I*dr[n] for n in range(4))
Zs = sp.symbols("Z0:4")
Ws = sp.symbols("W0:4")

# Q1
check("Q1 B -> M2 is multiplicative (general complex coefficients)",
      all(zero(t) for t in (toM(bmul(Zs, Ws)) - toM(Zs)*toM(Ws))))
carrier = sp.Matrix([[Zs[0] - I*Zs[3], -I*Zs[1] - Zs[2]], [-I*Zs[1] + Zs[2], Zs[0] + I*Zs[3]]])
check("Q1 it is the carrier J = sum Z_a e_a", all(zero(t) for t in (toM(Zs) - carrier)))

# Q2
A = sp.Matrix([[sp.diff(toM(Zs)[r, c], Zs[n]) for n in range(4)] for r in range(2) for c in range(2)])
check("Q2 bijection: complex 4x4 matrix of the map has determinant %s" % A.det(), A.det() != 0)

# Q3
check("Q3 quaternion conjugate -> adjugate", all(zero(t) for t in (toM(bstar(Zs)) - toM(Zs).adjugate())))
check("Q3 coefficient conjugate -> (adjugate)^dagger", all(zero(t) for t in (toM(bcc(P)) - toM(P).adjugate().H)))
check("Q3 both -> dagger", all(zero(t) for t in (toM(bstar(bcc(P))) - toM(P).H)))

# Q4
check("Q4 quaternion norm = det", zero(bmul(Zs, bstar(Zs))[0] - toM(Zs).det())
      and all(zero(t) for t in bmul(Zs, bstar(Zs))[1:]))
check("Q4 scalar coefficient = (1/2) trace", zero(Zs[0] - toM(Zs).trace()/2))

# Q5
check("Q5 B -> Cl3 is multiplicative (16 real parameters)", ceq(toC(bmul(P, Q)), cmul(toC(P), toC(Q))))
Mreal = sp.Matrix([[sp.diff(toC(P)[b], v) for v in list(ar) + list(br)] for b in BL])
check("Q5 bijection: real 8x8 matrix of the map has determinant %s" % Mreal.det(), Mreal.det() != 0)

# Q6
s1, s2, s3 = cvec({(1,): 1}), cvec({(2,): 1}), cvec({(3,): 1})
check("Q6 i, j, k -> s3 s2, s1 s3, s2 s1", ceq(toC(BI), cmul(s3, s2)) and ceq(toC(BJ), cmul(s1, s3)) and ceq(toC(BK), cmul(s2, s1)))
check("Q6 complex unit -> pseudoscalar s1 s2 s3, central, square -1",
      ceq(toC((I, 0, 0, 0)), cmul(cmul(s1, s2), s3))
      and ceq(cmul(toC((I, 0, 0, 0)), toC(P)), cmul(toC(P), toC((I, 0, 0, 0))))
      and ceq(cmul(toC((I, 0, 0, 0)), toC((I, 0, 0, 0))), cvec({(): -1})))

# Q7
c = toC(P)
g = lambda k: {v for b in BL if len(b) == k for v in c[b].free_symbols}
check("Q7 grades: 0 <- Re Z0, 1 <- Im Z_j, 2 <- Re Z_j, 3 <- Im Z0",
      g(0) == {ar[0]} and g(1) == set(br[1:]) and g(2) == set(ar[1:]) and g(3) == {br[0]})

# Q8
check("Q8 quaternion conjugate -> Clifford conjugate", ceq(toC(bstar(P)), cconj(toC(P))))
check("Q8 coefficient conjugate -> grade involution", ceq(toC(bcc(P)), cinv(toC(P))))
check("Q8 both -> reversion", ceq(toC(bstar(bcc(P))), crev(toC(P))))

# Q9  C (x)_R H : basis (z, u), z in {1, i}, u in {1, i, j, k}; product (z u)(w v) = (z w) (x) (u v)
HB = [ONE, BI, BJ, BK]
CB = [1, I]


def tens(z, u):                       # image of z (x) u in B
    return tuple(sp.sympify(z)*t for t in u)


ok = True
for z, w in itertools.product(CB, CB):
    for u, v in itertools.product(HB, HB):
        lhs = tens(z*w, bmul(u, v))             # image of the product in the tensor algebra
        rhs = bmul(tens(z, u), tens(w, v))      # product of the images
        ok = ok and all(zero(lhs[n] - rhs[n]) for n in range(4))
check("Q9 C (x) H -> B is multiplicative on the 64 pairs of basis elements", ok)
T = sp.Matrix([[f(tens(z, u)[n]) for z in CB for u in HB] for n in range(4) for f in (sp.re, sp.im)])
check("Q9 bijection: real 8x8 matrix has determinant %s" % T.det(), T.det() != 0)

print("%d of %d passed" % (sum(RES), len(RES)))
sys.exit(0 if all(RES) else 1)
