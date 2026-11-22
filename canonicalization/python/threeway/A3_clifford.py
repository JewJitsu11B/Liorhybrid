"""A3: Cl(3,0) as an 8-dim real algebra built from generator relations. No matrices
represent algebra elements. (sympy Matrix is used only for the 3x3 ad matrices of the
bivector Lie algebra, which is what tr(ad ad) requires.)"""
import itertools
import sympy as sp

# Basis blades: sorted tuples of generator indices drawn from (1,2,3).
BLADES = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]


def blade_mul(a, b):
    """Product of two blades using only s_j s_k = -s_k s_j (j != k) and s_j^2 = +1.
    Bubble-sort the concatenated word, counting swaps, then cancel adjacent equal pairs."""
    w = list(a) + list(b)
    sign = 1
    changed = True
    while changed:
        changed = False
        for i in range(len(w) - 1):
            if w[i] > w[i + 1]:
                w[i], w[i + 1] = w[i + 1], w[i]
                sign = -sign
                changed = True
            elif w[i] == w[i + 1]:
                del w[i:i + 2]          # s_j s_j = +1
                changed = True
                break
    return sign, tuple(w)


TABLE = {(a, b): blade_mul(a, b) for a in BLADES for b in BLADES}


class MV:
    def __init__(self, c=None):
        self.c = {b: sp.expand(v) for b, v in (c or {}).items() if sp.expand(v) != 0}

    def __add__(self, o):
        d = dict(self.c)
        for b, v in o.c.items():
            d[b] = d.get(b, 0) + v
        return MV(d)

    def __neg__(self):
        return MV({b: -v for b, v in self.c.items()})

    def __sub__(self, o):
        return self + (-o)

    def scale(self, k):
        return MV({b: k * v for b, v in self.c.items()})

    def __mul__(self, o):
        d = {}
        for a, va in self.c.items():
            for b, vb in o.c.items():
                s, r = TABLE[(a, b)]
                d[r] = d.get(r, 0) + s * va * vb
        return MV(d)

    def grade(self, k):
        return MV({b: v for b, v in self.c.items() if len(b) == k})

    def scalar(self):
        return self.c.get((), sp.Integer(0))

    def reverse(self):       # sign (-1)^{k(k-1)/2}
        return MV({b: (-1) ** (len(b) * (len(b) - 1) // 2) * v for b, v in self.c.items()})

    def involute(self):      # grade involution, sign (-1)^k
        return MV({b: (-1) ** len(b) * v for b, v in self.c.items()})

    def conj(self):          # Clifford conjugate = reversion composed with grade involution
        return self.reverse().involute()

    def is_zero(self):
        return not self.c

    def __repr__(self):
        return str({("s" + "".join(map(str, b)) if b else "1"): v for b, v in self.c.items()}) if self.c else "0"


def N(q):
    return q * q.conj()


def comm(a, b):
    return a * b - b * a


ONE = MV({(): 1})
s = {j: MV({(j,): 1}) for j in (1, 2, 3)}
I = s[1] * s[2] * s[3]

# --- sanity: generator relations, associativity, pseudoscalar ---
for j, k in itertools.product((1, 2, 3), repeat=2):
    assert (s[j] * s[k] + s[k] * s[j] - ONE.scale(2 if j == k else 0)).is_zero()
E = [MV({b: 1}) for b in BLADES]
for a, b, c in itertools.product(E, repeat=3):
    assert ((a * b) * c - a * (b * c)).is_zero()
assert (I * I + ONE).is_zero()
for j in (1, 2, 3):
    assert comm(I, s[j]).is_zero()
print("generator relations, associativity (512 triples), I^2=-1, I central: OK")

# --- anti-automorphism checks for reversion and conjugation ---
for a, b in itertools.product(E, repeat=2):
    assert ((a * b).reverse() - b.reverse() * a.reverse()).is_zero()
    assert ((a * b).conj() - b.conj() * a.conj()).is_zero()
print("reversion and Clifford conjugation are anti-automorphisms: OK")

# --- bivector Lie algebra and Killing form from tr(ad ad) ---
BIV = [I * s[1], I * s[2], I * s[3]]          # = s23, -s13, s12
print("bivector basis B_j = I s_j:", BIV)


def coords(q):
    """Coordinates of a pure bivector in BIV, by solving a linear system."""
    assert (q - q.grade(2)).is_zero(), "commutator left the bivector space"
    u = sp.symbols("u1:4")
    diff = q - (BIV[0].scale(u[0]) + BIV[1].scale(u[1]) + BIV[2].scale(u[2]))
    sol = sp.solve(list(diff.c.values()), u, dict=True) if diff.c else [{}]
    return [sol[0].get(ui, sp.Integer(0)) if diff.c else sp.Integer(0) for ui in u]


def ad(a):
    return sp.Matrix([coords(comm(a, Bk)) for Bk in BIV]).T   # column k = [a, B_k]


def kappa(a, b):
    return sp.expand((ad(a) * ad(b)).trace())


for j in range(3):
    for k in range(3):
        print("[B%d,B%d] =" % (j + 1, k + 1), comm(BIV[j], BIV[k]), end=";  ")
    print()
G = sp.Matrix(3, 3, lambda j, k: kappa(BIV[j], BIV[k]))
print("Killing Gram matrix on B_j:", G.tolist())
print("B_j^2:", [BIV[j] * BIV[j] for j in range(3)])
# Jacobi
for a, b, c in itertools.product(BIV, repeat=3):
    assert (comm(a, comm(b, c)) + comm(b, comm(c, a)) + comm(c, comm(a, b))).is_zero()
print("Jacobi on bivectors: OK")

# --- part 4 ---
t, x1, x2, x3 = sp.symbols("t x1 x2 x3", real=True)
x = s[1].scale(x1) + s[2].scale(x2) + s[3].scale(x3)
X = ONE.scale(t) + x
NX = N(X)
print("N(X) =", NX)
assert (NX - NX.grade(0)).is_zero()
print("N(X) pure scalar on paravectors: OK")
Ix = I * x
assert (Ix - Ix.grade(2)).is_zero()
kap = kappa(Ix, Ix)
print("kappa(Ix,Ix) =", kap)
resid = sp.expand(NX.scalar() - (X.scalar() ** 2 + sp.Rational(1, 8) * kap))
print("residual N(X) - (<X>_0^2 + kappa/8) =", resid)
assert resid == 0
assert sp.expand(kap + 8 * (x1**2 + x2**2 + x3**2)) == 0
assert (N(X) - X.conj() * X).is_zero()
print("PART 4 VERIFIED; also qbar q = q qbar on paravectors")

# general unit bivector: kappa(B,B) with B = n.B, and shortcut kappa = 8 <AB>_0 verified
b1, b2, b3, c1, c2, c3 = sp.symbols("b1:4 c1:4", real=True)
A = BIV[0].scale(b1) + BIV[1].scale(b2) + BIV[2].scale(b3)
C = BIV[0].scale(c1) + BIV[1].scale(c2) + BIV[2].scale(c3)
print("kappa(A,A) general bivector =", sp.factor(kappa(A, A)))
print("kappa(A,C) - 8<AC>_0 =", sp.expand(kappa(A, C) - 8 * (A * C).scalar()))

# --- scope: outside paravectors ---
q = sp.symbols("q0:8", real=True)
Q = MV({b: q[i] for i, b in enumerate(BLADES)})
NQ = N(Q)
print("general element: N(Q) grade-0 =", NQ.grade(0).scalar())
print("general element: N(Q) grade-3 coeff =", NQ.c.get((1, 2, 3), 0))
print("general element: N(Q) grade-1,2 parts zero:", NQ.grade(1).is_zero() and NQ.grade(2).is_zero())
# with reversion instead of Clifford conjugation the identity fails on paravectors
R = X * X.reverse()
print("X * reverse(X) =", R, " (not scalar; reversion is the wrong conjugation)")
# mutated constant
print("residual with 1/4 instead of 1/8:", sp.expand(NX.scalar() - (t**2 + sp.Rational(1, 4) * kap)))
