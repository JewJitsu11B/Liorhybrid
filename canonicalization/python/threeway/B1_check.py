# B1: high-precision random testing of
#   det X = (tr X)^2/4 + (1/8) kappa(i X0, i X0),  X0 = X - (tr X/2) 1
# kappa(A,B) = tr(ad_A ad_B), ad built from commutators in the basis E_j = i s_j.
import random
from mpmath import mp, mpf, mpc

mp.dps = 50
random.seed(20261121)
N = 10000
I = mpc(0, 1)

def mul(A, B):
    return ((A[0]*B[0] + A[1]*B[2], A[0]*B[1] + A[1]*B[3]),
            (A[2]*B[0] + A[3]*B[2], A[2]*B[1] + A[3]*B[3]))

def M(a, b, c, d):
    return (a, b, c, d)

def mmul(A, B):
    return (A[0]*B[0] + A[1]*B[2], A[0]*B[1] + A[1]*B[3],
            A[2]*B[0] + A[3]*B[2], A[2]*B[1] + A[3]*B[3])

def sub(A, B):
    return tuple(a - b for a, b in zip(A, B))

def comm(A, B):
    return sub(mmul(A, B), mmul(B, A))

def dag(A):
    return (A[0].conjugate(), A[2].conjugate(), A[1].conjugate(), A[3].conjugate())

def tr(A):
    return A[0] + A[3]

def det(A):
    return A[0]*A[3] - A[1]*A[2]

ONE = mpc(1); ZERO = mpc(0)
SX = (ZERO, ONE, ONE, ZERO)
SY = (ZERO, -I, I, ZERO)
SZ = (ONE, ZERO, ZERO, -ONE)
E = [tuple(I*z for z in S) for S in (SX, SY, SZ)]   # basis i sx, i sy, i sz

def coords(Mx):
    # Mx traceless = m1 E1 + m2 E2 + m3 E3; returns (m1,m2,m3) and trace defect
    a, b, c, d = Mx
    return ((b + c)/(2*I), (b - c)/2, a/I), abs(a + d)

def ad(A):
    # 3x3 matrix, column k = coordinates of [A, E_k]
    cols = [coords(comm(A, Ek))[0] for Ek in E]
    return [[cols[k][j] for k in range(3)] for j in range(3)]

def kappa(A, B):
    P, Q = ad(A), ad(B)
    return sum(P[i][j]*Q[j][i] for i in range(3) for j in range(3))

def r():
    return mpf(random.uniform(-1, 1)) + mpf(random.uniform(-1, 1))*mpf(2)**(-53) \
        + mp.rand()*mpf(10)**(-20)

def rc():
    return mpc(r(), r())

def herm(t, x):
    return (mpc(t + x[2]), mpc(x[0], -x[1]), mpc(x[0], x[1]), mpc(t - x[2]))

def terms(X, form="bilinear", c=mpf(1)/8):
    t2 = tr(X)/2
    X0 = (X[0] - t2, X[1], X[2], X[3] - t2)
    A = tuple(I*z for z in X0)
    if form == "bilinear":
        k = kappa(A, A)
    else:  # sesquilinear variant: tr(ad_A ad_A^conj-coords)
        P = ad(A)
        k = sum(P[i][j]*P[j][i].conjugate() for i in range(3) for j in range(3))
    d = det(X)
    rhs = t2*t2 + c*k
    scale = max(abs(d), abs(t2*t2), abs(c*k))
    aimag = max(abs(m.imag) for m in coords(A)[0])
    return d, rhs, scale, aimag, k

def run(name, gen, form="bilinear", c=mpf(1)/8, n=N):
    maxabs = mpf(0); maxrel = mpf(0); maxim = mpf(0); info = {}
    for _ in range(n):
        X, tag = gen()
        info[tag] = info.get(tag, 0) + 1
        d, rhs, scale, aimag, k = terms(X, form, c)
        res = abs(d - rhs)
        maxabs = max(maxabs, res)
        if scale > 0:
            maxrel = max(maxrel, res/scale)
            maxim = max(maxim, aimag/mp.sqrt(scale))
        else:
            maxim = max(maxim, aimag)
    print("%-34s n=%d  max|res|=%s  max rel=%s  max rel Im(coords of iX0)=%s"
          % (name, n, mp.nstr(maxabs, 3), mp.nstr(maxrel, 3), mp.nstr(maxim, 3)))
    print("      subclasses:", info)
    return maxrel

# sanity: Killing form on the basis
print("dps =", mp.dps)
print("kappa(E_j,E_k), E_j = i s_j:")
for j in range(3):
    print("   ", [mp.nstr(kappa(E[j], E[k]), 5) for k in range(3)])

def gen_a():
    J = (rc(), rc(), rc(), rc())
    return mmul(J, dag(J)), "JJdag"

def gen_b():
    t = r(); x = (r(), r(), r())
    X = herm(t, x)
    dd = det(X).real
    tag = "indefinite" if dd < 0 else ("posdef" if t > 0 else "negdef")
    return X, tag

cnt_c = [0]
def gen_c():
    cnt_c[0] += 1
    m = cnt_c[0] % 5
    if m == 0:
        return (ZERO, ZERO, ZERO, ZERO), "zero"
    v = (rc(), rc())
    X = (v[0]*v[0].conjugate(), v[0]*v[1].conjugate(),
         v[1]*v[0].conjugate(), v[1]*v[1].conjugate())
    if m == 1:
        return tuple(-z for z in X), "neg rank1"
    if m == 2:
        J = (v[0], v[0]*rc(), v[1], ZERO)
        J = (J[0], J[1], J[2], J[1]*J[2]/J[0])   # singular J
        return mmul(J, dag(J)), "JJdag singular J"
    return X, "rank1 vv^dag"

def gen_d():
    e = random.randint(-300, 300)
    s = mpf(10)**e
    t = r(); x = (r(), r(), r())
    which = random.randint(0, 2)
    if which == 0:
        return tuple(s*z for z in herm(t, x)), "uniform scale 1e-300..1e300"
    if which == 1:   # tiny traceless part on big trace
        return herm(t, tuple(s*xi for xi in x)), "scaled traceless part only"
    return herm(s*t, x), "scaled trace only"

def gen_e():
    return (rc(), rc(), rc(), rc()), "generic complex"

print("\nPart 1, bilinear kappa = tr(ad ad), constant 1/8:")
run("(a) X = J J^dagger", gen_a)
run("(b) random Hermitian", gen_b)
run("(c) rank-one and zero", gen_c)
run("(d) extreme scales (Hermitian)", gen_d)
run("(e) non-Hermitian complex, sl(2,C)", gen_e)

print("\nControls (should FAIL, i.e. residual O(1)):")
run("(b) Hermitian, constant 1/4", gen_b, c=mpf(1)/4, n=1000)
run("(b) Hermitian, constant -1/8", gen_b, c=-mpf(1)/8, n=1000)
run("(e) non-Herm, sesquilinear kappa", gen_e, form="sesqui", n=1000)
run("(b) Hermitian, sesquilinear kappa", gen_b, form="sesqui", n=1000)

def gen_e_hpart():
    X = (rc(), rc(), rc(), rc())
    return X, "generic complex"
# non-Hermitian X, but kappa evaluated on i*(Hermitian part of X0): does det X still match?
maxrel = mpf(0)
for _ in range(1000):
    X = (rc(), rc(), rc(), rc())
    H = tuple((a + b)/2 for a, b in zip(X, dag(X)))
    t2 = tr(X)/2
    tH = tr(H)/2
    A = tuple(I*z for z in (H[0] - tH, H[1], H[2], H[3] - tH))
    rhs = t2*t2 + kappa(A, A)/8
    maxrel = max(maxrel, abs(det(X) - rhs)/max(abs(det(X)), abs(t2*t2)))
print("(e) non-Herm, kappa on Hermitian part only: max rel res = %s (n=1000)" % mp.nstr(maxrel, 3))
