#!/usr/bin/env python3
"""
Containers over a finite data space: gluing, identifiability, max entropy.

Question: we have a space of data and a set of containers over it. How do we
model the data given the containers?

Setup. Omega is a finite set of outcomes. A container is an event C in Omega
whose mass P(C) is known. A marginal table on a block of variables is a bundle
of containers, one cylinder event per cell. Stack the rows [1; 1_C1; ...; 1_CK]
into an integer matrix M and the masses into y, with y_0 = 1.

  Gluing   The local data glue iff some p >= 0 has M p = y. If none does,
           Farkas' lemma gives w with M^T w >= 0 and y.w < 0: an inequality
           that every global distribution satisfies and these data violate.
  Fiber    The models consistent with the data are {p >= 0 : M p = y}. Two of
           them are metamers: they differ by a vector in ker M.
  d_unk    d_unk(p) = dim(ker M  intersect  L_p), L_p = {d : d_x = 0 where
           p_x = 0}. This is eq. (15) of "The Identifiability of Color" with the
           M_i taken to be the container rows and S_phys the probability simplex.
  Max ent  p*(x) is proportional to exp(theta . phi(x)) on the support of the
           fiber, where phi(x) = (1[x in C_1], ..., 1[x in C_K]) are the
           container coordinates of x. So p* is a softmax over container
           coordinates with logits theta (the Lagrange multipliers). Its
           optimality condition is log p* orthogonal to ker M: entropy only
           chooses inside the kernel.

Gluing, supports, kernels and d_unk are computed in exact rational arithmetic
(fractions.Fraction, exact simplex with Bland's rule). Only the max-entropy
solve (damped Newton on the convex dual) uses floats. No third-party packages.

Run:  python examples/containers_maxent.py
"""

import math
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations, product


# ---------------------------------------------------------------------------
# Exact linear algebra
# ---------------------------------------------------------------------------

def _rref(rows):
    A = [[Fraction(v) for v in r] for r in rows]
    ncols = len(A[0]) if A else 0
    pivots, r = [], 0
    for c in range(ncols):
        piv = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        lead = A[r][c]
        A[r] = [v / lead for v in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        pivots.append(c)
        r += 1
        if r == len(A):
            break
    return A[:r], pivots


def rank(rows):
    return len(_rref(rows)[1]) if rows else 0


def kernel(rows, ncols):
    """Integer basis of {d : rows . d = 0}, first nonzero entry positive."""
    R, pivots = _rref(rows)
    basis = []
    for f in (c for c in range(ncols) if c not in pivots):
        v = [Fraction(0)] * ncols
        v[f] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -R[i][f]
        scale = math.lcm(*(x.denominator for x in v))
        ints = [int(x * scale) for x in v]
        g = math.gcd(*ints)
        ints = [x // g for x in ints]
        if next(x for x in ints if x != 0) < 0:
            ints = [-x for x in ints]
        basis.append(ints)
    return basis


# ---------------------------------------------------------------------------
# Exact simplex: feasibility, Farkas certificates, support of the fiber
# ---------------------------------------------------------------------------

class _Tableau:
    """Tableau simplex in exact arithmetic for {A x = b, x >= 0}, Bland's rule."""

    def __init__(self, A, b):
        self.m, self.n = len(A), len(A[0])
        self.sign = [-1 if bi < 0 else 1 for bi in b]
        self.T = []
        for i in range(self.m):
            s = self.sign[i]
            art = [Fraction(int(i == k)) for k in range(self.m)]
            self.T.append([s * Fraction(a) for a in A[i]] + art + [s * Fraction(b[i])])
        self.basis = [self.n + i for i in range(self.m)]

    def _pivot(self, r, c):
        lead = self.T[r][c]
        self.T[r] = [v / lead for v in self.T[r]]
        for i in range(len(self.T)):
            if i != r and self.T[i][c] != 0:
                f = self.T[i][c]
                self.T[i] = [a - f * b for a, b in zip(self.T[i], self.T[r])]
        self.basis[r] = c

    def _minimize(self, cost, allowed):
        while True:
            cB = [cost[j] for j in self.basis]
            enter = None
            for j in allowed:
                if j in self.basis:
                    continue
                if cost[j] - sum(cb * row[j] for cb, row in zip(cB, self.T)) < 0:
                    enter = j
                    break
            if enter is None:
                return sum(cb * row[-1] for cb, row in zip(cB, self.T))
            best = None
            for i, row in enumerate(self.T):
                if row[enter] > 0:
                    key = (row[-1] / row[enter], self.basis[i])
                    if best is None or key < best[0]:
                        best = (key, i)
            if best is None:
                raise ArithmeticError("unbounded LP (cannot happen on a simplex)")
            self._pivot(best[1], enter)

    def phase_one(self):
        """None if feasible (tableau left at a feasible basis), else Farkas w."""
        cost = [Fraction(0)] * self.n + [Fraction(1)] * self.m
        if self._minimize(cost, range(self.n + self.m)) > 0:
            cB = [cost[j] for j in self.basis]
            u = [sum(cb * row[self.n + i] for cb, row in zip(cB, self.T))
                 for i in range(self.m)]
            return [-s * ui for s, ui in zip(self.sign, u)]
        i = 0
        while i < len(self.T):  # drive zero-level artificials out of the basis
            if self.basis[i] >= self.n:
                j = next((j for j in range(self.n) if self.T[i][j] != 0), None)
                if j is None:  # redundant row
                    del self.T[i]
                    del self.basis[i]
                    continue
                self._pivot(i, j)
            i += 1
        return None

    def maximize(self, j):
        cost = [Fraction(0)] * (self.n + self.m)
        cost[j] = Fraction(-1)
        return -self._minimize(cost, range(self.n))

    def solution(self):
        x = [Fraction(0)] * self.n
        for i, bj in enumerate(self.basis):
            if bj < self.n:
                x[bj] = self.T[i][-1]
        return x


def fiber_support(M, y):
    """(support, None) if the data glue; (None, w) with a Farkas certificate if not.

    The support is the set of outcomes some consistent model gives positive
    mass. It is the face of the simplex that L_p is computed on.
    """
    tab = _Tableau(M, y)
    w = tab.phase_one()
    if w is not None:
        assert all(sum(w[k] * M[k][x] for k in range(len(M))) >= 0 for x in range(len(M[0])))
        assert sum(wk * yk for wk, yk in zip(w, y)) < 0
        return None, w
    support = set()

    def mark():
        support.update(j for j, v in enumerate(tab.solution()) if v > 0)

    mark()
    for j in range(tab.n):
        if j not in support and tab.maximize(j) > 0:
            mark()
    return sorted(support), None


# ---------------------------------------------------------------------------
# Max entropy on the support (floats)
# ---------------------------------------------------------------------------

def _solve(A, b):
    n = len(b)
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    for c in range(n):
        piv = max(range(c, n), key=lambda i: abs(M[i][c]))
        M[c], M[piv] = M[piv], M[c]
        for i in range(c + 1, n):
            f = M[i][c] / M[c][c]
            for j in range(c, n + 1):
                M[i][j] -= f * M[c][j]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def max_entropy(M, y, support):
    """Max-entropy model on the support: p* = softmax(theta . phi) there.

    Uses a maximal set of container rows that are linearly independent on the
    support (together with the normalization row), so the dual Hessian
    Cov_p(phi) is positive definite and Newton's method is well posed.
    """
    chosen = [0]
    for k in range(1, len(M)):
        if rank([[M[i][x] for x in support] for i in chosen + [k]]) == len(chosen) + 1:
            chosen.append(k)
    rows = chosen[1:]
    feats = [[float(M[k][x]) for x in support] for k in rows]
    target = [float(y[k]) for k in rows]
    r, s = len(rows), len(support)

    def dual(th):
        z = [sum(th[a] * feats[a][i] for a in range(r)) for i in range(s)]
        zmax = max(z)
        logZ = zmax + math.log(sum(math.exp(v - zmax) for v in z))
        return logZ - sum(t * g for t, g in zip(th, target)), [math.exp(v - logZ) for v in z]

    theta = [0.0] * r
    f, p = dual(theta)
    for _ in range(200):
        mean = [sum(feats[a][i] * p[i] for i in range(s)) for a in range(r)]
        grad = [mean[a] - target[a] for a in range(r)]
        if max(map(abs, grad), default=0.0) < 1e-15:
            break
        H = [[sum(feats[a][i] * feats[b][i] * p[i] for i in range(s)) - mean[a] * mean[b]
              for b in range(r)] for a in range(r)]
        step = _solve(H, [-g for g in grad])
        slope = sum(g * d for g, d in zip(grad, step))
        t = 1.0
        while True:
            cand = [th + t * d for th, d in zip(theta, step)]
            f_new, p_new = dual(cand)
            if f_new <= f + 1e-4 * t * slope or t < 1e-12:
                break
            t /= 2
        if t < 1e-12:
            break
        theta, f, p = cand, f_new, p_new
    full = [0.0] * len(M[0])
    for i, x in enumerate(support):
        full[x] = p[i]
    return full, dict(zip(rows, theta))


# ---------------------------------------------------------------------------
# Building container systems
# ---------------------------------------------------------------------------

def build(points, containers):
    """containers: list of (label, members). Returns integer M (row 0 = total) and labels."""
    idx = {p: i for i, p in enumerate(points)}
    M, labels = [[1] * len(points)], ["total"]
    for label, members in containers:
        row = [0] * len(points)
        for m in members:
            row[idx[m]] = 1
        M.append(row)
        labels.append(label)
    return M, labels


VARS = "XYZ"
CUBE = ["".join(b) for b in product("01", repeat=3)]  # outcomes xyz


def cylinders(block, table):
    """Containers for a marginal table on `block` (e.g. 'XY'): one per cell."""
    pos = [VARS.index(v) for v in block]
    containers, masses = [], []
    for cell in ("".join(c) for c in product("01", repeat=len(block))):
        members = [w for w in CUBE if all(w[i] == cell[k] for k, i in enumerate(pos))]
        containers.append((f"{block}={cell}", members))
        masses.append(Fraction(table.get(cell, 0)))
    return containers, masses


def cover(blocks, tables):
    containers, masses = [], []
    for b in blocks:
        c, m = cylinders(b, tables[b])
        containers += c
        masses += m
    return containers, masses


def marginal(q, block):
    pos = [VARS.index(v) for v in block]
    t = defaultdict(Fraction)
    for w, v in q.items():
        t["".join(w[i] for i in pos)] += v
    return dict(t)


def is_acyclic(blocks):
    """GYO reduction. A cover is acyclic iff every family of marginals that agree
    on overlaps glues (Vorob'ev 1962; Beeri, Fagin, Maier, Yannakakis 1983), and
    iff the max-entropy fit has the closed junction-tree form."""
    edges = [set(b) for b in blocks]
    while True:
        count = Counter(v for e in edges for v in e)
        new = [{v for v in e if count[v] > 1} for e in edges]
        keep = [e for i, e in enumerate(new)
                if not any(j != i and (e < f or (e == f and j < i)) for j, f in enumerate(new))]
        if keep == edges:
            return len(edges) <= 1
        edges = keep


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(v):
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    return f"{v:.4f}"


def combo(coeffs, names):
    parts = []
    for c, n in zip(coeffs, names):
        if c:
            parts.append(("- " if c < 0 else "+ ") + ("" if abs(c) == 1 else f"{abs(c)} ") + n)
    s = " ".join(parts)
    return s[2:] if s.startswith("+ ") else "-" + s[2:]


def entropy_bits(p):
    return -sum(v * math.log2(v) for v in p if v > 0)


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def analyze(points, containers, masses, coords=True, theta=False):
    """Glue, fit, and measure identifiability. Returns a dict, or None if no glue."""
    M, labels = build(points, containers)
    y = [Fraction(1)] + [Fraction(m) for m in masses]
    if len(containers) <= 4:
        print("containers: " + "   ".join(
            f"{lab}={{{','.join(mem)}}}: {fmt(y[k + 1])}"
            for k, (lab, mem) in enumerate(containers)))
    support, w = fiber_support(M, y)
    if support is None:
        scale = math.lcm(*(x.denominator for x in w))
        wi = [int(x * scale) for x in w]
        g = math.gcd(*wi)
        wi = [x // g for x in wi]
        lhs = combo(wi[1:], [f"P({lab})" for lab in labels[1:]])
        value = sum(Fraction(a) * b for a, b in zip(wi[1:], y[1:]))
        print("glues: NO. Farkas certificate, true for every global distribution:")
        print(f"    {lhs} >= {-wi[0]}")
        print(f"  these local data give {fmt(value)}.")
        return None
    p, th = max_entropy(M, y, support)
    sub = [[row[x] for x in support] for row in M]
    K = []
    for k in kernel(sub, len(support)):
        full = [0] * len(points)
        for i, x in enumerate(support):
            full[x] = k[i]
        K.append(full)
    resid = max(abs(sum(M[k][x] * p[x] for x in range(len(points))) - float(y[k]))
                for k in range(len(M)))
    stationarity = max((abs(sum(k[x] * math.log(p[x]) for x in support)) for k in K), default=0.0)
    if coords:
        print("  point  container coords   p*")
        for x, name in enumerate(points):
            c = "(" + ",".join(str(M[k][x]) for k in range(1, len(M))) + ")"
            print(f"  {name:<6} {c:<18} {fmt(p[x])}")
    if theta:
        print("  logits theta (p* = softmax(theta . phi)): "
              + ", ".join(f"{labels[k]}: {v:+.4f}" for k, v in th.items()))
    print(f"glues: yes   support: {len(support)}/{len(points)} outcomes   "
          f"d_unk = {len(K)}   H(p*) = {entropy_bits(p):.4f} bits")
    for k in K:
        print(f"  invisible direction (ker M on the support): {combo(k, points)}")
    print(f"  checks: |M p* - y| = {resid:.1e}   max |<log p*, ker>| = {stationarity:.1e}")
    return {"M": M, "y": y, "p": p, "K": K, "support": support, "labels": labels}


def orthonormal(vectors):
    Q = []
    for v in vectors:
        u = [float(x) for x in v]
        for q in Q:
            d = sum(a * b for a, b in zip(u, q))
            u = [a - d * b for a, b in zip(u, q)]
        n = math.sqrt(sum(a * a for a in u))
        Q.append([a / n for a in u])
    return Q


def design(points, res, top=4):
    """Which single container best resolves the invisible directions?

    A container C resolves something iff 1_C is not orthogonal to ker M. If its
    mass is measured with error sigma, the recovered model moves by
    sigma / ||P_ker 1_C|| along the direction C resolves, so the score
    ||P_ker 1_C|| is the sensitivity: larger is better. The number of extra
    containers needed is forced to be d_unk (a matroid: each independent row
    lowers the nullity by exactly one); only the choice of containers is free.
    """
    support = res["support"]
    s = len(support)
    rows = [[row[x] for x in support] for row in res["M"]]
    chosen = []
    d0 = len(kernel(rows, s))
    while True:
        K = kernel(rows, s)
        if not K:
            break
        Q = orthonormal(K)
        cands = []
        for size in range(1, s):
            for C in combinations(range(s), size):
                if 2 * size > s or (2 * size == s and 0 not in C):
                    continue  # C and its complement carry the same information
                ind = [1 if i in C else 0 for i in range(s)]
                if not any(sum(a * b for a, b in zip(ind, k)) for k in K):
                    continue
                score = math.sqrt(sum(sum(a * b for a, b in zip(ind, q)) ** 2 for q in Q))
                cands.append((-round(score, 12), size, C, ind))
        cands.sort()
        name = lambda C: "{" + ",".join(points[support[i]] for i in C) + "}"
        if not chosen:
            print(f"  d_unk = {d0}; {len(cands)} containers (up to complement) would lower it.")
            print("  best single containers by sensitivity ||P_ker 1_C||:")
            for sc, _, C, _ in cands[:top]:
                print(f"    {name(C):<24} {-sc:.4f}")
            if len(K) == 1:
                k = Q[0]
                best = sum(v for v in k if v > 0)
                pos = [i for i in range(s) if k[i] > 0]
                print(f"  bound for d_unk = 1: max_C |<1_C, k>| = sum of positive parts of k "
                      f"= {best:.4f}, attained by {name(pos)}")
        _, _, C, ind = cands[0]
        chosen.append(name(C))
        rows.append(ind)
    print(f"  greedy completion to d_unk = 0: {', '.join(chosen)}  "
          f"({len(chosen)} container(s) = initial d_unk)")
    assert len(chosen) == d0


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

def main():
    F = Fraction
    ANTI = {"01": F(1, 2), "10": F(1, 2)}  # "the two variables always differ"
    q = dict(zip(CUBE, map(F, ["0.20", "0.05", "0.10", "0.15", "0.05", "0.15", "0.10", "0.20"])))

    header("1. Three points, two containers: the containers determine everything")
    pts = ["a", "b", "c"]
    con = [("C1", ["a", "b"]), ("C2", ["b", "c"])]
    m1, m2 = F("0.7"), F("0.6")
    analyze(pts, con, [m1, m2])
    print(f"  closed form: p_a = 1-m2 = {fmt(1 - m2)}, p_b = m1+m2-1 = {fmt(m1 + m2 - 1)}, "
          f"p_c = 1-m1 = {fmt(1 - m1)}")
    print("\n1'. Same containers, m1 + m2 < 1: even event containers can fail to glue")
    analyze(pts, con, ["0.3", "0.4"], coords=False)

    header("2. Four points, two containers: a metamer pair, entropy splits it evenly")
    pts4 = ["a", "b", "c", "d"]
    con4 = [("C1", ["a", "b", "c"]), ("C2", ["b", "c", "d"])]
    res2 = analyze(pts4, con4, ["0.8", "0.7"], theta=True)
    print("\n2'. Same containers, m1 + m2 = 1: the fiber collapses onto a face")
    analyze(pts4, con4, ["0.6", "0.4"], coords=False)
    print("  p_b = p_c = 0 is forced, L_p kills the b - c direction, d_unk drops to 0:")
    print("  the discrete version of the metamer set collapsing at the spectral locus.")

    header("2c. Copula case: two binary variables, containers = the two margins")
    pts2 = ["00", "01", "10", "11"]  # xy
    con2 = [("X=0", ["00", "01"]), ("Y=0", ["00", "10"])]
    res2c = analyze(pts2, con2, ["0.6", "0.3"])
    p = res2c["p"]
    print(f"  all four points have distinct container coordinates, yet d_unk = 1.")
    print(f"  odds ratio of p*: {p[0] * p[3] / (p[1] * p[2]):.6f}  (max entropy = independence)")
    t = 0.1
    alt = [v + t * k for v, k in zip(p, res2c["K"][0])]
    print(f"  metamer p* + {t}(00 - 01 - 10 + 11) = ({', '.join(f'{v:.2f}' for v in alt)}): "
          f"same margins, odds ratio {alt[0] * alt[3] / (alt[1] * alt[2]):.2f}")
    print("  The invisible direction is the dependence (log odds ratio) direction.")

    header("3. Gluing failure: three bits, pairwise containers, 'always differ'")
    tri = ["XY", "YZ", "XZ"]
    c3, m3 = cover(tri, {b: ANTI for b in tri})
    analyze(CUBE, c3, m3, coords=False)
    print("  Since P(XY=00) + P(XY=11) = 1 - P(X!=Y), the certificate reads 9 - 4S >= 1:")
    print("    S = P(X!=Y) + P(Y!=Z) + P(X!=Z) <= 2   for every global distribution,")
    print("  a Bell-type inequality (each 3-bit assignment has 0 or 2 unequal pairs).")
    print("  The local tables give S = 3, yet each agrees with its neighbours on the")
    print("  overlap (every single-variable margin is 1/2).")

    print("\n3'. Same triangle cover, tables taken from a real distribution q")
    c3b, m3b = cover(tri, {b: marginal(q, b) for b in tri})
    res3 = analyze(CUBE, c3b, m3b, coords=False)
    chi = [(-1) ** sum(map(int, w)) for w in CUBE]
    delta = [float(q[w]) - v for w, v in zip(CUBE, res3["p"])]
    tq = sum(d * c for d, c in zip(delta, chi)) / 8
    off = max(abs(d - tq * c) for d, c in zip(delta, chi))
    print(f"  q - p* = {tq:+.6f} * parity  (off-direction residual {off:.1e}):")
    print("  the truth and the max-ent fit are metamers; they differ only in the")
    print("  three-way interaction, which pairwise containers never see. The triangle")
    print("  is cyclic, so p* has no closed form (Newton / iterative fitting needed).")

    print("\n3''. Path cover {XY},{YZ}, tables from the same q")
    path = ["XY", "YZ"]
    c3c, m3c = cover(path, {b: marginal(q, b) for b in path})
    res3c = analyze(CUBE, c3c, m3c, coords=False)
    qxy, qyz, qy = marginal(q, "XY"), marginal(q, "YZ"), marginal(q, "Y")
    jt = [float(qxy[w[:2]] * qyz[w[1:]] / qy[w[1]]) for w in CUBE]
    err = max(abs(a - b) for a, b in zip(jt, res3c["p"]))
    print(f"  max |p* - p(x,y) p(y,z) / p(y)| = {err:.1e}: the path is acyclic, so the")
    print("  max-ent fit is the junction-tree product, i.e. X independent of Z given Y.")
    print("  This is the copula statement one level up: max entropy adds no dependence")
    print("  the containers did not supply, conditional on the separator.")

    header("4. Optimal containers: which ones remove d_unk, and at what sensitivity")
    print("Case 2 (metamer pair b, c):")
    design(pts4, res2)
    print("\nCase 2c (two margins):")
    design(pts2, res2c)
    print("\nCase 3' (triangle, three-way interaction invisible):")
    design(CUBE, res3)

    print("\nVariable covers on X, Y, Z (d_unk at an interior point):")
    parity = [("parity even", [w for w in CUBE if sum(map(int, w)) % 2 == 0])]
    rows = [
        ("independent {X},{Y},{Z}", ["X", "Y", "Z"], []),
        ("path {XY},{YZ}", ["XY", "YZ"], []),
        ("triangle {XY},{YZ},{XZ}", tri, []),
        ("triangle + parity event", tri, parity),
        ("full {XYZ}", ["XYZ"], []),
    ]
    print(f"  {'cover':<26}{'numbers':>8}{'d_unk':>7}   acyclic: local agreement => glue,"
          " closed-form max ent")
    for name, blocks, extra in rows:
        cons, _ = cover(blocks, {b: {} for b in blocks})
        M, _ = build(CUBE, cons + extra)
        rk = rank(M)
        acyc = "n/a (event container; triangle data can still fail)" if extra else (
            "yes" if is_acyclic(blocks) else "NO")
        print(f"  {name:<26}{rk - 1:>8}{len(CUBE) - rk:>7}   {acyc}")
    print("  For covers made only of variable blocks, d_unk = 0 forces a block that")
    print("  contains every variable. A single event container (parity) closes the")
    print("  triangle's gap, but it cannot repair the triangle's gluing obstruction.")


if __name__ == "__main__":
    main()
