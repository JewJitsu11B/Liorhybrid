"""Agent N follow-up checks, written after reading the author's script (uses its definitions where stated)."""
import sys, importlib.util
import numpy as np, mpmath as mp
from math import pi, gamma
from N_own import weights, W_jonq, root, iterate, packet_modes, fit_two

def fit_undamped(J, ph, cyc2):
    tail = J[-cyc2:]; nn = np.arange(len(tail))
    A = np.stack([np.exp(1j*ph*nn), np.exp(-1j*ph*nn)], axis=1)
    c, *_ = np.linalg.lstsq(A, tail, rcond=None)
    return abs(c[0])**2, abs(c[1])**2

print("(a) Table packet, my iteration, author's metric definitions (CAL renormalised, Schr at exactly 0.5, undamped fit, ratio cm/cp)")
k, c = packet_modes(); c = c/np.linalg.norm(c)
for e, r2 in ((1e-2,.06),(1e-3,.06),(1e-4,.24),(1e-4,.12),(1e-4,.06)):
    alpha = 1-e; g = np.exp(-r2*k**2/6)
    phis = np.array([root(alpha, gg) for gg in g]); r0 = root(alpha, 1.0)
    N = int(round(0.5/(r2/12)/r0.real)); cyc2 = int(round(4*pi/r0.real))
    w = weights(alpha, N+cyc2+5)
    Wh = np.array([complex(W_jonq(p.real, alpha)) for p in phis])
    J = iterate(alpha, g, c, N, past=(phis.real.copy(), Wh), w=w)
    schr = c*np.exp(-0.5j*k**2)
    def al(u, v):
        u = u/np.linalg.norm(u); ph = np.vdot(u, v); return np.linalg.norm(u*ph/abs(ph)-v)
    def raw(u, v):
        ph = np.vdot(u, v); return np.linalg.norm(u*ph/abs(ph)-v)
    eig = c*np.exp(1j*(phis.real-r0.real)*N)
    cp = cm = 0; cpd = cmd = 0
    for m in range(len(k)):
        a, b = fit_undamped(J[:, m], phis[m].real, cyc2); cp += a; cm += b
        A, B = fit_two(J[:, m], phis[m], N+1-cyc2, N+1); cpd += abs(A)**2; cmd += abs(B)**2
    hist = c[None,:]*np.exp(1j*phis.real[None,:]*np.arange(-cyc2//1, 0)[:,None])
    J2 = iterate(alpha, g, c, N, w=w, hist=hist)
    cp2 = cm2 = 0
    for m in range(len(k)):
        a, b = fit_undamped(J2[:, m], phis[m].real, cyc2); cp2 += a; cm2 += b
    print(f" eps={e:.0e} r2={r2} N={N}: err(renorm)={al(J[N],schr):.3e} err(phase only, no renorm)={raw(J[N],schr):.3e}"
          f" floor={al(eig,schr):.3e} ||J_N||-1={np.linalg.norm(J[N])-1:+.3e}"
          f" counter cm/cp undamped-fit={cm/cp:.1e} damped-fit={cmd/cpd:.1e} |"
          f" two-cycle history: cm/cp={cm2/cp2:.3f} cm/(cm+cp)={cm2/(cm2+cp2):.3f} ||J_N||={np.linalg.norm(J2[N]):.3f}", flush=True)

print("(b) Thm feedback (4) criterion thresholds at alpha=1.01: flat weights vs the power-law weights the run uses")
a = 1.01
j = np.arange(1, 200001, dtype=float)
wp = (j**a-(j-1)**a)/gamma(a+1)
for lam in (0.1, 0.01):
    Wf = 1/np.expm1(lam); Wp = np.sum(wp*np.exp(-lam*j))
    print(f" lam={lam}: W_lam(1) flat={Wf:.4f} -> b>{1-1/(a+(a-1)*Wf):.4f};  power-law={Wp:.4f} -> b>{1-1/(a+(a-1)*Wp):.4f}")

print("(c) author's Euler-Maclaurin W vs mpmath (30 digits)")
spec = importlib.util.spec_from_file_location("auth", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "..", "scripts", "cal_edge_transducer.py")); au = importlib.util.module_from_spec(spec); spec.loader.exec_module(au)
mx = 0
for al_ in (0.9, 0.99, 0.999, 1-1e-6):
    for ph in (0.3-0.02j, 0.05-0.005j, 0.0316+2.6e-5j, 0.01, 1.0, 3.0):
        d = abs(au.W(ph, al_) - complex(W_jonq(ph, al_)))/abs(au.W(ph, al_)); mx = max(mx, d)
print(f" max relative difference = {mx:.1e}")

print("(d) Table closure, my packets (same-width, both centred at 0), r2=0.12, swing = ptp/mean over all updates, 8 seeds")
ka, ca = packet_modes(1.0); kb, cb = packet_modes(-1.5)
kk = np.union1d(ka, kb); A0 = np.zeros(len(kk), complex); B0 = A0.copy()
A0[np.searchsorted(kk, ka)] = ca; B0[np.searchsorted(kk, kb)] = cb
for e in (1e-2, 1e-3, 1e-4):
    alpha = 1-e; g = np.exp(-0.12*kk**2/6); r0 = root(alpha, 1.0); N = int(round(60/r0.real))
    Ja = iterate(alpha, g, A0, N); Jb = iterate(alpha, g, B0, N)
    aa = np.sum(abs(Ja)**2, 1); bb = np.sum(abs(Jb)**2, 1); ab = np.sum(Ja*Jb.conj(), 1)
    sw = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        U = rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)); V = rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
        trQ = aa*np.trace(U@U.conj().T).real + bb*np.trace(V@V.conj().T).real + 2*(ab*np.trace(U@V.conj().T)).real
        sw.append(np.ptp(trQ)/trQ.mean())
    print(f" eps={e:.0e}: zero-past trQ ptp/mean {min(sw):.2f}..{max(sw):.2f}")
