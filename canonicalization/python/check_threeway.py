"""Three-readout relation, labels T1--T3.

    det X = (tr X)^2 / 4 + (1/8) kappa(iX, iX)      (X Hermitian 2x2)
    N(X)  = <X>_0^2     + (1/8) kappa(IX, IX)       (X a paravector of Cl(3,0))

T1  matrix form, exact, Killing form built from ad matrices   (threeway/A1_check.py)
T2  the SU(2)-invariant quadratics on Herm(2) are 2-dimensional,
    so the constants 1/4 and 1/8 are forced                   (threeway/C3_check.py)
T3  Clifford form, no matrices                                (threeway/A3_clifford.py)

Each label passes when its script exits 0.
"""
import os
import subprocess
import sys

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threeway")
checks = [("T1", "A1_check.py"), ("T2", "C3_check.py"), ("T3", "A3_clifford.py")]
ok = 0
for label, name in checks:
    r = subprocess.run([sys.executable, os.path.join(here, name)], capture_output=True, text=True)
    passed = r.returncode == 0
    ok += passed
    print(label, "PASS" if passed else "FAIL", name)
    if not passed:
        print(r.stdout[-2000:], r.stderr[-2000:])
print("%d of %d pass" % (ok, len(checks)))
sys.exit(0 if ok == len(checks) else 1)
