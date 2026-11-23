# Agent A: shared helpers (independent code; does not import the author's script).
import sys
from decimal import Decimal
import mpmath as mp

FAILS = []


def check(label, ok, text):
    print(("PASS " if ok else "FAIL ") + label + " " + text)
    sys.stdout.flush()
    if not ok:
        FAILS.append(label)


def info(label, text):
    print("INFO " + label + " " + text)
    sys.stdout.flush()


def finish(name):
    print("SUMMARY %s: %d failed check(s)%s" % (name, len(FAILS), (": " + "; ".join(FAILS)) if FAILS else ""))
    sys.exit(1 if FAILS else 0)


def digits_agree(printed, value):
    """Compare value with a printed decimal string. Returns (all_agree, n_agree, n_printed, mine_str)."""
    d = Decimal(printed)
    exp = d.as_tuple().exponent
    pint = int(d.scaleb(-exp))
    mint = int(mp.nint(mp.mpf(value) / mp.mpf(10) ** exp))
    ps, ms = str(abs(pint)), str(abs(mint))
    nprint = len(ps)
    mine = str(Decimal(mint).scaleb(exp))
    if pint == mint:
        return True, nprint, nprint, mine
    if (pint < 0) != (mint < 0) and pint != 0 and mint != 0:
        return False, 0, nprint, mine
    if len(ps) != len(ms):
        return False, 0, nprint, mine
    n = 0
    for a, b in zip(ps, ms):
        if a != b:
            break
        n += 1
    return False, n, nprint, mine


def table_check(label, printed, value):
    ok, n, npr, mine = digits_agree(printed, value)
    check(label, ok, "printed %s, mine %s (%s), %d of %d printed digits agree"
          % (printed, mine, mp.nstr(mp.mpf(value), 12), n, npr))
    return ok
