from __future__ import annotations
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import builtins
import traceback


def enable_matplotlib_import_trace() -> None:
    original_import = builtins.__import__

    def traced_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)

        top = name.split(".", 1)[0]
        if top == "matplotlib":
            stack = "".join(traceback.format_stack(limit=25))
            print("\n[import-trace] matplotlib import detected")
            print(f"[import-trace] name={name!r} fromlist={fromlist!r} level={level}")
            print("[import-trace] stack (most recent last):")
            print(stack)

        return module

    builtins.__import__ = traced_import