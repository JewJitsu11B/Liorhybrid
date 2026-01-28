"""
Execution Tracker - First-Call Detection

Tracks function/method first calls during training for execution flow visibility.
Logs: function name, module, file location, line number, and calling context.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import functools
import inspect
import os

# Global registry of called functions
_CALLED_REGISTRY = set()


def track_first_call(func):
    """
    Decorator to alert on first call of a function/method.

    Prints alert with:
    - Function name and module
    - Source file and line number
    - Caller information

    Usage:
        @track_first_call
        def my_function(...):
            ...

    Args:
        func: Function or method to track

    Returns:
        Wrapped function that logs on first call
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Optional: disable noisy banners (preferred when using pipeline audit one-line flags).
        if os.environ.get("BCF_DISABLE_FIRST_CALL_ALERTS", "").strip().lower() in ("1", "true", "yes", "on"):
            return func(*args, **kwargs)

        # Avoid graph breaks during torch.compile tracing.
        try:
            import torch

            if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
                return func(*args, **kwargs)
        except Exception:
            pass

        # Create unique identifier for this function
        func_id = f"{func.__module__}.{func.__qualname__}"

        # Check if this is the first call
        if func_id not in _CALLED_REGISTRY:
            _CALLED_REGISTRY.add(func_id)

            # Get source location
            try:
                source_file = inspect.getsourcefile(func)
                if source_file is None:
                    source_file = "unknown"
            except:
                source_file = "unknown"

            try:
                source_lines = inspect.getsourcelines(func)
                source_line = source_lines[1]
            except:
                source_line = 0

            # Get caller context
            try:
                frame = inspect.currentframe().f_back
                caller_file = frame.f_code.co_filename
                caller_line = frame.f_lineno
                caller_func = frame.f_code.co_name
                caller_info = f"{caller_file}:{caller_line} in {caller_func}()"
            except:
                caller_info = "unknown"

            # Print alert
            if os.environ.get("BCF_FIRST_CALL_COMPACT", "").strip().lower() in ("1", "true", "yes", "on"):
                print(f"[FIRSTCALL] {func.__module__}.{func.__qualname__} :: {source_file}:{source_line}", flush=True)
            else:
                print(f"\n{'='*80}")
                print(f"⚠  FIRST CALL ALERT")
                print(f"{'='*80}")
                print(f"Function:  {func.__qualname__}")
                print(f"Module:    {func.__module__}")
                print(f"Location:  {source_file}:{source_line}")
                print(f"Called by: {caller_info}")
                print(f"{'='*80}\n")

        # Call the original function
        return func(*args, **kwargs)

    return wrapper


def reset_tracker():
    """
    Reset the call registry.
    Useful for testing or multi-run scenarios.
    """
    global _CALLED_REGISTRY
    _CALLED_REGISTRY.clear()


def get_called_functions():
    """
    Get list of all functions that have been called.

    Returns:
        List of function identifiers (module.qualname)
    """
    return list(_CALLED_REGISTRY)
