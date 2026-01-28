"""
Usage Tracker for Liorhybrid
Logs when each file is imported/executed to help identify unused files.
"""
import json
import os
from datetime import datetime
from pathlib import Path
import threading

_lock = threading.Lock()
_TRACKER_FILE = Path(__file__).parent / "file_usage.json"


def track(filepath: str) -> None:
    """Record that a file was imported/executed."""
    # Get relative path from project root
    project_root = Path(__file__).parent
    try:
        rel_path = str(Path(filepath).relative_to(project_root))
    except ValueError:
        rel_path = filepath

    with _lock:
        # Load existing data
        if _TRACKER_FILE.exists():
            try:
                with open(_TRACKER_FILE, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = {}
        else:
            data = {}

        # Update entry
        now = datetime.now().isoformat()
        if rel_path not in data:
            data[rel_path] = {
                "first_called": now,
                "last_called": now,
                "call_count": 1
            }
        else:
            data[rel_path]["last_called"] = now
            data[rel_path]["call_count"] += 1

        # Write back
        with open(_TRACKER_FILE, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)


def report() -> dict:
    """Get the current usage report."""
    if _TRACKER_FILE.exists():
        with open(_TRACKER_FILE, "r") as f:
            return json.load(f)
    return {}


def clear() -> None:
    """Clear the usage data."""
    if _TRACKER_FILE.exists():
        os.remove(_TRACKER_FILE)


def find_unused() -> list:
    """Find Python files that haven't been tracked."""
    project_root = Path(__file__).parent
    all_py_files = set()

    for py_file in project_root.rglob("*.py"):
        # Skip env, __pycache__, and this tracker
        rel = str(py_file.relative_to(project_root))
        if any(skip in rel for skip in ['env/', '__pycache__', 'usage_tracker.py']):
            continue
        all_py_files.add(rel)

    used_files = set(report().keys())
    unused = sorted(all_py_files - used_files)
    return unused


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "report":
            data = report()
            print(json.dumps(data, indent=2))
        elif cmd == "unused":
            unused = find_unused()
            print(f"Found {len(unused)} unused files:")
            for f in unused:
                print(f"  {f}")
        elif cmd == "clear":
            clear()
            print("Usage data cleared.")
    else:
        print("Usage: python usage_tracker.py [report|unused|clear]")
