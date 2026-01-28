#!/usr/bin/env python
"""
Structure Validation Script

Checks that all modules can be imported and basic functionality works.
Run this after installation to verify the package structure.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import sys
from pathlib import Path

print("=" * 70)
print("Bayesian Cognitive Field - Structure Validation")
print("=" * 70)

# Add package to path for testing before installation
package_root = Path(__file__).parent
sys.path.insert(0, str(package_root.parent))

errors = []
warnings = []

# Test 1: Import core modules
print("\n1. Testing core module imports...")
try:
    from Liorhybrid.core import (
        CognitiveTensorField,
        FieldConfig,
        get_default_config,
        MNIST_CONFIG,
        FAST_TEST_CONFIG
    )
    print("   ✓ Core imports successful")
except ImportError as e:
    errors.append(f"Core import failed: {e}")
    print(f"   ✗ Core import failed: {e}")

# Test 2: Import kernel modules
print("\n2. Testing kernel module imports...")
try:
    from Liorhybrid.kernels import (
        hamiltonian_evolution,
        bayesian_recursive_term,
        fractional_memory_term,
        spatial_laplacian
    )
    print("   ✓ Kernel imports successful")
except ImportError as e:
    errors.append(f"Kernel import failed: {e}")
    print(f"   ✗ Kernel import failed: {e}")

# Test 3: Import operator modules
print("\n3. Testing operator module imports...")
try:
    from Liorhybrid.operators import (
        collapse_operator,
        measure_observable,
        soft_projection
    )
    print("   ✓ Operator imports successful")
except ImportError as e:
    errors.append(f"Operator import failed: {e}")
    print(f"   ✗ Operator import failed: {e}")

# Test 4: Import utility modules
print("\n4. Testing utility module imports...")
try:
    from Liorhybrid.utils import (
        compute_norm_conservation,
        compute_local_correlation,
        compute_effective_dimension
    )
    print("   ✓ Utility imports successful")
except ImportError as e:
    errors.append(f"Utility import failed: {e}")
    print(f"   ✗ Utility import failed: {e}")

# Test 5: Create a field instance
print("\n5. Testing field instantiation...")
try:
    import torch
    from Liorhybrid.core import CognitiveTensorField, FAST_TEST_CONFIG

    field = CognitiveTensorField(FAST_TEST_CONFIG)
    print(f"   ✓ Field created: shape {field.T.shape}")
    print(f"     - Device: {field.device}")
    print(f"     - Initial norm: {field.get_norm_squared():.6f}")
except Exception as e:
    errors.append(f"Field instantiation failed: {e}")
    print(f"   ✗ Field instantiation failed: {e}")

# Test 6: Run a few evolution steps
print("\n6. Testing basic evolution...")
try:
    for i in range(5):
        field.evolve_step()

    print(f"   ✓ Evolution successful (5 steps)")
    print(f"     - Final norm: {field.get_norm_squared():.6f}")
    print(f"     - Time: {field.t:.6f}")
    print(f"     - History size: {len(field.history)}")
except Exception as e:
    errors.append(f"Evolution failed: {e}")
    print(f"   ✗ Evolution failed: {e}")

# Test 7: Check file structure
print("\n7. Checking file structure...")
expected_files = [
    "core/__init__.py",
    "core/config.py",
    "core/tensor_field.py",
    "kernels/__init__.py",
    "kernels/hamiltonian.py",
    "kernels/bayesian.py",
    "kernels/fractional_memory.py",
    "operators/__init__.py",
    "operators/collapse.py",
    "utils/__init__.py",
    "utils/metrics.py",
    "utils/visualization.py",
    "tests/__init__.py",
    "examples/simple_evolution.py",
    "README.md",
    "requirements.txt",
    "setup.py",
]

missing_files = []
for file_path in expected_files:
    full_path = package_root / file_path
    if not full_path.exists():
        missing_files.append(file_path)

if missing_files:
    warnings.append(f"Missing files: {', '.join(missing_files)}")
    print(f"   ⚠ Some files missing: {len(missing_files)}")
    for f in missing_files:
        print(f"     - {f}")
else:
    print("   ✓ All expected files present")

# Summary
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

if not errors and not warnings:
    print("\n✓ All tests passed! Package structure is valid.")
    sys.exit(0)
elif not errors:
    print(f"\n⚠ Tests passed with {len(warnings)} warning(s):")
    for w in warnings:
        print(f"  - {w}")
    sys.exit(0)
else:
    print(f"\n✗ Validation failed with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    if warnings:
        print(f"\n  and {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    sys.exit(1)
