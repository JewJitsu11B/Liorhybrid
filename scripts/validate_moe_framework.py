#!/usr/bin/env python3
"""Quick validation script for MoE framework."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    print("Checking imports...")
    try:
        from moe_framework import MoEConfig
        print("✅ Imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def check_config():
    print("\nChecking configuration...")
    try:
        from moe_framework import MoEConfig
        config = MoEConfig()
        print(f"✅ Config created: {config.num_experts} experts")
        return True
    except Exception as e:
        print(f"❌ Config failed: {e}")
        return False

def main():
    print("=" * 60)
    print("MoE Framework Validation")
    print("=" * 60 + "\n")
    
    results = [
        check_imports(),
        check_config(),
    ]
    
    print("\n" + "=" * 60)
    if all(results):
        print("🎉 ALL CHECKS PASSED")
        return 0
    else:
        print("⚠️ SOME CHECKS FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(main())
