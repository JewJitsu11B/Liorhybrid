"""
Core Classes for Bayesian Cognitive Field

Main components:
- FieldConfig: Configuration dataclass with all parameters
- CognitiveTensorField: Main evolution class implementing Paper Algorithm 1

Paper Reference: Section 5 (Implementation)
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from .config import FieldConfig, get_default_config, MNIST_CONFIG, FAST_TEST_CONFIG
from .tensor_field import CognitiveTensorField

__all__ = [
    'FieldConfig',
    'get_default_config',
    'MNIST_CONFIG',
    'FAST_TEST_CONFIG',
    'CognitiveTensorField',
]
