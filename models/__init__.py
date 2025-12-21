"""
Model Components

Additional model components for the cognitive field system.

Components:
- Language model head (vocabulary projection)
- Multimodal fusion layers
- Task-specific heads (classification, regression, etc.)
- LIoR kernel and memory state (O(1) recurrence)
- Complex metric tensor (G = A + iB)
- Cognitive manifold (coordinate spacetime from geometry)
- Causal field layer (parallel field evolution with Pi, Gamma, Phi)
- Activation functions (SwiGLU, FFN)
"""

from .language_head import LanguageModelHead

# LIoR kernel components
from .lior_kernel import LiorKernel, LiorMemoryState

# Complex metric components
from .complex_metric import ComplexMetricTensor, SpinorBilinears, PhaseOrthogonalProjector

# Cognitive manifold
from .manifold import CognitiveManifold

# Causal field components
from .causal_field import (
    AssociatorCurrent,
    ParallelTransport,
    CliffordConnection,
    CausalFieldLayer,
    CausalFieldBlock,
)

# Activation functions
from .activations import SwiGLU, FFN

__all__ = [
    'LanguageModelHead',
    # LIoR kernel
    'LiorKernel',
    'LiorMemoryState',
    # Complex metric
    'ComplexMetricTensor',
    'SpinorBilinears',
    'PhaseOrthogonalProjector',
    # Manifold
    'CognitiveManifold',
    # Causal field
    'AssociatorCurrent',
    'ParallelTransport',
    'CliffordConnection',
    'CausalFieldLayer',
    'CausalFieldBlock',
    # Activations
    'SwiGLU',
    'FFN',
]
