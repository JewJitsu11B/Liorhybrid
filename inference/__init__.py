"""
Geometric Attention Inference Module

Implements the cognitive interface to the T_ij tensor field.

Architecture (Full Stack):
1. CausalField Encoder (O(N log N) parallel via FFT)
   - BiQuaternion state space
   - Causal structure through convolution
2. SBERT Pooling (O(N) aggregation)
3. Composite K Structure (future: 9-field address)
4. Geometric Attention (field-contracted products)
   - Wedge: Causal divergence
   - Tensor: Signal strength
   - Spinor: Rotational alignment
   - Hodge: Dual space

This module bridges the physics-based field evolution with
modern transformer-based inference at O(N log N) complexity.
"""

from .geometric_products import (
    wedge_product,
    tensor_product,
    spinor_product,
    geometric_score
)

from .field_extraction import (
    extract_keys_values_from_field
)

from .geometric_attention import (
    GeometricAttention,
    GeometricTransformer
)

from .geometric_stack import (
    SBERTPooling,
    GeometricStack,
    GeometricTransformerWithMamba,
    demonstrate_geometric_operators
)

from .composite_k import (
    CompositeK,
    CompositeKFields,
    OutputSplitter,
    ThawSchedule,
    LocalMetricComputer,
    ChristoffelComputer,
    KNNModule,
    BCHEncoder,
    create_composite_k_system
)

from .constitutive_state import (
    MaterialProperties,
    BivectorDecomposition,
    StagedConcept,
    ConstitutiveState,
    ConstitutiveLayer,
    ConstitutiveStack
)

__all__ = [
    # Geometric products
    'wedge_product',
    'tensor_product',
    'spinor_product',
    'geometric_score',
    # Field extraction
    'extract_keys_values_from_field',
    # Legacy geometric attention (O(N²))
    'GeometricAttention',
    'GeometricTransformer',
    # Full geometric stack
    'SBERTPooling',
    'GeometricStack',
    'GeometricTransformerWithMamba',
    'demonstrate_geometric_operators',
    # Composite K structure
    'CompositeK',
    'CompositeKFields',
    'OutputSplitter',
    'ThawSchedule',
    'LocalMetricComputer',
    'ChristoffelComputer',
    'KNNModule',
    'BCHEncoder',
    'create_composite_k_system',
    # Constitutive cognitive state (non-Markovian)
    'MaterialProperties',
    'BivectorDecomposition',
    'StagedConcept',
    'ConstitutiveState',
    'ConstitutiveLayer',
    'ConstitutiveStack',
]
