"""
Geometric Attention Inference Module

Implements the cognitive interface to the T_ij tensor field.

Architecture (Full Stack):
1. Geometric Mamba Encoder (O(N) base processing)
   - CI8 state space with Trinor/Wedge/Spinor operators
   - Non-associative dynamics for causal structure
2. SBERT Pooling (O(N) aggregation)
3. DPR K/V Generation (statistical optimization)
4. Geometric Attention (field-contracted products)
   - Wedge: Causal divergence
   - Tensor: Signal strength
   - Spinor: Rotational alignment

This module bridges the physics-based field evolution with
modern transformer-based inference at O(N) complexity.
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

from .geometric_mamba import (
    ComplexOctonion,
    TrinorOperator,
    WedgeProjection,
    SpinorProjection,
    GeometricMambaLayer,
    GeometricMambaEncoder,
    field_to_ci8,
    ci8_to_field
)

from .geometric_stack import (
    SBERTPooling,
    GeometricStack,
    GeometricTransformerWithMamba,
    demonstrate_geometric_operators
)

from .dpr_encoder import (
    DPRKeyValueGenerator,
    DPRIntegrationConfig
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
    # Geometric Mamba (O(N))
    'ComplexOctonion',
    'TrinorOperator',
    'WedgeProjection',
    'SpinorProjection',
    'GeometricMambaLayer',
    'GeometricMambaEncoder',
    'field_to_ci8',
    'ci8_to_field',
    # Full geometric stack
    'SBERTPooling',
    'GeometricStack',
    'GeometricTransformerWithMamba',
    'demonstrate_geometric_operators',
    # DPR integration
    'DPRKeyValueGenerator',
    'DPRIntegrationConfig'
]
