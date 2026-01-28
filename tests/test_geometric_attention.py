"""
Test Geometric Attention Mechanism

Verifies field-to-KV extraction and geometric attention layers.

Tests:
- Field state flattening and projection
- Positional encoding
- Attention weight normalization (softmax)
- Full transformer forward pass
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import pytest
from ..core import CognitiveTensorField, FAST_TEST_CONFIG
from ..inference.field_extraction import (
    flatten_field_state,
    FieldToKeyValue,
    extract_keys_values_from_field
)
from ..inference.geometric_attention import (
    GeometricAttention,
    GeometricTransformer
)


def test_flatten_field_state():
    """
    Test that field flattening produces correct shape.

    T_field: (N_x, N_y, D, D) complex
    Output: (N_x * N_y, 2*D*D) real
    """
    N_x, N_y, D = 8, 8, 4
    T_field = torch.randn(N_x, N_y, D, D, dtype=torch.complex64)

    T_flat = flatten_field_state(T_field)

    expected_tokens = N_x * N_y
    expected_dim = 2 * D * D  # Real + imaginary

    assert T_flat.shape == (expected_tokens, expected_dim)
    assert T_flat.dtype == torch.float32


def test_field_to_kv_shape():
    """
    Test that FieldToKeyValue produces correct K, V shapes.
    """
    N_x, N_y, D = 8, 8, 4
    d_model = 512

    T_field = torch.randn(N_x, N_y, D, D, dtype=torch.complex64)

    field_to_kv = FieldToKeyValue(field_dim=D, d_model=d_model)
    K, V = field_to_kv(T_field)

    expected_tokens = N_x * N_y

    assert K.shape == (expected_tokens, d_model)
    assert V.shape == (expected_tokens, d_model)


def test_field_to_kv_positional_encoding():
    """
    Test that positional encoding changes K, V values.
    """
    N_x, N_y, D = 8, 8, 4
    d_model = 64

    T_field = torch.randn(N_x, N_y, D, D, dtype=torch.complex64)

    # With positional encoding
    field_to_kv_pos = FieldToKeyValue(
        field_dim=D,
        d_model=d_model,
        use_positional_encoding=True
    )
    K_pos, V_pos = field_to_kv_pos(T_field)

    # Without positional encoding
    field_to_kv_no_pos = FieldToKeyValue(
        field_dim=D,
        d_model=d_model,
        use_positional_encoding=False
    )
    K_no_pos, V_no_pos = field_to_kv_no_pos(T_field)

    # Should be different
    assert not torch.allclose(K_pos, K_no_pos)
    assert not torch.allclose(V_pos, V_no_pos)


def test_field_to_kv_temporal_encoding():
    """
    Test that temporal encoding affects K, V.
    """
    N_x, N_y, D = 8, 8, 4
    d_model = 64

    T_field = torch.randn(N_x, N_y, D, D, dtype=torch.complex64)

    field_to_kv = FieldToKeyValue(
        field_dim=D,
        d_model=d_model,
        use_temporal_encoding=True
    )

    # Different times
    K1, V1 = field_to_kv(T_field, time=0.0)
    K2, V2 = field_to_kv(T_field, time=10.0)

    # Should be different
    assert not torch.allclose(K1, K2)
    assert not torch.allclose(V1, V2)


def test_geometric_attention_forward():
    """
    Test that GeometricAttention produces correct output shape.
    """
    batch_size = 2
    seq_len_q = 10
    seq_len_k = 64  # N_x * N_y
    d_model = 64
    n_heads = 4

    Q_input = torch.randn(batch_size, seq_len_q, d_model)
    K = torch.randn(batch_size, seq_len_k, d_model)
    V = torch.randn(batch_size, seq_len_k, d_model)
    T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)

    attention = GeometricAttention(d_model=d_model, n_heads=n_heads)
    output, attn_weights = attention(Q_input, K, V, T_field)

    # Check output shape
    assert output.shape == (batch_size, seq_len_q, d_model)

    # Check attention weights shape
    assert attn_weights.shape == (batch_size, n_heads, seq_len_q, seq_len_k)


def test_attention_weights_sum_to_one():
    """
    Test that attention weights are normalized (sum to 1 via softmax).
    """
    batch_size = 2
    seq_len_q = 10
    seq_len_k = 64
    d_model = 64
    n_heads = 4

    Q_input = torch.randn(batch_size, seq_len_q, d_model)
    K = torch.randn(batch_size, seq_len_k, d_model)
    V = torch.randn(batch_size, seq_len_k, d_model)
    T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)

    attention = GeometricAttention(d_model=d_model, n_heads=n_heads)
    attention.eval()  # Set to evaluation mode (disables dropout)

    with torch.no_grad():
        _, attn_weights = attention(Q_input, K, V, T_field)

    # Sum over keys dimension should equal 1
    weights_sum = torch.sum(attn_weights, dim=-1)  # (batch, n_heads, seq_len_q)

    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)


def test_attention_weights_non_negative():
    """
    Test that attention weights are non-negative (softmax property).
    """
    batch_size = 2
    seq_len_q = 10
    seq_len_k = 64
    d_model = 64
    n_heads = 4

    Q_input = torch.randn(batch_size, seq_len_q, d_model)
    K = torch.randn(batch_size, seq_len_k, d_model)
    V = torch.randn(batch_size, seq_len_k, d_model)
    T_field = torch.randn(8, 8, 4, 4, dtype=torch.complex64)

    attention = GeometricAttention(d_model=d_model, n_heads=n_heads)
    _, attn_weights = attention(Q_input, K, V, T_field)

    # All weights should be >= 0
    assert torch.all(attn_weights >= 0)


def test_geometric_transformer_integration():
    """
    Test full GeometricTransformer with evolved field.

    This is the key integration test: field evolution → transformer query.
    """
    # Create and evolve field
    config = FAST_TEST_CONFIG
    field = CognitiveTensorField(config)

    # Evolve for a few steps
    for _ in range(10):
        field.evolve_step()

    # Create transformer
    field_dim = config.tensor_dim  # Should be 4
    d_model = 64
    n_heads = 4

    transformer = GeometricTransformer(
        field_dim=field_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=2
    )

    # Create query input (simulated prompt embedding)
    batch_size = 1
    seq_len = 10
    Q_input = torch.randn(batch_size, seq_len, d_model)

    # Query the field
    output, attn_weights_list = transformer(Q_input, field.T, time=field.t)

    # Check output
    assert output.shape == (batch_size, seq_len, d_model)
    assert len(attn_weights_list) == 2  # 2 layers


def test_geometric_transformer_field_dependence():
    """
    Test that transformer output depends on field state.

    Different field states should produce different outputs.
    """
    config = FAST_TEST_CONFIG
    field_dim = config.tensor_dim
    d_model = 64

    transformer = GeometricTransformer(
        field_dim=field_dim,
        d_model=d_model,
        n_heads=4,
        n_layers=1
    )

    # Same query
    Q_input = torch.randn(1, 10, d_model)

    # Two different field states
    torch.manual_seed(42)
    field1 = CognitiveTensorField(config)

    torch.manual_seed(99)
    field2 = CognitiveTensorField(config)

    output1, _ = transformer(Q_input, field1.T)
    output2, _ = transformer(Q_input, field2.T)

    # Outputs should be different
    assert not torch.allclose(output1, output2, rtol=0.1)


def test_geometric_transformer_learnable_weights():
    """
    Test that geometric weights are learnable parameters.
    """
    transformer = GeometricTransformer(
        field_dim=4,
        d_model=64,
        n_heads=4,
        n_layers=1
    )

    # Check that geometric_weights are parameters
    found_geometric_weights = False
    for name, param in transformer.named_parameters():
        if 'geometric_weights' in name:
            found_geometric_weights = True
            assert param.requires_grad, "Geometric weights should be learnable"

    assert found_geometric_weights, "Should have geometric_weights parameters"


def test_extract_keys_values_convenience():
    """
    Test convenience function extract_keys_values_from_field.
    """
    N_x, N_y, D = 8, 8, 4
    d_model = 64

    T_field = torch.randn(N_x, N_y, D, D, dtype=torch.complex64)
    field_to_kv = FieldToKeyValue(field_dim=D, d_model=d_model)

    K, V = extract_keys_values_from_field(T_field, field_to_kv)

    assert K.shape == (N_x * N_y, d_model)
    assert V.shape == (N_x * N_y, d_model)
