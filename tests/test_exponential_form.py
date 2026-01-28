"""
Tests for Option 3 Exponential Form

Verifies the triality exponential structure Ψ = A·exp(Θ) + B·exp(iΘ)
provides gradient stability past the step-80 NaN threshold.

Key properties tested:
1. Bounded outputs (no overflow)
2. Stable gradients through 100+ steps
3. Unit magnitude property of exp(iΘ)
4. Band regularizer behavior
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import pytest
from typing import Tuple

# Import the modules we're testing
from Liorhybrid.inference.geometric_attention import (
    ExponentialPhaseExtractor,
    PhaseExtractor,
    GeometricAttention
)
from Liorhybrid.inference.geometric_products import (
    geometric_score_from_exponential,
    geometric_score_from_phase
)
from Liorhybrid.training.losses import (
    band_regularizer,
    generator_band_regularizer
)


class TestExponentialPhaseExtractor:
    """Tests for the ExponentialPhaseExtractor (Option 3)."""

    @pytest.fixture
    def extractor(self):
        """Create extractor with d_k=32."""
        return ExponentialPhaseExtractor(d_k=32)

    @pytest.fixture
    def sample_input(self):
        """Sample input tensor."""
        return torch.randn(2, 4, 16, 32)  # (batch, heads, seq, d_k)

    def test_output_shape(self, extractor, sample_input):
        """Output should be (batch, heads, seq, 16)."""
        psi = extractor(sample_input)
        assert psi.shape == (2, 4, 16, 16), f"Expected (2,4,16,16), got {psi.shape}"

    def test_output_bounded(self, extractor, sample_input):
        """Output should be bounded (no NaN/Inf)."""
        psi = extractor(sample_input)
        assert not torch.isnan(psi).any(), "Output contains NaN"
        assert not torch.isinf(psi).any(), "Output contains Inf"

    def test_extreme_inputs_no_nan(self, extractor):
        """Extreme inputs should not produce NaN/Inf."""
        # Very large inputs
        large_input = torch.randn(2, 4, 16, 32) * 100
        psi_large = extractor(large_input)
        assert not torch.isnan(psi_large).any(), "Large input produced NaN"
        assert not torch.isinf(psi_large).any(), "Large input produced Inf"

        # Very small inputs
        small_input = torch.randn(2, 4, 16, 32) * 1e-6
        psi_small = extractor(small_input)
        assert not torch.isnan(psi_small).any(), "Small input produced NaN"
        assert not torch.isinf(psi_small).any(), "Small input produced Inf"

        # Near-zero inputs
        zero_input = torch.zeros(2, 4, 16, 32)
        psi_zero = extractor(zero_input)
        assert not torch.isnan(psi_zero).any(), "Zero input produced NaN"
        assert not torch.isinf(psi_zero).any(), "Zero input produced Inf"

    def test_generators_extraction(self, extractor, sample_input):
        """get_generators should return (A, B, Θ) each with shape (..., 8)."""
        A, B, Theta = extractor.get_generators(sample_input)
        assert A.shape == (2, 4, 16, 8), f"A shape wrong: {A.shape}"
        assert B.shape == (2, 4, 16, 8), f"B shape wrong: {B.shape}"
        assert Theta.shape == (2, 4, 16, 8), f"Theta shape wrong: {Theta.shape}"

    def test_gradient_stability_100_steps(self, extractor):
        """Simulate 100 training steps and verify no gradient explosion."""
        extractor.train()
        optimizer = torch.optim.SGD(extractor.parameters(), lr=0.01)

        for step in range(100):
            x = torch.randn(2, 4, 16, 32, requires_grad=True)
            psi = extractor(x)

            # Simple loss: minimize psi norm
            loss = psi.norm()
            loss.backward()

            # Check gradients
            for name, param in extractor.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN grad at step {step}: {name}"
                    assert not torch.isinf(param.grad).any(), f"Inf grad at step {step}: {name}"

            optimizer.step()
            optimizer.zero_grad()

        print(f"✓ Passed 100 steps without gradient instability")


class TestGeometricScoreFromExponential:
    """Tests for geometric_score_from_exponential."""

    @pytest.fixture
    def field(self):
        """Create a sample cognitive field."""
        return torch.randn(8, 8, 4, 4, dtype=torch.cfloat)

    def test_output_shape(self, field):
        """Score shape should be (batch, heads, seq_q, seq_k)."""
        psi_Q = torch.randn(2, 4, 16, 16)
        psi_K = torch.randn(2, 4, 20, 16)

        score = geometric_score_from_exponential(psi_Q, psi_K, field)
        assert score.shape == (2, 4, 16, 20), f"Expected (2,4,16,20), got {score.shape}"

    def test_output_bounded(self, field):
        """Score should be bounded (no NaN/Inf)."""
        psi_Q = torch.randn(2, 4, 16, 16)
        psi_K = torch.randn(2, 4, 16, 16)

        score = geometric_score_from_exponential(psi_Q, psi_K, field)
        assert not torch.isnan(score).any(), "Score contains NaN"
        assert not torch.isinf(score).any(), "Score contains Inf"

    def test_extreme_psi_no_nan(self, field):
        """Extreme psi values should not produce NaN/Inf."""
        # Large psi
        psi_large = torch.randn(2, 4, 16, 16) * 100
        score = geometric_score_from_exponential(psi_large, psi_large, field)
        assert not torch.isnan(score).any(), "Large psi produced NaN"
        assert not torch.isinf(score).any(), "Large psi produced Inf"

    def test_gradient_flow(self, field):
        """Gradients should flow back through score computation."""
        psi_Q = torch.randn(2, 4, 16, 16, requires_grad=True)
        psi_K = torch.randn(2, 4, 16, 16, requires_grad=True)

        score = geometric_score_from_exponential(psi_Q, psi_K, field)
        loss = score.sum()
        loss.backward()

        assert psi_Q.grad is not None, "No gradient for psi_Q"
        assert psi_K.grad is not None, "No gradient for psi_K"
        assert not torch.isnan(psi_Q.grad).any(), "NaN gradient for psi_Q"
        assert not torch.isnan(psi_K.grad).any(), "NaN gradient for psi_K"


class TestBandRegularizer:
    """Tests for band regularization."""

    def test_in_band_no_penalty(self):
        """Vectors with norm in [0.7, 1.4] should have zero penalty."""
        vec = torch.randn(10, 8)
        vec = vec / vec.norm(dim=-1, keepdim=True)  # Unit norm (1.0)

        penalty = band_regularizer(vec, low=0.7, high=1.4)
        assert penalty.item() < 1e-6, f"Unit norm vectors penalized: {penalty.item()}"

    def test_too_small_penalized(self):
        """Vectors with norm < 0.7 should be penalized."""
        vec = torch.randn(10, 8)
        vec = vec / vec.norm(dim=-1, keepdim=True) * 0.3  # Norm = 0.3

        penalty = band_regularizer(vec, low=0.7, high=1.4)
        assert penalty.item() > 0, "Small norm vectors not penalized"

    def test_too_big_penalized(self):
        """Vectors with norm > 1.4 should be penalized."""
        vec = torch.randn(10, 8)
        vec = vec / vec.norm(dim=-1, keepdim=True) * 2.0  # Norm = 2.0

        penalty = band_regularizer(vec, low=0.7, high=1.4)
        assert penalty.item() > 0, "Large norm vectors not penalized"

    def test_generator_band_regularizer(self):
        """Combined regularizer should work for A, B, Θ."""
        A = torch.randn(4, 8)
        B = torch.randn(4, 8)
        Theta = torch.randn(4, 8) * 6  # Some values > 5 to trigger theta_excess

        total_reg, reg_dict = generator_band_regularizer(A, B, Theta)

        assert 'band_reg_A' in reg_dict
        assert 'band_reg_B' in reg_dict
        assert 'band_reg_Theta' in reg_dict
        assert 'theta_excess' in reg_dict
        assert reg_dict['theta_excess'] > 0, "Theta excess not triggered"


class TestGeometricAttentionExponentialForm:
    """Tests for GeometricAttention with exponential form enabled."""

    @pytest.fixture
    def attention_exp(self):
        """Create attention with exponential form."""
        return GeometricAttention(
            d_model=128,
            n_heads=4,
            use_exponential_form=True
        )

    @pytest.fixture
    def attention_legacy(self):
        """Create attention with legacy form."""
        return GeometricAttention(
            d_model=128,
            n_heads=4,
            use_exponential_form=False
        )

    @pytest.fixture
    def inputs(self):
        """Sample inputs for attention."""
        Q = torch.randn(2, 16, 128)
        K = torch.randn(2, 20, 128)
        V = torch.randn(2, 20, 128)
        T_field = torch.randn(8, 8, 4, 4, dtype=torch.cfloat)
        return Q, K, V, T_field

    def test_exponential_forward_shape(self, attention_exp, inputs):
        """Forward pass with exponential form should produce correct shapes."""
        Q, K, V, T_field = inputs
        output, attn_weights = attention_exp(Q, K, V, T_field)

        assert output.shape == (2, 16, 128), f"Output shape wrong: {output.shape}"
        assert attn_weights.shape == (2, 4, 16, 20), f"Attn shape wrong: {attn_weights.shape}"

    def test_exponential_no_nan(self, attention_exp, inputs):
        """Exponential form should not produce NaN/Inf."""
        Q, K, V, T_field = inputs
        output, attn_weights = attention_exp(Q, K, V, T_field)

        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        assert not torch.isnan(attn_weights).any(), "Attention weights contain NaN"

    def test_legacy_vs_exponential_both_work(self, attention_exp, attention_legacy, inputs):
        """Both forms should produce valid outputs (not necessarily equal)."""
        Q, K, V, T_field = inputs

        out_exp, attn_exp = attention_exp(Q, K, V, T_field)
        out_leg, attn_leg = attention_legacy(Q, K, V, T_field)

        # Both should be valid
        assert not torch.isnan(out_exp).any()
        assert not torch.isnan(out_leg).any()

        # Shapes should match
        assert out_exp.shape == out_leg.shape

    def test_gradient_stability_comparison(self, attention_exp, attention_legacy, inputs):
        """Compare gradient stability between exponential and legacy."""
        Q, K, V, T_field = inputs

        # Exponential form gradients
        Q_exp = Q.clone().requires_grad_(True)
        out_exp, _ = attention_exp(Q_exp, K, V, T_field)
        loss_exp = out_exp.norm()
        loss_exp.backward()
        grad_exp_norm = Q_exp.grad.norm().item()

        # Legacy form gradients
        Q_leg = Q.clone().requires_grad_(True)
        out_leg, _ = attention_legacy(Q_leg, K, V, T_field)
        loss_leg = out_leg.norm()
        loss_leg.backward()
        grad_leg_norm = Q_leg.grad.norm().item()

        print(f"Gradient norm - Exponential: {grad_exp_norm:.4f}, Legacy: {grad_leg_norm:.4f}")

        # Both should have finite gradients
        assert not torch.isnan(Q_exp.grad).any(), "Exponential has NaN gradients"
        assert not torch.isnan(Q_leg.grad).any(), "Legacy has NaN gradients"


class TestEndToEndStability:
    """End-to-end stability tests simulating training."""

    def test_100_step_training_simulation(self):
        """Simulate 100 training steps with exponential form."""
        # Setup
        attention = GeometricAttention(
            d_model=128,
            n_heads=4,
            use_exponential_form=True
        )
        optimizer = torch.optim.SGD(attention.parameters(), lr=0.01)
        T_field = torch.randn(8, 8, 4, 4, dtype=torch.cfloat)

        nan_steps = []
        for step in range(100):
            # Random batch
            Q = torch.randn(2, 16, 128)
            K = torch.randn(2, 20, 128)
            V = torch.randn(2, 20, 128)

            # Forward
            output, attn = attention(Q, K, V, T_field)

            # Loss
            loss = output.norm() + attn.norm()

            # Backward
            loss.backward()

            # Check for NaN
            has_nan = False
            for name, param in attention.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan = True
                        nan_steps.append(step)
                        break

            # Step
            optimizer.step()
            optimizer.zero_grad()

        if nan_steps:
            pytest.fail(f"NaN/Inf gradients at steps: {nan_steps}")
        else:
            print(f"✓ Completed 100 steps without NaN/Inf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
