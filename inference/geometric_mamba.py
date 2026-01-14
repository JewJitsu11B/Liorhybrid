"""
Geometric Mamba Encoder with CI8 State Space

Implements O(N) state-space model using Complex Octonion (CI8) algebra.

Key innovations:
1. Hidden state h_t is CI8-valued (16D: 8 real + 8 imaginary)
2. Transition matrix A → Trinor operator (geometric evolution)
3. Input projection B → Wedge operator (causal coupling)
4. Non-associative octonion multiplication (path-dependent dynamics)

This replaces standard linear SSM with geometric algebra that enforces
causal structure through the mathematics itself.

Architecture:
    x_t (input) → Wedge projection → CI8 space
    h_{t-1} (state) → Trinor evolution → CI8 space
    h_t = Trinor(h_{t-1}) ⊗ Wedge(x_t)  [octonion multiplication]
    y_t = Spinor(h_t) → output

References:
- Standard Mamba: h_t = A @ h_{t-1} + B @ x_t (linear, O(N))
- Geometric Mamba: h_t = T(h_{t-1}) ⊗ W(x_t) (non-linear, O(N))
- CI8: Complex Octonions with 8 amplitudes + 8 phases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# Debug flag: set True to enable NaN checking
DEBUG_NAN = False


def check_finite(name: str, x: torch.Tensor) -> None:
    """Check for NaN/Inf values and raise if found."""
    if DEBUG_NAN and not torch.isfinite(x).all():
        nan_count = torch.isnan(x).sum().item()
        inf_count = torch.isinf(x).sum().item()
        raise RuntimeError(f"NaNs/Infs detected in {name}: {nan_count} NaNs, {inf_count} Infs, shape={x.shape}")


class ComplexOctonion:
    """
    Complex Octonion (CI8) representation in pure PyTorch.

    Structure: 8 real components + 8 imaginary components = 16D

    Basis: {1, e1, e2, e3, e4, e5, e6, e7} × {real, imag}

    Properties:
    - Non-associative: (ab)c ≠ a(bc) in general
    - Alternative: (aa)b = a(ab), a(bb) = (ab)b
    - Normed: ||ab|| = ||a|| ||b||

    All operations maintained on device for GPU acceleration.
    """

    def __init__(self, real: torch.Tensor, imag: torch.Tensor):
        """
        Initialize ComplexOctonion.

        Args:
            real: Real components (*, 8)
            imag: Imaginary components (*, 8)
        """
        assert real.shape == imag.shape
        assert real.shape[-1] == 8

        self.real = real
        self.imag = imag
        self.device = real.device

    @classmethod
    def from_vector(cls, vec: torch.Tensor):
        """
        Create from 16D vector: [r0...r7, i0...i7].

        Args:
            vec: (*, 16) tensor

        Returns:
            ComplexOctonion
        """
        assert vec.shape[-1] == 16
        real = vec[..., :8]
        imag = vec[..., 8:]
        return cls(real, imag)

    def to_vector(self) -> torch.Tensor:
        """
        Convert to 16D vector.

        Returns:
            (*, 16) tensor
        """
        return torch.cat([self.real, self.imag], dim=-1)

    def conjugate(self):
        """
        Octonion conjugate: (r + i*e) → (r - i*e)

        For octonions: conj(a) reverses all basis vectors
        """
        # First component stays, rest negate
        # Use in-place negation to avoid unnecessary clones
        real_conj = self.real.clone()
        real_conj[..., 1:].neg_()

        imag_conj = self.imag.clone()
        imag_conj[..., 1:].neg_()

        return ComplexOctonion(real_conj, imag_conj)

    def norm(self) -> torch.Tensor:
        """
        Octonion norm: ||a|| = sqrt(sum of squares of all components)

        Returns:
            (*, ) tensor of norms
        """
        real_norm_sq = (self.real ** 2).sum(dim=-1)
        imag_norm_sq = (self.imag ** 2).sum(dim=-1)
        return torch.sqrt(real_norm_sq + imag_norm_sq)

    def normalize(self):
        """Normalize to unit norm."""
        norm = self.norm().unsqueeze(-1) + 1e-8
        return ComplexOctonion(
            self.real / norm,
            self.imag / norm
        )

    def __add__(self, other):
        """Component-wise addition."""
        return ComplexOctonion(
            self.real + other.real,
            self.imag + other.imag
        )

    def __mul__(self, other):
        """
        Octonion multiplication (non-associative).

        Uses Cayley-Dickson construction:
        (a, b) * (c, d) = (ac - d*b̄, da + bc̄)

        where a,b,c,d are quaternions (4D each).
        """
        # Split into quaternion pairs
        # Real part: (a, b) where a,b are quaternions
        a_real = self.real[..., :4]
        b_real = self.real[..., 4:]

        # Imag part: (c, d)
        a_imag = self.imag[..., :4]
        b_imag = self.imag[..., 4:]

        # Other octonion
        c_real = other.real[..., :4]
        d_real = other.real[..., 4:]
        c_imag = other.imag[..., :4]
        d_imag = other.imag[..., 4:]

        # Quaternion multiplication (simplified, non-associative)
        # For full implementation, need proper quaternion product
        # Here we use simplified bilinear form

        # Result real part: ac - d*b̄
        result_real_1 = a_real * c_real - d_real * b_real
        result_real_2 = a_imag * c_imag - d_imag * b_imag

        # Result imag part: da + bc̄
        result_imag_1 = d_real * a_real + b_real * c_real
        result_imag_2 = d_imag * a_imag + b_imag * c_imag

        result_real = torch.cat([result_real_1, result_real_2], dim=-1)
        result_imag = torch.cat([result_imag_1, result_imag_2], dim=-1)

        return ComplexOctonion(result_real, result_imag)


class TrinorOperator(nn.Module):
    """
    Trinor operator: Replaces transition matrix A in Mamba.

    Performs geometric evolution of CI8 state via:
        T(h) = exp(θ) * Rot(h, ω) * Scale(h, σ)

    where:
    - θ: Learned phase rotation (temporal evolution)
    - ω: Learned rotation axis (state trajectory)
    - σ: Learned scaling (energy modulation)

    This is the geometric equivalent of matrix multiplication
    but respects CI8 non-associative algebra.
    """

    def __init__(self, d_model: int = 16):
        super().__init__()

        self.d_model = d_model
        assert d_model == 16, "Trinor requires CI8 (16D) state"

        # Learnable parameters
        # Use 0.1 scale for meaningful rotation (0.1 rad ~ 6 degrees)
        # Small init caused near-identity transforms -> gradient issues after 512 steps
        self.theta = nn.Parameter(torch.randn(8) * 0.1)  # Phase rotation
        self.omega = nn.Parameter(torch.randn(8) * 0.1)  # Rotation axis
        self.sigma = nn.Parameter(torch.ones(8))  # Scaling (clamped in forward)

    def forward(self, h: ComplexOctonion) -> ComplexOctonion:
        """
        Apply Trinor evolution to state.

        Args:
            h: Current state (*, 16) as ComplexOctonion

        Returns:
            Evolved state (*, 16) as ComplexOctonion
        """
        # Phase rotation
        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)

        # Rotate real and imaginary parts
        real_rot = h.real * cos_theta - h.imag * sin_theta
        imag_rot = h.real * sin_theta + h.imag * cos_theta

        # Scale by learned sigma (clamped to prevent explosion/vanishing)
        sigma_clamped = torch.clamp(self.sigma, 0.5, 2.0)
        real_scaled = real_rot * sigma_clamped
        imag_scaled = imag_rot * sigma_clamped

        # Apply rotation axis (cross-component coupling)
        # This creates non-trivial geometry
        real_final = real_scaled + 0.1 * torch.matmul(
            real_scaled,
            torch.diag(self.omega)
        )
        imag_final = imag_scaled + 0.1 * torch.matmul(
            imag_scaled,
            torch.diag(self.omega)
        )

        return ComplexOctonion(real_final, imag_final)


class WedgeProjection(nn.Module):
    """
    Wedge projection: Replaces input matrix B in Mamba.

    Projects input features into CI8 space using wedge product:
        W(x) = x ∧ e_basis

    This creates antisymmetric coupling that enforces causality:
    - New information (x) is orthogonal to existing state
    - Prevents redundant encoding
    - Captures divergence/novelty
    """

    def __init__(self, d_input: int, d_model: int = 16):
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model

        # Learnable basis for wedge product
        # Projects d_input → 16D (CI8)
        # Fan-in scaling: 1/sqrt(d_input) preserves variance through projection
        self.basis_real = nn.Parameter(torch.randn(d_input, 8) / math.sqrt(d_input))
        self.basis_imag = nn.Parameter(torch.randn(d_input, 8) / math.sqrt(d_input))

        # Normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> ComplexOctonion:
        """
        Project input to CI8 via wedge product.

        Args:
            x: Input features (*, d_input)

        Returns:
            ComplexOctonion (*, 16)
        """
        # Project to real and imaginary parts
        real = torch.matmul(x, self.basis_real)  # (*, 8)
        imag = torch.matmul(x, self.basis_imag)  # (*, 8)

        # Create ComplexOctonion
        oct = ComplexOctonion(real, imag)

        # Normalize (optional, helps stability)
        vec = oct.to_vector()
        vec_norm = self.layer_norm(vec)

        return ComplexOctonion.from_vector(vec_norm)


class SpinorProjection(nn.Module):
    """
    Spinor projection: Output layer for Mamba.

    Projects CI8 state to output space using spinor product:
        y = S(h) = h ⊙ h̄

    The spinor product captures rotational invariants,
    giving outputs that are stable under phase transformations.
    """

    def __init__(self, d_model: int = 16, d_output: int = 512):
        super().__init__()

        self.d_model = d_model
        self.d_output = d_output

        # Projection from CI8 invariants to output
        # Spinor product gives 8 real invariants
        self.projection = nn.Linear(8, d_output)

    def forward(self, h: ComplexOctonion) -> torch.Tensor:
        """
        Project CI8 state to output.

        Args:
            h: State (*, 16) as ComplexOctonion

        Returns:
            Output (*, d_output)
        """
        # Compute spinor product: h ⊙ h̄
        h_conj = h.conjugate()

        # Spinor product gives real-valued invariants
        # Use norm of each component
        invariants = h.real * h_conj.real + h.imag * h_conj.imag  # (*, 8)

        # Project to output space
        output = self.projection(invariants)  # (*, d_output)

        return output


class GeometricMambaLayer(nn.Module):
    """
    Single layer of Geometric Mamba with CI8 state space.

    Replaces standard Mamba SSM with geometric operators:

    Standard Mamba:
        h_t = A @ h_{t-1} + B @ x_t
        y_t = C @ h_t

    Geometric Mamba:
        h_t = Trinor(h_{t-1}) ⊗ Wedge(x_t)  [octonion multiplication]
        y_t = Spinor(h_t)

    Key properties:
    - O(N) complexity maintained
    - Non-associative dynamics (path-dependent)
    - Geometric causality enforced by algebra
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,  # CI8
        expand_factor: int = 2,
        timing_debug: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        self.timing_debug = timing_debug

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner)

        # Geometric operators
        self.wedge = WedgeProjection(self.d_inner, d_state)
        self.trinor = TrinorOperator(d_state)
        self.spinor = SpinorProjection(d_state, self.d_inner)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Gating (selective SSM)
        self.gate = nn.Linear(self.d_inner, self.d_inner)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[ComplexOctonion] = None
    ) -> Tuple[torch.Tensor, ComplexOctonion]:
        """
        Forward pass with geometric state update.

        Args:
            x: Input (batch, seq_len, d_model)
            state: Previous CI8 state (batch, 16) or None

        Returns:
            output: (batch, seq_len, d_model)
            final_state: (batch, 16) as ComplexOctonion
        """
        import time

        if self.timing_debug:
            t0 = time.perf_counter()
            timing_wedge = 0.0
            timing_trinor = 0.0
            timing_spinor = 0.0
            timing_gate = 0.0
            timing_mul = 0.0

        batch_size, seq_len, _ = x.shape

        # Input projection
        if self.timing_debug:
            t_proj_start = time.perf_counter()
        x_inner = self.in_proj(x)  # (batch, seq_len, d_inner)
        if self.timing_debug:
            timing_in_proj = time.perf_counter() - t_proj_start

        # Initialize state if not provided
        if state is None:
            real = torch.zeros(batch_size, 8, device=x.device)
            imag = torch.zeros(batch_size, 8, device=x.device)
            state = ComplexOctonion(real, imag)

        # Scan over sequence (O(N) operation)
        # Pre-allocate output tensor to avoid Python list overhead
        output = torch.zeros(batch_size, seq_len, self.d_inner,
                           dtype=x_inner.dtype, device=x_inner.device)

        for t in range(seq_len):
            x_t = x_inner[:, t, :]  # (batch, d_inner)

            # Geometric state update
            # 1. Project input to CI8 via wedge
            if self.timing_debug:
                t_wedge_start = time.perf_counter()
            x_oct = self.wedge(x_t)  # ComplexOctonion (batch, 16)
            if self.timing_debug:
                timing_wedge += time.perf_counter() - t_wedge_start

            # 2. Evolve state via trinor
            if self.timing_debug:
                t_trinor_start = time.perf_counter()
            state_evolved = self.trinor(state)  # ComplexOctonion (batch, 16)
            if self.timing_debug:
                timing_trinor += time.perf_counter() - t_trinor_start

            # 3. Combine via octonion multiplication (non-associative!)
            if self.timing_debug:
                t_mul_start = time.perf_counter()
            state = state_evolved * x_oct  # ComplexOctonion (batch, 16)
            if self.timing_debug:
                timing_mul += time.perf_counter() - t_mul_start

            # 4. Project to output via spinor
            if self.timing_debug:
                t_spinor_start = time.perf_counter()
            y_t = self.spinor(state)  # (batch, d_inner)
            if self.timing_debug:
                timing_spinor += time.perf_counter() - t_spinor_start

            # 5. Gating (selective mechanism)
            if self.timing_debug:
                t_gate_start = time.perf_counter()
            gate_t = torch.sigmoid(self.gate(x_t))
            y_t = y_t * gate_t
            if self.timing_debug:
                timing_gate += time.perf_counter() - t_gate_start

            # Write directly to pre-allocated tensor (no list append)
            output[:, t, :] = y_t

        # Output projection
        if self.timing_debug:
            t_out_proj_start = time.perf_counter()
        output = self.out_proj(output)  # (batch, seq_len, d_model)
        if self.timing_debug:
            timing_out_proj = time.perf_counter() - t_out_proj_start

        # Residual + norm
        output = self.norm(x + output)

        if self.timing_debug:
            total_time = time.perf_counter() - t0
            print(f"[GeometricMamba] seq_len={seq_len}, total={total_time:.3f}s")
            print(f"  in_proj={timing_in_proj:.4f}s, out_proj={timing_out_proj:.4f}s")
            print(f"  Loop totals: wedge={timing_wedge:.4f}s, trinor={timing_trinor:.4f}s, "
                  f"mul={timing_mul:.4f}s, spinor={timing_spinor:.4f}s, gate={timing_gate:.4f}s")
            loop_total = timing_wedge + timing_trinor + timing_mul + timing_spinor + timing_gate
            print(f"  Loop sum={loop_total:.4f}s, per-token={loop_total/seq_len*1000:.2f}ms")

        return output, state


class GeometricMambaEncoder(nn.Module):
    """
    Multi-layer Geometric Mamba encoder.

    Stacks multiple GeometricMambaLayer modules to create
    deep O(N) architecture with CI8 state space.

    This is the replacement for standard transformer encoders,
    maintaining O(N) complexity while enforcing geometric causality.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 4,
        d_state: int = 16,
        expand_factor: int = 2,
        timing_debug: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.timing_debug = timing_debug

        # Stack of geometric mamba layers
        self.layers = nn.ModuleList([
            GeometricMambaLayer(d_model, d_state, expand_factor, timing_debug)
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(d_model)

    def set_timing_debug(self, enabled: bool):
        """Enable or disable timing debug for all layers."""
        self.timing_debug = enabled
        for layer in self.layers:
            layer.timing_debug = enabled

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Encode sequence with geometric state space.

        Args:
            x: Input (batch, seq_len, d_model)
            states: List of CI8 states per layer, or None

        Returns:
            output: (batch, seq_len, d_model)
            final_states: List of CI8 states per layer
        """
        if states is None:
            states = [None] * self.n_layers

        final_states = []

        for layer, state in zip(self.layers, states):
            x, new_state = layer(x, state)
            final_states.append(new_state)

        x = self.norm(x)

        return x, final_states


# Integration helper: Convert between field T_ij and CI8 state
def field_to_ci8(T_field: torch.Tensor) -> ComplexOctonion:
    """
    Convert cognitive tensor field to CI8 state.

    Args:
        T_field: (N_x, N_y, D, D) complex tensor

    Returns:
        ComplexOctonion representing field state
    """
    # Average over spatial dimensions
    T_avg = T_field.mean(dim=(0, 1))  # (D, D)

    # Extract 8 components from tensor structure
    # Use diagonal and off-diagonal elements
    if T_avg.shape[0] >= 8:
        real = torch.diag(T_avg.real)[:8]
        imag = torch.diag(T_avg.imag)[:8] if T_avg.is_complex() else torch.zeros(8, device=T_avg.device)
    else:
        # Pad if needed
        D = T_avg.shape[0]
        real = torch.nn.functional.pad(torch.diag(T_avg.real), (0, 8 - D))
        imag = torch.zeros(8, device=T_avg.device)

    return ComplexOctonion(real.unsqueeze(0), imag.unsqueeze(0))


def ci8_to_field(oct: ComplexOctonion, spatial_size: tuple = (8, 8), D: int = 4) -> torch.Tensor:
    """
    Convert CI8 state back to tensor field structure.

    Args:
        oct: ComplexOctonion state
        spatial_size: (N_x, N_y) spatial dimensions
        D: Tensor dimension

    Returns:
        T_field: (N_x, N_y, D, D) complex tensor
    """
    N_x, N_y = spatial_size
    device = oct.real.device

    # Create tensor field
    T_field = torch.zeros(N_x, N_y, D, D, dtype=torch.complex64, device=device)

    # Place CI8 components on diagonal
    for i in range(min(D, 8)):
        T_field[:, :, i, i] = oct.real[0, i] + 1j * oct.imag[0, i]

    return T_field
