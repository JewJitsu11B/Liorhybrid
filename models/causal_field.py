"""
Causal Field Layer: Parallel Field Evolution with Pi, Gamma, Phi Tensors

Implements the fundamental causal accumulation law:

    T^{mu nu}_{rho sigma}(x) = alpha J^{mu nu}_{rho sigma}(x)
        - (1-alpha) integral_{J^-(x)} k(tau) Pi^{mu nu}_{rho sigma||alpha beta}^{gamma delta}
                                           Gamma^gamma_delta(x,x') J^{alpha beta}_{gamma delta}(x') d^4x'

Where:
    T: Observable field (output)
    J: Source current (associator measuring non-associativity)
    k(tau): Retarded kernel (LIoR memory)
    Pi: Parallel transport tensor
    Gamma: Clifford connection
    Phi^[rho sigma]: Bivector field (indices RAISED)

The source current J is the associator:
    J^{mu nu}_{rho sigma}(x) = (psi_Sigma * psi_Lambda) * psi_alpha
                             - psi_Sigma * (psi_Lambda * psi_alpha)

This measures where the spinor algebra fails to be associative.

Holomorphic constraint:
    nabla^{(c D^alpha)}_mu (Pi Gamma Phi) = 0

Key properties:
    - O(N log N) via FFT convolution (not O(N) sequential)
    - O(1) memory update via finite-pole recurrence
    - Complex metric G = A + iB with phase orthogonality
    - No sequential loops over seq_len
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from .manifold import CognitiveManifold
from .lior_kernel import LiorKernel, LiorMemoryState
from .biquaternion import BiQuatCausalLayer
from Liorhybrid.training.execution_tracker import track_first_call


class AssociatorCurrent(nn.Module):
    """
    Complex octonion associator in 16-d real representation.
    Uses fixed Fano-plane structure constants (non-learnable).

    J^{mu nu}_{rho sigma}(x) = (psi_Sigma * psi_Lambda) * psi_alpha
                             - psi_Sigma * (psi_Lambda * psi_alpha)

    This measures the non-associativity of the complex octonion algebra,
    which is genuinely non-associative (unlike learned bilinear forms).
    """

    def __init__(self, d_model: int, d_field: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_field = d_field  # Must be 16 for complex octonions
        self.d_oct = 8  # Real octonion dimension

        # Project to three spinor factors (16-d each)
        self.sigma_proj = nn.Linear(d_model, d_field)
        self.lambda_proj = nn.Linear(d_model, d_field)
        self.alpha_proj = nn.Linear(d_model, d_field)

        # Fixed octonion structure constants (Fano-plane)
        f = torch.zeros(8, 8, 8)
        triples = [
            (0, 1, 2), (0, 3, 4), (0, 5, 6),
            (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)
        ]
        for i, j, k in triples:
            f[i, j, k] = 1.0
            f[j, k, i] = 1.0
            f[k, i, j] = 1.0
            f[j, i, k] = -1.0
            f[k, j, i] = -1.0
            f[i, k, j] = -1.0
        self.register_buffer('oct_struct', f)  # Fixed, not learnable

        # Learnable expansion to rank-2 tensor
        self.J_expand = nn.Parameter(torch.randn(d_field, d_field, d_field) * 0.1)

    def oct_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Real octonion product using structure constants."""
        return torch.einsum('...j,...k,jki->...i', a, b, self.oct_struct)

    def complex_oct_mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Complex octonion product in 16-d real representation.
        x, y: [..., 16] where [:8] = real, [8:] = imag
        Returns: [..., 16]
        """
        a, b = x[..., :8], x[..., 8:]  # a + ib
        c, d = y[..., :8], y[..., 8:]  # c + id

        # (a + ib)(c + id) = (ac - bd) + i(ad + bc)
        real = self.oct_mul(a, c) - self.oct_mul(b, d)
        imag = self.oct_mul(a, d) + self.oct_mul(b, c)

        return torch.cat([real, imag], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute associator current J = (ab)c - a(bc).

        Args:
            x: Input [B, N, d_model]

        Returns:
            J: Associator [B, N, d_field, d_field] antisymmetric tensor
        """
        psi_sigma = self.sigma_proj(x)   # [B, N, 16]
        psi_lambda = self.lambda_proj(x)  # [B, N, 16]
        psi_alpha = self.alpha_proj(x)    # [B, N, 16]

        # (sigma * lambda) * alpha
        prod_sl = self.complex_oct_mul(psi_sigma, psi_lambda)
        left_assoc = self.complex_oct_mul(prod_sl, psi_alpha)

        # sigma * (lambda * alpha)
        prod_la = self.complex_oct_mul(psi_lambda, psi_alpha)
        right_assoc = self.complex_oct_mul(psi_sigma, prod_la)

        # Associator J = left - right (non-zero for non-associative algebra)
        J_vector = left_assoc - right_assoc  # [B, N, 16]

        # Expand to rank-2 tensor via learnable projection
        J_tensor = torch.einsum('...i,ijk->...jk', J_vector, self.J_expand)
        J_tensor = J_tensor - J_tensor.transpose(-1, -2)  # Antisymmetric

        return J_tensor


class ParallelTransport(nn.Module):
    """
    Parallel transport tensor Pi^{mu nu}_{rho sigma||alpha beta}^{gamma delta}.

    This encodes how information moves between tangent and dual spaces
    while preserving direction and phase coherence.

    Index structure:
        (mu nu): Target indices (where derivative acts)
        [rho sigma]: Source bivector indices (from Phi)
        ||(alpha beta): Fractional causal memory channel
        ^{gamma delta}: Spinor channel (Clifford action coupling)
    """

    def __init__(self, d_field: int = 16, d_spinor: int = 4):
        super().__init__()
        self.d_field = d_field
        self.d_spinor = d_spinor

        # Decompose the rank-8 tensor into manageable factors
        # Pi = Pi_target @ Pi_source @ Pi_memory @ Pi_spinor

        # Target indices (mu, nu) -> symmetric
        self.Pi_target = nn.Parameter(
            torch.randn(d_field, d_field, d_field) / d_field
        )

        # Source indices [rho, sigma] -> antisymmetric bivector
        self.Pi_source = nn.Parameter(
            torch.randn(d_field, d_field, d_field) / d_field
        )

        # Memory channel (alpha, beta)
        self.Pi_memory = nn.Parameter(
            torch.randn(d_field, d_field, d_field) / d_field
        )

        # Spinor channel (gamma, delta)
        self.Pi_spinor = nn.Parameter(
            torch.randn(d_spinor, d_spinor, d_field) / d_spinor
        )

    def forward(
        self,
        J: torch.Tensor,
        Gamma: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply parallel transport to source current.

        Pi^{...}^{gamma delta} Gamma^gamma_delta J^{...}

        Args:
            J: Source current [B, N, d_field, d_field]
            Gamma: Clifford connection [d_spinor, d_spinor]

        Returns:
            Transported field [B, N, d_field, d_field]
        """
        # Contract J with source factor
        # J @ Pi_source -> intermediate
        inter1 = torch.einsum('...ij,ijk->...k', J, self.Pi_source)  # [B, N, d_field]

        # Apply target structure
        inter2 = torch.einsum('...k,mnk->...mn', inter1, self.Pi_target)  # [B, N, d_f, d_f]

        # Apply Clifford connection via spinor factor
        # Pi_spinor @ Gamma gives spinor contribution
        spinor_contrib = torch.einsum('gdk,gd->k', self.Pi_spinor, Gamma)  # [d_field]

        # Modulate by spinor contribution
        output = inter2 * spinor_contrib.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return output


class CliffordConnection(nn.Module):
    """
    Clifford connection Gamma^gamma_delta.

    This operator performs the local Clifford action on spinor indices.
    It absorbs the tetrad (vierbein) and metric structure:

        Gamma^gamma_delta = e^lambda_a (gamma^a)^gamma_delta

    where gamma^a are the Clifford algebra generators.

    The connection swaps spinor index orientation, ensuring the product
    transforms covariantly under parallel transport.
    """

    def __init__(self, d_spinor: int = 4):
        super().__init__()
        self.d_spinor = d_spinor

        # Learnable Clifford generators (4 for Dirac)
        self.gamma_matrices = nn.Parameter(
            torch.randn(4, d_spinor, d_spinor) / d_spinor
        )

        # Tetrad / vierbein
        self.tetrad = nn.Parameter(
            torch.eye(4) + 0.1 * torch.randn(4, 4)
        )

    def forward(self) -> torch.Tensor:
        """
        Compute Clifford connection.

        Returns:
            Gamma^gamma_delta [d_spinor, d_spinor]
        """
        # Contract tetrad with gamma matrices
        # Gamma = sum_a tetrad[a, :] @ gamma_a
        Gamma = torch.einsum('ab,bcd->acd', self.tetrad, self.gamma_matrices)

        # Sum over spacetime index to get single connection
        Gamma = Gamma.sum(dim=0)  # [d_spinor, d_spinor]

        return Gamma


class CausalFieldLayer(nn.Module):
    """
    Causal field layer implementing the fundamental accumulation law.

    T = alpha * J - (1-alpha) * integral k(tau) Pi Gamma J d^4x'

    Replaces sequential Mamba scan with parallel FFT convolution.
    """

    def __init__(
        self,
        d_model: int,
        d_field: int = 16,
        d_spinor: int = 4,
        kernel_size: int = 64
    ):
        super().__init__()

        self.d_model = d_model
        self.d_field = d_field
        self.d_spinor = d_spinor
        self.kernel_size = kernel_size

        # === Core tensors ===
        # Phi^[rho sigma] - bivector field (RAISED indices)
        self.Phi = nn.Parameter(
            torch.randn(d_field, d_field) / d_field
        )
        # Make antisymmetric
        with torch.no_grad():
            self.Phi.data = self.Phi.data - self.Phi.data.T

        # Parallel transport
        self.Pi = ParallelTransport(d_field, d_spinor)

        # Clifford connection
        self.Gamma_conn = CliffordConnection(d_spinor)

        # === Source current ===
        self.associator = AssociatorCurrent(d_model, d_field)

        # === LIoR memory ===
        # Use d_field^2 since that's the dimension of transported_flat
        self.memory = LiorMemoryState(d_field * d_field)

        # === Manifold ===
        self.manifold = CognitiveManifold(d_model, d_field, d_spinor)

        # === Projections ===
        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_field * d_field, d_model)

        # === Normalization ===
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[Dict[str, torch.Tensor]] = None,
        diagnose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with causal field evolution.

        Args:
            x: Input [B, N, d_model]
            memory: Previous memory state
            diagnose: If True, print diagnostic info about NaN/Inf

        Returns:
            output: [B, N, d_model]
            new_memory: Updated memory state
        """
        B, N, D = x.shape
        residual = x

        def check_tensor(name: str, t: torch.Tensor) -> bool:
            """Check tensor for NaN/Inf and report stats."""
            if not diagnose:
                return False
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            if has_nan or has_inf:
                print(f"  [NaN DETECTED] {name}: nan={has_nan}, inf={has_inf}, "
                      f"shape={list(t.shape)}, min={t[~torch.isnan(t)].min().item() if not t.isnan().all() else 'all_nan':.4g}, "
                      f"max={t[~torch.isnan(t)].max().item() if not t.isnan().all() else 'all_nan':.4g}", flush=True)
                return True
            else:
                print(f"  [OK] {name}: shape={list(t.shape)}, "
                      f"min={t.min().item():.4g}, max={t.max().item():.4g}, "
                      f"mean={t.mean().item():.4g}, std={t.std().item():.4g}", flush=True)
            return False

        if diagnose:
            print("\n" + "="*60, flush=True)
            print("CausalFieldLayer DIAGNOSTIC", flush=True)
            print("="*60, flush=True)

        check_tensor("input_x", x)

        # === Project input ===
        x = self.input_proj(x)
        check_tensor("after_input_proj", x)

        # === Compute source current J (associator) ===
        J = self.associator(x)  # [B, N, d_field, d_field]
        check_tensor("J_associator", J)

        # === Get Clifford connection ===
        Gamma = self.Gamma_conn()  # [d_spinor, d_spinor]
        check_tensor("Gamma_connection", Gamma)

        # === Apply parallel transport Pi Gamma J ===
        transported = self.Pi(J, Gamma)  # [B, N, d_field, d_field]
        check_tensor("transported", transported)

        # === Memory integration via LIoR kernel ===
        # Flatten for convolution
        transported_flat = transported.view(B, N, -1)  # [B, N, d_field^2]
        check_tensor("transported_flat", transported_flat)

        # Apply O(1) memory update
        memory_out, new_memory = self.memory(transported_flat, memory, diagnose=diagnose)
        check_tensor("memory_out", memory_out)

        # === Combine: T = alpha * J + (1-alpha) * memory_term ===
        alpha = self.memory.kernel.weights[0]  # Instantaneous weight
        if diagnose:
            print(f"  [INFO] alpha (kernel weight[0]) = {alpha.item():.4g}", flush=True)

        J_flat = J.view(B, N, -1)
        T_flat = alpha * J_flat + (1 - alpha) * memory_out
        check_tensor("T_flat_combined", T_flat)

        # === Project to output ===
        output = self.output_proj(T_flat)
        check_tensor("after_output_proj", output)

        # === Residual + norm ===
        output = self.norm(residual + output)
        check_tensor("final_output", output)

        if diagnose:
            print("="*60 + "\n", flush=True)

        return output, new_memory

    def get_phi(self) -> torch.Tensor:
        """Get current Phi^[rho sigma] (antisymmetric)."""
        return 0.5 * (self.Phi - self.Phi.T)


class CausalFieldBlock(nn.Module):
    """
    Complete causal field block as drop-in replacement for GeometricMambaLayer.

    Structure:
    1. CausalFieldLayer (field evolution with Pi, Gamma, Phi)
    2. Optional attention over normal coordinates
    3. FFN
    4. Fully parallel (no sequential loops)
    """

    def __init__(
        self,
        d_model: int,
        d_field: int = 16,
        d_spinor: int = 4,
        kernel_size: int = 64,
        n_heads: int = 8,
        expand_factor: float = None,
        use_attention: bool = True,
        ffn_activation: str = 'swiglu',
        dropout: float = 0.0
    ):
        super().__init__()

        self.d_model = d_model
        self.use_attention = use_attention

        # === Causal field layer ===
        self.field_layer = CausalFieldLayer(
            d_model, d_field, d_spinor, kernel_size
        )

        # === Optional attention over normal coordinates ===
        if use_attention:
            self.attention = nn.MultiheadAttention(
                d_model, n_heads, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(d_model)

        # === FFN (SwiGLU default) ===
        from .activations import FFN
        self.ffn = FFN(
            d_model=d_model,
            expansion_factor=expand_factor,
            activation=ffn_activation,
            dropout=dropout
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        diagnose: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input [B, N, d_model]
            memory: Previous memory state
            attention_mask: Optional attention mask
            diagnose: If True, print diagnostic info

        Returns:
            output: [B, N, d_model]
            new_memory: Updated memory state
        """
        # === Causal field evolution ===
        x, new_memory = self.field_layer(x, memory, diagnose=diagnose)

        # === Optional attention ===
        if self.use_attention:
            residual = x
            attn_out, _ = self.attention(x, x, x, attn_mask=attention_mask)
            x = self.attn_norm(residual + attn_out)

        # === FFN ===
        residual = x
        x = self.ffn_norm(residual + self.ffn(x))

        return x, new_memory


class BiQuatCausalBlock(nn.Module):
    """
    Causal field block using biquaternion algebra (PURE REAL, no torch.complex).

    Replaces cubic O(d^3) octonionic ops with O(N) structured quaternion matmuls.
    Drop-in replacement for CausalFieldBlock.

    Structure:
    1. BiQuatCausalLayer (field evolution with quaternionic transforms)
    2. Optional attention
    3. FFN
    """

    def __init__(
        self,
        d_model: int,
        d_field: int = 16,
        n_heads: int = 8,
        use_attention: bool = True,
        alpha: float = 0.5,
        detach_memory: bool = True,
        bptt_window: int = 0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.d_model = d_model
        self.d_field = d_field
        self.use_attention = use_attention

        # Biquaternion causal layer (O(N) pure real, no torch.complex)
        self.field_layer = BiQuatCausalLayer(
            d_model=d_model,
            d_field=d_field,
            alpha=alpha,
            detach_memory=detach_memory,
            bptt_window=bptt_window
        )

        # Optional attention
        if use_attention:
            # Wrap MultiheadAttention with first-call tracker
            class TrackedMultiheadAttention(nn.MultiheadAttention):
                @track_first_call
                def forward(self, *args, **kwargs):
                    return super().forward(*args, **kwargs)

            self.attention = TrackedMultiheadAttention(
                d_model, n_heads, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(d_model)

    @track_first_call
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        diagnose: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input [B, N, d_model]
            memory: Previous memory state
            attention_mask: Optional attention mask
            diagnose: If True, print diagnostic info

        Returns:
            output: [B, N, d_model]
            new_memory: Updated memory state
        """
        if diagnose:
            print(f"\n{'='*60}")
            print(f"[BiQuatCausalBlock] START")
            print(f"{'='*60}")

        # Biquaternion causal field evolution
        x, new_memory = self.field_layer(x, memory, diagnose=diagnose)

        if diagnose:
            print(f"[BiQuatCausalBlock] After field_layer: nan={x.isnan().any().item()}, "
                  f"inf={x.isinf().any().item()}")

        # Optional attention
        if self.use_attention:
            residual = x
            attn_out, _ = self.attention(x, x, x, attn_mask=attention_mask)
            x = self.attn_norm(residual + attn_out)
            if diagnose:
                print(f"[BiQuatCausalBlock] After attention: nan={x.isnan().any().item()}, "
                      f"inf={x.isinf().any().item()}")

        if diagnose:
            print(f"[BiQuatCausalBlock] END")
            print(f"{'='*60}\n")

        return x, new_memory
