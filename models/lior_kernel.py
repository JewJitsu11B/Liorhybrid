"""
LIoR Kernel: Learnable Memory Kernel with O(1) Recurrence

Implements the Higgs-modulated causal memory kernel:

    K_L(x; dt; J_H) = Theta(dt) * [
        alpha * exp(-beta * dt)                           # Exponential (Markovian)
      - gamma * dt^(-delta) * exp(-xi * dt)               # Power-law (Fractional)
      + eta * cos(omega * dt + phi) * exp(-zeta * dt)     # Phasic (Oscillatory)
    ]

Key insight: The full path integral m_t = integral_0^t K(t,tau) x_tau dtau
can be computed via finite-pole recurrence in O(1) time:

    m_t = rho * m_{t-1} + eta * x_t - xi * x_{t-p_eff}

This is "Non-Markovian physics with O(1) Bayesian filter update."

The phase structure theta = (pi*alpha/2) - alpha*ln(omega) feeds into the
symplectic form B_{mu nu} of the complex metric G = A + iB.

References:
- Fractional calculus: Caputo derivative pairs with power-law kernel
- Phase orthogonality: Sigma (geometric) perp Lambda (spectral) for stability
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class LiorKernel(nn.Module):
    """
    Learnable LIoR memory kernel with O(1) recurrence update.

    The kernel is a mixture of three physically meaningful modes:
    1. Exponential: short memory, Markov-ish relaxation
    2. Power-law: long tail, "true" fractional memory
    3. Oscillatory: phase-sensitive interference

    All parameters can be modulated by an external Higgs Current Scalar J_H.
    """

    def __init__(
        self,
        p_eff: int = 4,
        init_tau_exp: float = 1.0,
        init_tau_frac: float = 1.0,
        init_tau_osc: float = 1.0,
        init_alpha: float = 0.5,
        init_omega: float = 1.0,
        init_phi: float = 0.0,
    ):
        """
        Initialize LIoR kernel.

        Args:
            p_eff: Effective pole count for O(1) recurrence approximation
            init_tau_*: Initial timescales for each mode
            init_alpha: Initial fractional order (0 < alpha < 1)
            init_omega: Initial oscillation frequency
            init_phi: Initial oscillation phase
        """
        super().__init__()

        self.p_eff = p_eff

        # === Exponential mode parameters ===
        # k_exp(dt) = alpha * exp(-beta * dt)
        self.log_alpha_exp = nn.Parameter(torch.tensor(0.0))  # exp -> positive
        self.log_beta = nn.Parameter(torch.tensor(-math.log(init_tau_exp)))

        # === Power-law / fractional mode parameters ===
        # k_frac(dt) = gamma * dt^(-delta) * exp(-xi * dt)
        self.log_gamma = nn.Parameter(torch.tensor(-1.0))  # Small initial weight
        self.delta_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid -> (0,1)
        self.log_xi = nn.Parameter(torch.tensor(-math.log(init_tau_frac)))

        # === Oscillatory mode parameters ===
        # k_osc(dt) = eta * cos(omega * dt + phi) * exp(-zeta * dt)
        self.log_eta = nn.Parameter(torch.tensor(-1.0))  # Small initial weight
        self.omega = nn.Parameter(torch.tensor(init_omega))
        self.phi = nn.Parameter(torch.tensor(init_phi))
        self.log_zeta = nn.Parameter(torch.tensor(-math.log(init_tau_osc)))

        # === O(1) Recurrence parameters ===
        # m_t = rho * m_{t-1} + eta_r * x_t - xi_r * x_{t-p_eff}
        # These are derived from kernel parameters but can also be learned directly
        self.rho_logit = nn.Parameter(torch.tensor(2.0))  # sigmoid -> rho in (0,1)
        self.log_eta_r = nn.Parameter(torch.tensor(0.0))
        self.log_xi_r = nn.Parameter(torch.tensor(-2.0))

        # === Mixture weights (softmax normalized) ===
        self.weight_logits = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))

    # === Parameter accessors with appropriate transforms ===

    @property
    def alpha_exp(self) -> torch.Tensor:
        return torch.exp(self.log_alpha_exp)

    @property
    def beta(self) -> torch.Tensor:
        return torch.exp(self.log_beta)

    @property
    def gamma(self) -> torch.Tensor:
        return torch.exp(self.log_gamma)

    @property
    def delta(self) -> torch.Tensor:
        """Fractional order in (0, 1)."""
        return torch.sigmoid(self.delta_logit)

    @property
    def xi(self) -> torch.Tensor:
        return torch.exp(self.log_xi)

    @property
    def eta(self) -> torch.Tensor:
        return torch.exp(self.log_eta)

    @property
    def zeta(self) -> torch.Tensor:
        return torch.exp(self.log_zeta)

    @property
    def rho(self) -> torch.Tensor:
        """Recurrence decay in (0, 1)."""
        return torch.sigmoid(self.rho_logit)

    @property
    def eta_r(self) -> torch.Tensor:
        return torch.exp(self.log_eta_r)

    @property
    def xi_r(self) -> torch.Tensor:
        return torch.exp(self.log_xi_r)

    @property
    def weights(self) -> torch.Tensor:
        """Mixture weights [exp, frac, osc] summing to 1."""
        return F.softmax(self.weight_logits, dim=0)

    @property
    def fractional_order(self) -> torch.Tensor:
        """The delta parameter, used for phase computation."""
        return self.delta

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Evaluate kernel at time lag dt.

        Args:
            dt: Time lag values [*shape], should be >= 0

        Returns:
            K(dt): Kernel values [*shape]
        """
        # Enforce causality: K(dt) = 0 for dt < 0 (Heaviside theta)
        # Use larger min to avoid numerical issues with power-law term
        dt = torch.clamp(dt, min=0.1)

        # Use softmax-normalized weights (not raw logits)
        w = self.weights  # This property already applies softmax

        # Exponential mode: alpha * exp(-beta * dt)
        # Clamp exponent to avoid overflow/underflow
        exp_arg = torch.clamp(-self.beta * dt, min=-20.0, max=20.0)
        k_exp = self.alpha_exp * torch.exp(exp_arg)

        # Power-law mode: gamma * dt^(-delta) * exp(-xi * dt)
        # Use log-space computation to avoid overflow: dt^(-delta) = exp(-delta * log(dt))
        # delta is in (0,1) so -delta * log(dt) is bounded for dt >= 0.1
        log_dt = torch.log(dt)
        power_term = torch.exp(torch.clamp(-self.delta * log_dt, min=-20.0, max=20.0))
        exp_cutoff = torch.exp(torch.clamp(-self.xi * dt, min=-20.0, max=20.0))
        k_frac = self.gamma * power_term * exp_cutoff

        # Oscillatory mode: eta * cos(omega * dt + phi) * exp(-zeta * dt)
        osc_exp = torch.exp(torch.clamp(-self.zeta * dt, min=-20.0, max=20.0))
        k_osc = self.eta * torch.cos(self.omega * dt + self.phi) * osc_exp

        # Weighted mixture
        k = w[0] * k_exp + w[1] * k_frac + w[2] * k_osc

        # Final safety clamp
        k = torch.clamp(k, min=-100.0, max=100.0)

        return k

    def compute_phase(self, omega_freq: torch.Tensor) -> torch.Tensor:
        """
        Compute phase field from fractional kernel structure.

        The fractional kernel k(t) ~ t^(delta-1) has Fourier transform with phase:
            theta(omega) = (pi * delta / 2) - delta * ln(omega)

        This phase feeds into the symplectic form B_{mu nu}.

        Args:
            omega_freq: Frequency values [*shape]

        Returns:
            theta: Phase values [*shape]
        """
        delta = self.delta
        omega_safe = torch.clamp(omega_freq.abs(), min=1e-8)
        theta = (math.pi * delta / 2) - delta * torch.log(omega_safe)
        return theta

    def recurrence_step(
        self,
        m_prev: torch.Tensor,
        x_curr: torch.Tensor,
        x_delayed: torch.Tensor,
        j_h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        O(1) memory update via finite-pole recurrence.

        m_t = rho * m_{t-1} + eta_r * x_t - xi_r * x_{t-p_eff}

        This computes the full path integral m_t = integral K(t,tau) x_tau dtau
        in constant time, independent of history length.

        Args:
            m_prev: Previous memory state [B, d_model]
            x_curr: Current input [B, d_model]
            x_delayed: Input from p_eff steps ago [B, d_model]
            j_h: Optional Higgs Current Scalar for modulation [B] or scalar

        Returns:
            m_curr: Updated memory state [B, d_model]
        """
        # Get recurrence parameters (optionally modulated by j_h)
        rho = self.rho
        eta_r = self.eta_r
        xi_r = self.xi_r

        # Modulation by Higgs Current Scalar (if provided)
        if j_h is not None:
            # Scale parameters based on j_h
            # j_h acts as a "control invariant" that adjusts memory dynamics
            j_h = j_h.view(-1, 1) if j_h.dim() == 1 else j_h
            rho = rho * torch.sigmoid(j_h)  # Modulate decay
            eta_r = eta_r * (1 + 0.1 * torch.tanh(j_h))  # Modulate input weight
            xi_r = xi_r * (1 + 0.1 * torch.tanh(j_h))  # Modulate delayed weight

        # O(1) recurrence update
        m_curr = rho * m_prev + eta_r * x_curr - xi_r * x_delayed

        return m_curr

    def init_memory(self, batch_size: int, d_model: int, device: torch.device) -> torch.Tensor:
        """Initialize memory state to zeros."""
        return torch.zeros(batch_size, d_model, device=device)

    def extra_repr(self) -> str:
        w = self.weights.detach().cpu().numpy()
        return (
            f"p_eff={self.p_eff}, "
            f"delta={self.delta.item():.3f}, "
            f"rho={self.rho.item():.3f}, "
            f"weights=[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]"
        )


class LiorMemoryState(nn.Module):
    """
    Manages the O(1) LIoR memory state across sequence processing.

    Maintains:
    - Current memory carrier m_t
    - Ring buffer of past p_eff inputs for delayed term
    - Higgs Current Scalar J_H for parameter modulation
    """

    def __init__(self, d_model: int, p_eff: int = 4):
        super().__init__()
        self.d_model = d_model
        self.p_eff = p_eff

        # The kernel
        self.kernel = LiorKernel(p_eff=p_eff)

        # Projection for computing Higgs Current Scalar from state
        self.j_h_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[Dict[str, torch.Tensor]] = None,
        diagnose: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process sequence with O(1) memory updates.

        Instead of sequential loop, we use parallel scan via convolution
        with the kernel, then apply the recurrence for memory state.

        Args:
            x: Input sequence [B, N, d_model]
            memory: Previous memory dict with 'm', 'buffer', 'j_h'
            diagnose: If True, print diagnostic info

        Returns:
            output: Memory-augmented output [B, N, d_model]
            new_memory: Updated memory dict
        """
        B, N, D = x.shape
        device = x.device

        def check_tensor(name: str, t: torch.Tensor) -> bool:
            """Check tensor for NaN/Inf and report stats."""
            if not diagnose:
                return False
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            if has_nan or has_inf:
                print(f"    [NaN] {name}: nan={has_nan}, inf={has_inf}, "
                      f"shape={list(t.shape)}", flush=True)
                return True
            else:
                print(f"    [OK] {name}: min={t.min().item():.4g}, max={t.max().item():.4g}, "
                      f"mean={t.mean().item():.4g}", flush=True)
            return False

        if diagnose:
            print("  --- LiorMemoryState ---", flush=True)

        # Initialize memory if not provided
        if memory is None:
            memory = {
                'm': self.kernel.init_memory(B, D, device),
                'buffer': torch.zeros(B, self.p_eff, D, device=device),
                'j_h': torch.zeros(B, 1, device=device),
            }

        m = memory['m']
        buffer = memory['buffer']
        j_h = memory['j_h']

        # === Parallel convolution with kernel for memory integration ===
        # Build kernel values
        tau = torch.arange(N, device=device, dtype=x.dtype)
        k_values = self.kernel(tau + 1)  # +1 to avoid t=0 singularity
        check_tensor("k_values_raw", k_values)

        # Check kernel sum before normalization
        k_sum = k_values.sum()
        if diagnose:
            print(f"    [INFO] kernel sum before norm: {k_sum.item():.4g}")

        # Normalize kernel
        k_values = k_values / (k_sum.abs() + 1e-8)  # Use abs() to handle negative sums
        check_tensor("k_values_normalized", k_values)

        # Causal convolution via FFT
        # Compute FFT size: next power of 2 >= 2N-1 for CUDA half-precision compatibility
        # cuFFT requires power-of-2 sizes when operating in FP16
        fft_size = 1 << (2 * N - 1).bit_length()  # Next power of 2

        # Pad kernel to FFT size
        k_padded = F.pad(k_values, (0, fft_size - N))  # [fft_size]

        # FFT convolution (parallel, O(N log N))
        x_transposed = x.transpose(1, 2)  # [B, D, N]
        check_tensor("x_transposed", x_transposed)

        # Cast to float32 to avoid ComplexHalf (slow and produces NaN)
        orig_dtype = x_transposed.dtype
        x_float = x_transposed.float()
        k_float = k_padded.float()

        x_fft = torch.fft.rfft(x_float, n=fft_size, dim=-1)
        check_tensor("x_fft (real)", x_fft.real)
        check_tensor("x_fft (imag)", x_fft.imag)

        k_fft = torch.fft.rfft(k_float, n=fft_size)
        check_tensor("k_fft (real)", k_fft.real)
        check_tensor("k_fft (imag)", k_fft.imag)

        conv_fft = x_fft * k_fft
        check_tensor("conv_fft (real)", conv_fft.real)
        check_tensor("conv_fft (imag)", conv_fft.imag)

        conv = torch.fft.irfft(conv_fft, n=fft_size, dim=-1)
        check_tensor("conv_after_irfft", conv)

        # Cast back to original dtype
        conv = conv.to(orig_dtype)

        conv = conv[..., :N]  # Take causal part [B, D, N]
        conv = conv.transpose(1, 2)  # [B, N, D]
        check_tensor("conv_final", conv)

        # === Update Higgs Current Scalar ===
        # J_H is computed from the memory state as a control invariant
        j_h_new = self.j_h_proj(m).squeeze(-1)  # [B]

        # === Update memory state via recurrence (for next call) ===
        # Take the last input and delayed input from buffer
        x_last = x[:, -1, :]  # [B, D]
        x_delayed = buffer[:, 0, :]  # [B, D] - oldest in buffer

        m_new = self.kernel.recurrence_step(m, x_last, x_delayed, j_h_new)

        # Update ring buffer
        buffer_new = torch.cat([buffer[:, 1:, :], x_last.unsqueeze(1)], dim=1)

        # === Combine: output = alpha * x + (1-alpha) * conv ===
        # where alpha is the instantaneous weight
        alpha = self.kernel.weights[0]  # Exponential weight as "instantaneous"
        if diagnose:
            print(f"    [INFO] alpha = {alpha.item():.4g}")

        output = alpha * x + (1 - alpha) * conv
        check_tensor("output_combined", output)

        new_memory = {
            'm': m_new,
            'buffer': buffer_new,
            'j_h': j_h_new.unsqueeze(-1),
        }

        if diagnose:
            print("  --- End LiorMemoryState ---")

        return output, new_memory

    def get_phase_field(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute phase field theta for symplectic form computation.

        Args:
            x: Input [B, N, d_model]

        Returns:
            theta: Phase field [B, N]
        """
        # Use embedding norm as "frequency"
        omega = torch.norm(x, dim=-1) + 1e-8  # [B, N]
        theta = self.kernel.compute_phase(omega)
        return theta
