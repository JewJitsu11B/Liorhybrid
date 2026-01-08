"""
Main Tensor Field Class

Implements the cognitive tensor field T_ij(x,t) with all evolution operators.

Paper References:
- Equation (1): Master equation
- Equation (13): Discretized evolution
- Algorithm 1: Full evolution loop
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
from collections import deque
from .config import FieldConfig
from ..kernels.hamiltonian import hamiltonian_evolution
from ..kernels.bayesian import bayesian_recursive_term
from ..kernels.fractional_memory import fractional_memory_term


class CognitiveTensorField:
    """
    Rank-2 tensor field evolving under Bayesian recursive dynamics.

    Field structure:
        T_ij[x,y] ∈ ℂ^{D×D} at each spatial location (x,y)

    Total shape: (N_x, N_y, D, D) complex
    """

    def __init__(self, config: FieldConfig):
        """
        Initialize tensor field.

        Args:
            config: Field configuration (see config.py)

        Implementation follows Paper Section 5.
        """
        self.config = config
        self.device = torch.device(config.device)

        # Initialize field (Paper Algorithm 1, line 1)
        # T_ij[x,y,0] ~ N(0, σ²) random Gaussian
        self.T = self._initialize_field()

        # History buffer for fractional memory (Paper Equation 8)
        # Using deque for O(1) append/pop operations
        self.history: deque = deque(maxlen=config.memory_window)

        # Store previous collapsed state for QR (Paper Equation 4)
        self.T_prev_collapsed: Optional[torch.Tensor] = None

        # Time tracking
        self.t = 0.0
        self.step_count = 0

        # Adaptive parameters (Paper Corollary: Adaptive Learning)
        if self.config.adaptive_learning:
            # Convert to learnable spatial fields
            self.alpha = torch.nn.Parameter(
                torch.full((1,), self.config.alpha, device=self.device, dtype=torch.float32)
            )
            self.nu = torch.nn.Parameter(
                torch.full(self.config.spatial_size, self.config.nu, device=self.device, dtype=torch.float32)
            )
            self.tau = torch.nn.Parameter(
                torch.full(self.config.spatial_size, self.config.tau, device=self.device, dtype=torch.float32)
            )
        else:
            # Use config values directly (non-adaptive mode)
            self.alpha = self.config.alpha
            self.nu = self.config.nu
            self.tau = self.config.tau

    def _initialize_field(self) -> torch.Tensor:
        """
        Random initialization of T_ij.

        Returns:
            Complex tensor of shape (N_x, N_y, D, D)

        TODO: Implement structured initialization for specific tasks
        """
        N_x, N_y = self.config.spatial_size
        D = self.config.tensor_dim

        # Complex Gaussian: T = (real + i*imag) / sqrt(2)
        real = torch.randn(N_x, N_y, D, D, device=self.device)
        imag = torch.randn(N_x, N_y, D, D, device=self.device)

        T = (real + 1j * imag) / (2.0 ** 0.5)

        return T.to(dtype=self.config.dtype)

    def evolve_step(
        self,
        evidence: Optional[torch.Tensor] = None,
        external_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single timestep evolution with Bayesian gradient-modulated update.

        **Modified Bayesian Formulation:**
        ∂_t T = (1/iℏ)[-(1 - w_mem)∇H[T] + Λ_QR[T] + J]

        where:
        - ∇H[T] = Hamiltonian gradient (energy direction)
        - w_mem = memory weight from fractional kernel
        - (1 - w_mem) modulates gradient based on prior
        - Λ_QR[T] = Bayesian recursive term (likelihood update)
        - J = external input

        **Key Difference from Original:**
        Old: ∂_t T = (1/iℏ)[H[T] + Λ_QR[T] - Λ_F[T] + J]
             → Adds memory field Λ_F, violates probability conservation

        New: Memory modulates gradient instead of adding field vector
             → Ensures P(ψ|D) ∝ P(D|ψ) P(ψ_prior) (Bayes' rule)
             → Conserves energy, prevents field explosion

        Args:
            evidence: E_ij for Bayesian update (Paper Eq 6)
            external_input: J_ij external stimulus (Paper Eq 1)

        Returns:
            Updated field T at t + Δt

        Paper Algorithm 1, lines 5-15 (with Bayesian modification)
        """
        from Liorhybrid.utils.pipeline_audit import audit_file_once
        audit_file_once("field_evolve_step", __file__)

        # 1. Hamiltonian evolution (Paper Eq 2)
        H_T = hamiltonian_evolution(
            self.T,
            hbar_cog=self.config.hbar_cog,
            m_cog=self.config.m_cog
        )

        # 2. Bayesian recursive term (Paper Eq 4)
        # Extract parameter values (handle both scalar and Parameter cases)
        tau_val = self.tau.mean() if self.config.adaptive_learning else self.tau

        Lambda_QR = bayesian_recursive_term(
            T_current=self.T,
            T_prev_collapsed=self.T_prev_collapsed,
            evidence=evidence,
            lambda_QR=self.config.lambda_QR,
            tau=tau_val
        )

        # 3. Fractional memory - now as gradient modulation (Bayesian formulation)
        # Instead of adding Lambda_F as a field vector, we use it to modulate gradients
        # This ensures probability conservation and prevents energy injection
        if len(self.history) > 0:
            # Extract alpha value (handle both scalar and Parameter cases)
            alpha_val = self.alpha if self.config.adaptive_learning else self.alpha

            from ..kernels.fractional_memory import fractional_memory_weight
            memory_weight = fractional_memory_weight(
                history=self.history,
                alpha=alpha_val,
                lambda_F=self.config.lambda_F,
                dt=self.config.dt
            )
        else:
            # No history yet: no memory damping
            memory_weight = torch.zeros((), device=self.T.device, dtype=torch.float32)

        # 4. External input (Paper Eq 1)
        J = external_input if external_input is not None else 0.0

        # 5. Compute Hamiltonian gradient for Bayesian update
        # ∇H[T] represents the direction of energy increase
        from ..kernels.gradients import compute_hamiltonian_gradient
        grad_H = compute_hamiltonian_gradient(
            T=self.T,
            H_T=H_T,
            hbar_cog=self.config.hbar_cog
        )

        # 6. Memory-modulated gradient (Bayesian posterior)
        # P(ψ|D) ∝ P(D|ψ) P(ψ_prior)
        # Gradient is weighted by (1 - memory_weight):
        #   - memory_weight = 0: Full gradient (pure likelihood)
        #   - memory_weight = 1: No gradient (pure prior)
        #   - In between: Bayesian combination
        # memory_weight is a scalar tensor in [0,1] (no CPU sync).
        effective_grad = (1.0 - memory_weight) * grad_H

        # 7. Update (Modified Bayesian formulation)
        # Old (energy-injecting): dT = (dt/iℏ)(H_T + Lambda_QR - Lambda_F + J)
        # New (energy-conserving): dT = (dt/iℏ)(-effective_grad + Lambda_QR + J)
        dT = (self.config.dt / (1j * self.config.hbar_cog)) * (
            -effective_grad + Lambda_QR + J
        )

        self.T = self.T + dT

        # 6. Update history (Paper Algorithm 1, line 16)
        self._update_history()

        # 7. Optional local adaptive update (only for standalone/debug use).
        # In training, adaptive gradients are handled by the main backward.
        if getattr(self.config, "adaptive_learning", False) and getattr(self.config, "local_adapt", False):
            self.adapt_parameters(use_autograd=True, apply_grads=True)

        # 8. Update time
        self.t += self.config.dt
        self.step_count += 1

        return self.T

    def _update_history(self):
        """
        Maintain history buffer for fractional memory.

        Paper Algorithm 1, lines 16-19

        Note: Using deque with maxlen automatically handles overflow.
        """
        self.history.append(self.T.clone())

    def get_norm_squared(self) -> float:
        """
        Compute squared field norm for conservation check.

        ||T||² = Σ_ijxy |T_ij[x,y]|²

        Paper Section 6.1 (Conservation Laws)
        """
        return torch.sum(torch.abs(self.T)**2).item()

    def compute_energy(self) -> float:
        """
        Compute total Hamiltonian energy of the field.

        Energy definition:
            E = Re[⟨T|H|T⟩] = Re[Σ_xy Tr(T†(x,y) H[T](x,y))]

        where H[T] is the Hamiltonian evolution operator.

        Returns:
            Scalar energy value (real number)

        Physical Interpretation:
            - Positive kinetic energy from field gradients (∇²T term)
            - Potential energy from V·T term (if V present)
            - Total energy should be conserved in pure Hamiltonian evolution
            - With Bayesian/memory terms, energy is not conserved (non-unitary)

        Paper Section 6.1: Conservation Laws

        Note: Current implementation uses explicit loops for clarity and numerical
        stability. For large grids, this could be optimized using vectorized operations
        (e.g., torch.einsum) but at the cost of code readability. Benchmark before
        optimizing if performance is critical.
        """
        # Compute Hamiltonian operator applied to current field
        H_T = hamiltonian_evolution(
            self.T,
            hbar_cog=self.config.hbar_cog,
            m_cog=self.config.m_cog
        )

        # Compute energy as Re[⟨T|H|T⟩] = Re[Σ Tr(T† H_T)]
        # For each spatial point (x,y), compute Tr(T†(x,y) @ H_T(x,y))
        energy = 0.0
        N_x, N_y, D, _ = self.T.shape

        for x in range(N_x):
            for y in range(N_y):
                T_xy = self.T[x, y, :, :]  # (D, D) tensor at point (x,y)
                H_xy = H_T[x, y, :, :]     # (D, D) Hamiltonian at point (x,y)

                # Compute trace of T†(x,y) @ H_T(x,y)
                # For complex matrices: ⟨A|B⟩ = Tr(A† B)
                trace_val = torch.trace(T_xy.conj().T @ H_xy)
                energy += torch.real(trace_val).item()

        return energy

    def compute_unitarity_deviation(self) -> float:
        """
        Measure deviation from unitary evolution.

        For unitary evolution, the operator U satisfies U†U = I.
        The Bayesian recursive and memory terms break unitarity,
        making the evolution non-unitary (dissipative).

        This function computes a measure of how much the field
        deviates from unitary behavior by checking if the norm
        of spatial density matrix deviates from identity.

        Measure:
            δ_unitarity = |⟨T|T⟩ - 1| / 1

        where ⟨T|T⟩ = Σ_xy Tr(T†(x,y) T(x,y)) / (N_x * N_y * D)

        Returns:
            Deviation from unitarity (0 = unitary, >0 = non-unitary)

        Physical Interpretation:
            - δ ≈ 0: Nearly unitary (pure Hamiltonian evolution)
            - δ > 0: Non-unitary (Bayesian updates dominating)
            - Large δ: Strong dissipation from memory/Bayesian terms

        Paper Section 6.1: Modified Unitarity
        """
        N_x, N_y, D, _ = self.T.shape

        # Compute average trace of T†T per spatial point
        total_trace = 0.0
        for x in range(N_x):
            for y in range(N_y):
                T_xy = self.T[x, y, :, :]
                trace_val = torch.trace(T_xy.conj().T @ T_xy)
                total_trace += torch.real(trace_val).item()

        # Average over spatial points and tensor dimension
        avg_trace = total_trace / (N_x * N_y * D)

        # Deviation from unity (perfect unitarity would give avg_trace = 1)
        # Normalize current norm
        current_norm_sq = self.get_norm_squared()
        expected_norm_sq = N_x * N_y * D * D  # For normalized field

        deviation = abs(current_norm_sq / expected_norm_sq - 1.0)

        return deviation

    def compute_entropy(self) -> torch.Tensor:
        """
        Compute variable-order entropy functional H^(ν(x))[Ψ].

        Paper Definition 2:
            H^(ν(x))[Ψ] = ∫ |Ψ(y,t)|^(2ν(x)) φ(x,y) dV

        Simplified form (uniform kernel φ=1):
            H = Σ_xy |T_ij(x,y)|^(2ν(x))

        Returns:
            Scalar entropy value (differentiable for autograd)

        Note: This computation maintains gradient information when
        parameters are torch.nn.Parameter objects, enabling automatic
        differentiation for adaptive learning.
        """
        # Get spatial dimensions
        N_x, N_y, D, D_out = self.T.shape

        # Compute |Ψ|² for each spatial point (sum over tensor indices)
        T_magnitude_sq = torch.sum(torch.abs(self.T)**2, dim=(2, 3))  # (N_x, N_y)

        # Apply variable-order scaling: |Ψ|^(2ν(x))
        if self.config.adaptive_learning:
            nu_expanded = self.nu  # Already (N_x, N_y) spatial field
        else:
            # Fixed nu: broadcast scalar to spatial shape
            nu_expanded = self.nu

        # Compute entropy with variable-order exponent
        # Use torch.pow for differentiability
        entropy = torch.sum(torch.pow(T_magnitude_sq, nu_expanded))

        return entropy

    def adapt_parameters(self, use_autograd: bool = True, apply_grads: bool = True):
        """
        Adaptive learning: update {α, ν, τ} via entropy gradients.

        Paper Corollary (Adaptive Learning):
            d/dt{α, ν, τ} = -∇_{α,ν,τ} E[H^(ν(x))[Ψ]]

        Modes:
            - use_autograd=True: compute entropy and populate grads via backward.
            - apply_grads=True: apply updates using current grads (manual step).

        Trainer-controlled flows can disable autograd here and reuse grads from
        the main loss to avoid multiple backwards per TBPTT window.

        Parameter constraints enforced:
            - α ∈ (0, 2): Memory persistence (Paper Axiom 4)
            - ν ∈ (0, 1]: Spatial entropy variation (Paper Axiom 3)
            - τ > 0: Temperature gating (Paper Axiom 5)
        """
        if not self.config.adaptive_learning:
            return

        if use_autograd:
            # Compute entropy (requires_grad=True from Parameters)
            H = self.compute_entropy()

            # Compute gradients via backpropagation
            H.backward(retain_graph=False)

        if not apply_grads:
            # Leave grads populated for trainer/optimizer handling
            return

        # Gradient descent update with torch.no_grad() context (manual step mode)
        with torch.no_grad():
            # Update alpha (memory persistence)
            if self.alpha.grad is not None:
                self.alpha -= self.config.param_learning_rate * self.alpha.grad
                self.alpha.clamp_(0.01, 1.99)  # α ∈ (0, 2)
                self.alpha.grad.zero_()

            # Update nu (spatial entropy variation)
            if self.nu.grad is not None:
                self.nu -= self.config.param_learning_rate * self.nu.grad
                self.nu.clamp_(0.01, 1.0)  # ν ∈ (0, 1]
                self.nu.grad.zero_()  # Safe because manual step mode

            # Update tau (temperature gating)
            if self.tau.grad is not None:
                self.tau -= self.config.param_learning_rate * self.tau.grad
                self.tau.clamp_(min=0.01)  # τ > 0
                self.tau.grad.zero_()

    def detach_state(self):
        """
        Detach state and cached history at TBPTT window boundaries.

        - If T is a state tensor: rebind to a detached tensor.
        - If T is an nn.Parameter tracked by an optimizer: copy detached data
          in-place under no_grad to preserve parameter identity.
        """
        if isinstance(self.T, torch.nn.Parameter):
            with torch.no_grad():
                self.T.copy_(self.T.detach())
        else:
            self.T = self.T.detach()

        if hasattr(self, "history") and isinstance(self.history, deque):
            self.history = deque(
                [h.detach() for h in self.history],
                maxlen=self.history.maxlen
            )

    def assign_tokens(self):
        """
        Assign token IDs to field regions for semantic clustering.

        TODO: Implement token-based semantic distance clustering.

        Architecture:
        - Distance metric: Normalized similarity to token group members
        - Addressing: Route-hashable with coordinates, parent path
        - Geometric data: Local metric tensor, Christoffel symbols
        - Products: Tensor, wedge, spinor products with entity metrics
        - Neighbor structure: 32 NN, 16 min-heap, 16 max-heap
        - Error correction: 4x8 BCH ECC for addressing

        This replaces the naive outer-product approach which would
        create an OOM N²xN² correlation matrix.
        """
        raise NotImplementedError(
            "Token-based clustering not yet implemented. "
            "Will use LIoR distance clustering with semantic addressing."
        )

    def get_cluster_distances(self, token_id: int):
        """
        Compute distances from field elements to token cluster centroid.

        TODO: Implement distance calculation with metric tensor.

        Args:
            token_id: ID of the token cluster

        Returns:
            Distance map for elements to this cluster
        """
        raise NotImplementedError(
            "Distance-based clustering not yet implemented."
        )

    def state_dict(self) -> dict:
        """
        Serialize field state for checkpoints / inference.

        Note: This is intentionally lightweight and does not include history buffers.
        """
        from Liorhybrid.utils.pipeline_audit import audit_file_once
        audit_file_once("field_state_dict", __file__)

        state = {
            'T': self.T,
            't': self.t,
            'step_count': self.step_count,
            'spatial_size': self.config.spatial_size,
            'tensor_dim': self.config.tensor_dim,
            'adaptive_learning': bool(self.config.adaptive_learning),
        }

        if self.config.adaptive_learning:
            state.update({
                'alpha': self.alpha,
                'nu': self.nu,
                'tau': self.tau,
            })
        else:
            # Store as python floats to avoid device-specific restore needs.
            state.update({
                'alpha': float(self.alpha),
                'nu': float(self.nu),
                'tau': float(self.tau),
            })

        return state

    def load_state_dict(self, state: dict) -> None:
        """
        Restore field state from checkpoint / inference load.
        """
        from Liorhybrid.utils.pipeline_audit import audit_file_once
        audit_file_once("field_load_state_dict", __file__)

        if 'T' not in state:
            raise KeyError("Field state_dict missing key 'T'")

        self.T = state['T'].to(self.device)
        self.t = float(state.get('t', 0.0))
        self.step_count = int(state.get('step_count', 0))

        if bool(state.get('adaptive_learning', self.config.adaptive_learning)) and self.config.adaptive_learning:
            with torch.no_grad():
                if 'alpha' in state:
                    self.alpha.copy_(state['alpha'].to(self.device, dtype=self.alpha.dtype).view_as(self.alpha))
                if 'nu' in state:
                    self.nu.copy_(state['nu'].to(self.device, dtype=self.nu.dtype).view_as(self.nu))
                if 'tau' in state:
                    self.tau.copy_(state['tau'].to(self.device, dtype=self.tau.dtype).view_as(self.tau))
        else:
            # Non-adaptive: restore scalar config-style values.
            if 'alpha' in state:
                self.alpha = float(state['alpha'])
            if 'nu' in state:
                self.nu = float(state['nu'])
            if 'tau' in state:
                self.tau = float(state['tau'])
