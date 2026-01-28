"""
LIoR-Based Training Loop

Implements Learning through Informational Recursive geodesic carving.

Key differences from standard gradient descent:
1. Loss = CrossEntropy + w_geodesic * GeodesicCost
2. Field evolution IS learning (α, ν, τ converge to optimal physics)
3. Adaptive parameters update via ∇H (entropy gradient), not ∇L
4. Geodesic cost guides optimization through physics

References:
- LIoR action: ∫ R(x) √|g_μν ẋ^μ ẋ^ν| dτ
- Field entropy H = -Tr(T log T)
- Adaptive dynamics: dα/dt = -η ∂H/∂α
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


def compute_geodesic_cost(
    embeddings: torch.Tensor,
    field_state: torch.Tensor,
    metric: Optional[torch.Tensor] = None,
    resilience_field: Optional[torch.Tensor] = None,
    attention_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute proper LIoR action: S = ∫ R(x) √(g_μν ẋ^μ ẋ^ν) dτ
    
    CORRECTED FORMULA: Uses square root of metric inner product for proper
    Riemannian arc length (was using squared norm before).
    
    LIoR action measures path deviation from geodesic carved by the cognitive field.
    
    Physics:
        - R(x): Local curvature/resilience (higher = more costly)
        - √(g(ẋ,ẋ)): Proper Riemannian arc length (not squared!)
        - Integration over proper time τ

    Args:
        embeddings: Sequence of embeddings (batch, seq_len, d_model)
        field_state: Cognitive tensor field (N_x, N_y, D, D)
        metric: (D, D) or None - Riemannian metric tensor
        resilience_field: (N_x, N_y) or None - R(x) curvature field
        attention_weights: Optional attention weights for weighting

    Returns:
        Geodesic cost (scalar tensor)

    Mathematical detail:
        OLD (wrong): cost = g_dx_dx / ||dx||
        NEW (correct): cost = √(g_dx_dx)
        
    High cost = Path deviates from geodesic (not following field flow)
    Low cost = Path follows natural geodesic (aligned with field)
    """
    from Liorhybrid.utils.pipeline_audit import audit_file_once
    audit_file_once("lior_geodesic", __file__)

    batch, seq_len, d_model = embeddings.shape
    
    # Extract field dimension
    D = field_state.shape[2]

    # Compute trajectory velocities: ẋ = dx/dτ
    dx = embeddings[:, 1:] - embeddings[:, :-1]  # (B, T-1, d)
    
    # Project to field subspace (D dimensions)
    if d_model > D:
        # Use first D dimensions (or could use learned projection)
        proj = dx[..., :D]  # (B, T-1, D)
    else:
        proj = dx
    
    # Build metric tensor from field
    if metric is None:
        # Derive from field: g_μν = ⟨Ψ|Ψ⟩
        T_avg = field_state.mean(dim=(0, 1))  # (D, D)
        T_real = torch.abs(T_avg) if T_avg.is_complex() else T_avg
        metric = T_real.T @ T_real  # (D, D) positive-definite
        metric = metric + 1e-6 * torch.eye(D, device=metric.device, dtype=metric.dtype)
        metric = metric.detach()  # Don't backprop through field
    
    # Cast metric to match embeddings dtype/device for stability
    metric = metric.to(device=proj.device, dtype=proj.dtype)
    
    # Metric inner product: g_μν ẋ^μ ẋ^ν
    # g(ẋ,ẋ) = ẋᵀ g ẋ
    g_dx_dx = torch.einsum('bti,ij,btj->bt', proj, metric, proj)  # (B, T-1)
    
    # CORRECT: Square root for proper Riemannian distance
    # OLD (wrong): cost = g_dx_dx / ||dx||
    # NEW (correct): cost = √(g_dx_dx)
    arc_length = torch.sqrt(torch.clamp(g_dx_dx, min=1e-8))  # (B, T-1)
    
    # Resilience field R(x)
    if resilience_field is not None:
        # Average over spatial points (or use attention-weighted)
        R = resilience_field.mean()
    else:
        # Default: R = 1 (flat space)
        R = 1.0
    
    # Weight by attention if provided (attend to important transitions)
    if attention_weights is not None:
        # Handle list of attention weights (from multiple layers)
        if isinstance(attention_weights, list):
            # Use last layer's attention (most refined)
            attention_weights = attention_weights[-1]

        # Average attention over heads: (batch, n_heads, seq, seq) -> (batch, seq, seq)
        attn_avg = attention_weights.mean(dim=1)  # (batch, seq, seq)

        # Extract transition weights: attn[t, t+1]
        transition_weights = torch.diagonal(attn_avg, offset=1, dim1=-2, dim2=-1)  # (batch, seq-1)

        arc_length = arc_length * transition_weights
    
    # LIoR action: ∫ R(x) √(g(ẋ,ẋ)) dτ
    lior_cost = R * arc_length.sum()
    
    return lior_cost


def compute_field_entropy(field_state: torch.Tensor) -> torch.Tensor:
    """
    Compute von Neumann entropy of cognitive field.

    H = -Tr(ρ log ρ)

    where ρ is the density matrix derived from field state.

    Used for adaptive parameter updates via ∇H.

    Args:
        field_state: Cognitive tensor field (N_x, N_y, D, D) complex

    Returns:
        Field entropy H (scalar)
    """
    from Liorhybrid.utils.pipeline_audit import audit_file_once
    audit_file_once("lior_field_entropy", __file__)

    # Average over spatial dimensions
    T_avg = field_state.mean(dim=(0, 1))  # (D, D)

    # Construct density matrix ρ = T^† T / Tr(T^† T)
    T_dag = torch.conj(T_avg).T  # Hermitian conjugate
    rho = torch.matmul(T_dag, T_avg)  # (D, D)

    # Normalize to unit trace
    trace = torch.trace(rho).real
    rho = rho / (trace + 1e-8)

    # Compute eigenvalues for entropy
    eigenvalues = torch.linalg.eigvalsh(rho)  # Returns real eigenvalues
    eigenvalues = torch.clamp(eigenvalues, min=1e-10)  # Avoid log(0)

    # Von Neumann entropy: H = -Σ λ_i log(λ_i)
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))

    return entropy


def update_adaptive_parameters(
    field,
    learning_rate_alpha: float = 1e-4,
    learning_rate_nu: float = 1e-5,
    learning_rate_tau: float = 1e-5
):
    """
    Update adaptive field parameters α, ν, τ via entropy gradient.

    These update independently of the main loss via:
        dα/dt = -η_α ∂H/∂α
        dν/dt = -η_ν ∂H/∂ν
        dτ/dt = -η_τ ∂H/∂τ

    This allows the field to learn optimal physics parameters that
    minimize entropy while the main loss trains the model.

    Args:
        field: CognitiveTensorField instance
        learning_rate_alpha: Learning rate for α updates
        learning_rate_nu: Learning rate for ν updates
        learning_rate_tau: Learning rate for τ updates
    """
    from Liorhybrid.utils.pipeline_audit import audit_file_once
    audit_file_once("lior_adaptive_update", __file__)

    if not field.config.adaptive_learning:
        return

    # Compute field entropy
    H = compute_field_entropy(field.T)

    # Compute gradients w.r.t. adaptive parameters
    # NOTE: This requires field.T to have been evolved WITH gradient tracking to alpha/nu/tau
    # If gradients are not available (field evolved without graph), skip update
    try:
        if field.alpha.requires_grad and field.T.requires_grad:
            alpha_grad = torch.autograd.grad(
                H, field.alpha, retain_graph=True, create_graph=False, allow_unused=True
            )[0]

            if alpha_grad is not None:
                # Update: α ← α - η ∂H/∂α
                with torch.no_grad():
                    field.alpha -= learning_rate_alpha * alpha_grad
                    field.alpha.clamp_(min=1e-6, max=1.0)  # Keep in valid range

        if field.nu.requires_grad and field.T.requires_grad:
            nu_grad = torch.autograd.grad(
                H, field.nu, retain_graph=True, create_graph=False, allow_unused=True
            )[0]

            if nu_grad is not None:
                with torch.no_grad():
                    field.nu -= learning_rate_nu * nu_grad
                    field.nu.clamp_(min=0.0, max=10.0)

        if field.tau.requires_grad and field.T.requires_grad:
            tau_grad = torch.autograd.grad(
                H, field.tau, retain_graph=True, create_graph=False, allow_unused=True
            )[0]

            if tau_grad is not None:
                with torch.no_grad():
                    field.tau -= learning_rate_tau * tau_grad
                    field.tau.clamp_(min=1e-6, max=1.0)

    except RuntimeError:
        # Gradients not available - field evolved without tracking
        # This is expected when field evolution is physics-based (not differentiable)
        pass


def lior_loss(
    outputs_dict: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    field_state: torch.Tensor,
    embeddings: torch.Tensor,
    attention_weights: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
    controller: Optional[Dict[str, float]] = None,
    controller_state: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute LIoR-based composite loss.

    Loss = w_lm * L_CrossEntropy + w_geodesic * L_Geodesic + w_contrastive * L_Contrastive

    Args:
        outputs_dict: Model outputs
            - logits: (batch, seq_len, vocab_size)
            - text_embedding: (batch, d_model) for contrastive
        batch: Training batch
            - input_ids: (batch, seq_len)
            - labels: (batch, seq_len)
            - attention_mask: (batch, seq_len)
        field_state: Cognitive tensor field (N_x, N_y, D, D)
        embeddings: Sequence embeddings (batch, seq_len, d_model)
        attention_weights: Attention weights for geodesic weighting
        weights: Loss component weights

    Returns:
        total_loss: Combined loss
        loss_dict: Individual loss components
    """
    from Liorhybrid.utils.pipeline_audit import audit_file_once
    audit_file_once("lior_loss", __file__)

    if weights is None:
        weights = {
            'lm': 1.0,
            'geodesic': 0.1,
            'contrastive': 0.01,
            'field_entropy': 0.001
        }

    loss_dict: Dict[str, torch.Tensor] = {}

    # 1. Language modeling loss (cross-entropy, causal LM shift, ignore padding)
    if 'logits' in outputs_dict and 'labels' in batch:
        from .losses import language_modeling_loss

        logits = outputs_dict['logits']
        labels = batch['labels']
        attention_mask = batch.get('attention_mask', None)

        lm_loss = language_modeling_loss(
            logits=logits,
            labels=labels,
            attention_mask=attention_mask,
            ignore_index=-100
        )
        loss_dict['lm_loss'] = lm_loss.detach()
    else:
        lm_loss = torch.zeros((), device=field_state.device, dtype=torch.float32)
        loss_dict['lm_loss'] = lm_loss

    # 2. Geodesic cost (LIoR-specific)
    geodesic_cost = compute_geodesic_cost(
        embeddings=embeddings,
        field_state=field_state,
        attention_weights=attention_weights
    )
    geodesic_cost = geodesic_cost.to(device=lm_loss.device, dtype=lm_loss.dtype)
    loss_dict['geodesic_cost'] = geodesic_cost.detach()

    # 2b. LIoR controller path (no-grad), LM-only scaling
    # g(ℓ)=clip(exp(-k(ℓ-ℓ0)), g_min, g_max), where ℓ0 is EMA baseline.
    if controller is None:
        controller = {}
    if controller_state is None:
        controller_state = {}

    enabled = bool(controller.get('enabled', True))
    mode = str(controller.get('mode', 'lm_only')).lower()
    g_t = torch.ones((), device=lm_loss.device, dtype=lm_loss.dtype)

    if enabled and mode == 'lm_only':
        geo_ctrl = geodesic_cost.detach()
        ema_prev = controller_state.get('geodesic_ema', None)

        with torch.no_grad():
            beta = float(controller.get('beta', 0.98))
            if ema_prev is None:
                ema = geo_ctrl
            else:
                ema = ema_prev.to(device=geo_ctrl.device, dtype=geo_ctrl.dtype) * beta + geo_ctrl * (1.0 - beta)
            controller_state['geodesic_ema'] = ema.detach()

            k_t = lm_loss.new_tensor(float(controller.get('k', 1.0)))
            g_min_t = lm_loss.new_tensor(float(controller.get('g_min', 0.1)))
            g_max_t = lm_loss.new_tensor(float(controller.get('g_max', 1.0)))

            g_t = torch.exp(-k_t * (geo_ctrl - ema))
            g_t = torch.clamp(g_t, min=g_min_t, max=g_max_t).detach()

        loss_dict['lior_ctrl_ema'] = controller_state['geodesic_ema']

    loss_dict['lior_ctrl_g'] = g_t

    # 3. Contrastive loss (if multimodal)
    if 'text_embedding' in outputs_dict and 'image_embedding' in outputs_dict:
        from .losses import contrastive_loss as compute_contrastive

        contrastive = compute_contrastive(
            outputs_dict['text_embedding'],
            outputs_dict['image_embedding']
        )
        contrastive = contrastive.to(device=lm_loss.device, dtype=lm_loss.dtype)
        loss_dict['contrastive_loss'] = contrastive.detach()
    else:
        contrastive = lm_loss.new_zeros(())
        loss_dict['contrastive_loss'] = contrastive

    # 4. Field entropy regularization
    field_entropy = compute_field_entropy(field_state).to(device=lm_loss.device, dtype=lm_loss.dtype)
    loss_dict['field_entropy'] = field_entropy.detach()

    # Combine losses
    g_t = g_t if torch.is_tensor(g_t) else lm_loss.new_tensor(g_t)
    g_t = g_t.to(device=lm_loss.device, dtype=lm_loss.dtype).detach()

    w_lm = lm_loss.new_tensor(weights['lm'])
    w_geo = lm_loss.new_tensor(weights['geodesic'])
    w_con = lm_loss.new_tensor(weights['contrastive'])
    w_ent = lm_loss.new_tensor(weights['field_entropy'])

    total_loss = (
        w_lm * (lm_loss * g_t) +
        w_geo * geodesic_cost +
        w_con * contrastive +
        w_ent * field_entropy
    )

    loss_dict['total_loss'] = total_loss.detach()

    return total_loss, loss_dict


class LIoRTrainingMixin:
    """
    Mixin class to add LIoR training capabilities to CognitiveTrainer.

    Provides:
    - Geodesic cost computation
    - Adaptive parameter updates via ∇H
    - Physics-guided optimization

    Usage:
        class MyTrainer(LIoRTrainingMixin, CognitiveTrainer):
            pass
    """

    def training_step_lior(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        LIoR-based training step.

        Replaces standard training_step with geodesic learning.

        Args:
            batch: Training batch

        Returns:
            loss: Total LIoR loss
            loss_dict: Loss components
        """
        # Evolve field (physics simulation)
        if self.config.get('evolve_field_during_training', True):
            self.field.evolve_step()

        # Get modality
        modality = batch.get('modality', 'text')
        if isinstance(modality, list):
            modality = modality[0]

        # Embed inputs
        if modality == 'text':
            Q_input = self.input_embedding(batch['input_ids'], modality='text')
        elif modality == 'image':
            Q_input = self.input_embedding(batch['image'], modality='image')
        elif modality == 'video':
            Q_input = self.input_embedding(batch['video'], modality='video')
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Forward through geometric transformer
        from torch.cuda.amp import autocast

        with autocast(enabled=self.use_amp):
            output, attn_weights = self.model(
                Q_input,
                self.field.T,
                time=self.field.t
            )

            # Project to vocabulary
            if hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(output)
            else:
                vocab_size = 32000
                if not hasattr(self, '_lm_head_temp'):
                    self._lm_head_temp = nn.Linear(output.shape[-1], vocab_size).to(self.device)
                logits = self._lm_head_temp(output)

            # Compute LIoR loss
            outputs_dict = {
                'logits': logits,
                'text_embedding': output.mean(dim=1)
            }

            loss, loss_dict = lior_loss(
                outputs_dict=outputs_dict,
                batch=batch,
                field_state=self.field.T,
                embeddings=output,
                attention_weights=attn_weights,
                weights=self.config.get('lior_loss_weights')
            )

        # Update adaptive parameters via ∇H (separate from main backprop)
        if self.field.config.adaptive_learning:
            update_adaptive_parameters(
                field=self.field,
                learning_rate_alpha=self.config.get('lr_alpha', 1e-4),
                learning_rate_nu=self.config.get('lr_nu', 1e-5),
                learning_rate_tau=self.config.get('lr_tau', 1e-5)
            )

        return loss, loss_dict
