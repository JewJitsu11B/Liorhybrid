"""
Stability Monitoring and Gauge Renormalization

Diagnostic-first approach to handling NaN/Inf during training.

Separation of concerns:
- StabilityMonitor ANALYZES (returns level + diagnostic)
- GaugeRenormalizer DECIDES action based on level

Usage:
    from .stability import StabilityMonitor, GaugeRenormalizer, StabilityLevel

    monitor = StabilityMonitor(strict=False)
    renorm = GaugeRenormalizer()

    # In training loop:
    level, diag = monitor.analyze("attention_scores", tensor, step=42)
    action = renorm.respond(level, diag, tensor, optimizer)
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
from typing import Dict, Optional


class StabilityLevel:
    """Severity levels for numerical instability."""
    CLEAN = 0       # No issues
    MICRO = 1       # 0 < fraction < 1e-6 (negligible, log only)
    LOCAL = 2       # 1e-6 ≤ fraction < 0.1 (localized, patch tensor)
    LAYER_MELT = 3  # 0.1 ≤ fraction < 0.5 (layer-level, restore from EMA)
    GLOBAL_MELT = 4 # ≥ 0.5 (catastrophic, halt training)


class StabilityError(Exception):
    """Raised when stability check fails in strict mode."""
    pass


class StabilityMonitor:
    """
    Analyzes tensors for NaN/Inf and returns diagnostic info.

    Separation of concerns:
    - StabilityMonitor ANALYZES (returns level + diagnostic)
    - GaugeRenormalizer DECIDES action based on level

    Usage:
        monitor = StabilityMonitor(strict=False)  # Training mode
        level, diag = monitor.analyze("attention_scores", tensor, step=42)
        if level >= StabilityLevel.LOCAL:
            # Take action based on level
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, raise StabilityError on LOCAL or worse.
                    Use True for development/debugging, False for training.
        """
        self.strict = strict
        self.history = []  # Track instability events for post-mortem

    def analyze(
        self,
        name: str,
        tensor: torch.Tensor,
        step: int
    ) -> tuple:
        """
        Analyze tensor for NaN/Inf values.

        Args:
            name: Parameter/tensor name for logging
            tensor: Tensor to analyze
            step: Current training step

        Returns:
            (level, diagnostic): StabilityLevel and diagnostic dict (or None if CLEAN)
        """
        nan_mask = torch.isnan(tensor)
        inf_mask = torch.isinf(tensor)
        bad_mask = nan_mask | inf_mask

        count = bad_mask.sum().item()
        total = tensor.numel()

        if count == 0:
            return StabilityLevel.CLEAN, None

        frac = count / total

        # Classify severity
        if frac < 1e-6:
            level = StabilityLevel.MICRO
        elif frac < 0.1:
            level = StabilityLevel.LOCAL
        elif frac < 0.5:
            level = StabilityLevel.LAYER_MELT
        else:
            level = StabilityLevel.GLOBAL_MELT

        # Build diagnostic
        diagnostic = {
            'name': name,
            'step': step,
            'level': level,
            'level_name': ['CLEAN', 'MICRO', 'LOCAL', 'LAYER_MELT', 'GLOBAL_MELT'][level],
            'nan_count': nan_mask.sum().item(),
            'inf_count': inf_mask.sum().item(),
            'bad_count': count,
            'total': total,
            'frac': frac,
            'shape': tuple(tensor.shape),
        }

        # Add healthy value stats if any exist
        healthy = tensor[~bad_mask]
        if healthy.numel() > 0:
            diagnostic['healthy_min'] = healthy.min().item()
            diagnostic['healthy_max'] = healthy.max().item()
            diagnostic['healthy_mean'] = healthy.mean().item()
            diagnostic['healthy_std'] = healthy.std().item() if healthy.numel() > 1 else 0.0

        # Track in history
        self.history.append(diagnostic)

        # Strict mode: fail on LOCAL or worse
        if self.strict and level >= StabilityLevel.LOCAL:
            raise StabilityError(f"INSTABILITY DETECTED: {diagnostic}")

        return level, diagnostic

    def get_history(self) -> list:
        """Return all recorded instability events."""
        return self.history

    def clear_history(self):
        """Clear instability history."""
        self.history = []


class GaugeRenormalizer:
    """
    Takes action based on StabilityLevel from StabilityMonitor.

    Actions by level:
    - CLEAN: Continue
    - MICRO: Log, continue
    - LOCAL: Patch NaN/Inf in tensor, log warning
    - LAYER_MELT: Restore layer from EMA/checkpoint, reduce LR
    - GLOBAL_MELT: Halt training, dump state for post-mortem

    Usage:
        renorm = GaugeRenormalizer(ema_state=model_ema)
        action = renorm.respond(level, diagnostic, tensor, optimizer)
    """

    def __init__(
        self,
        ema_state: Optional[Dict] = None,
        lr_reduction: float = 0.5,
        patch_value: float = 0.0
    ):
        """
        Args:
            ema_state: Optional EMA state dict for LAYER_MELT recovery
            lr_reduction: Factor to reduce LR on LAYER_MELT (default 0.5)
            patch_value: Value to replace NaN/Inf with on LOCAL (default 0.0)
        """
        self.ema_state = ema_state
        self.lr_reduction = lr_reduction
        self.patch_value = patch_value
        self.action_log = []

    def respond(
        self,
        level: int,
        diagnostic: Optional[Dict],
        tensor: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> str:
        """
        Take action based on stability level.

        Args:
            level: StabilityLevel value
            diagnostic: Diagnostic dict from StabilityMonitor
            tensor: The tensor to potentially patch (modified in-place)
            optimizer: Optional optimizer for LR reduction

        Returns:
            action: String describing action taken
        """
        if level == StabilityLevel.CLEAN:
            return "continue"

        if level == StabilityLevel.MICRO:
            # Log only, no action
            print(f"[STABILITY:MICRO] {diagnostic['name']} has {diagnostic['bad_count']} bad values ({diagnostic['frac']:.2e})")
            self.action_log.append(('MICRO', diagnostic))
            return "logged"

        if level == StabilityLevel.LOCAL:
            # Patch tensor in-place
            bad_mask = torch.isnan(tensor) | torch.isinf(tensor)
            tensor[bad_mask] = self.patch_value
            print(f"[STABILITY:LOCAL] {diagnostic['name']}: Patched {diagnostic['bad_count']} values with {self.patch_value}")
            self.action_log.append(('LOCAL', diagnostic, 'patched'))
            return "patched"

        if level == StabilityLevel.LAYER_MELT:
            # Reduce LR if optimizer provided
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = old_lr * self.lr_reduction
                print(f"[STABILITY:LAYER_MELT] {diagnostic['name']}: Reduced LR by {self.lr_reduction}x")

            # TODO: Restore from EMA if available
            if self.ema_state is not None:
                print(f"[STABILITY:LAYER_MELT] EMA restore not yet implemented")

            # Patch tensor as fallback
            bad_mask = torch.isnan(tensor) | torch.isinf(tensor)
            tensor[bad_mask] = self.patch_value
            self.action_log.append(('LAYER_MELT', diagnostic, 'lr_reduced', 'patched'))
            return "lr_reduced+patched"

        if level == StabilityLevel.GLOBAL_MELT:
            # Catastrophic - halt training
            print(f"[STABILITY:GLOBAL_MELT] {diagnostic['name']}: HALTING - {diagnostic['frac']*100:.1f}% values are NaN/Inf")
            print(f"  Shape: {diagnostic['shape']}")
            print(f"  Step: {diagnostic['step']}")
            if 'healthy_min' in diagnostic:
                print(f"  Healthy range: [{diagnostic['healthy_min']:.4f}, {diagnostic['healthy_max']:.4f}]")
            self.action_log.append(('GLOBAL_MELT', diagnostic, 'halted'))
            raise StabilityError(f"GLOBAL_MELT at step {diagnostic['step']}: {diagnostic['name']}")

        return "unknown"

    def get_action_log(self) -> list:
        """Return all actions taken."""
        return self.action_log
