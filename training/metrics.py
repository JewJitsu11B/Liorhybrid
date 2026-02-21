"""
Training Metrics and Logging

Comprehensive metrics tracking for Bayesian Cognitive Field training.

Tracks:
- Training progress (epoch, batch, step)
- Timing (batch time, step time, throughput)
- Computational complexity
- All loss components
- Field state metrics (alpha, nu, tau, entropy)
- Gradient information
- Symplectic integrator diagnostics (energy conservation)
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import time
import math
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class TrainingMetrics:
    """
    Comprehensive training metrics.

    Tracks everything needed for LIoR-based training monitoring.
    """

    # Progress tracking
    epoch: int = 0
    batch: int = 0
    step: int = 0

    # Timing
    batch_time: float = 0.0
    step_time: float = 0.0
    data_load_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0

    # Throughput
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    seq_length: int = 0

    # Computational complexity
    complexity: str = "O(N^2)"  # Will be O(N) with Mamba
    flops_estimate: float = 0.0  # GFLOPs per batch
    flops_per_token: float = 0.0  # MFLOPs per token (at current N)
    base_ops_per_token: float = 0.0  # Base ops: O(N^2)->per token*N, O(N)->per token
    tflops_per_sec: float = 0.0  # TFLOP/s throughput
    mfu_percent: float = 0.0  # Model FLOPs Utilization (%)

    # Loss components
    total_loss: float = 0.0
    lm_loss: float = 0.0
    contrastive_loss: float = 0.0
    alignment_loss: float = 0.0
    geodesic_cost: float = 0.0
    field_entropy: float = 0.0

    # Field state metrics
    field_alpha: float = 0.0
    field_nu_mean: float = 0.0
    field_tau_mean: float = 0.0
    field_hamiltonian: float = 0.0
    field_entropy_gradient_norm: float = 0.0
    field_energy: float = 0.0

    # Gradient statistics
    grad_norm: float = 0.0
    max_grad: float = 0.0

    # Learning rate
    learning_rate: float = 0.0

    # Memory usage (if CUDA)
    memory_allocated_mb: float = 0.0
    memory_reserved_mb: float = 0.0

    # Geometric weights
    weight_wedge: float = 0.0
    weight_tensor: float = 0.0
    weight_spinor: float = 0.0
    temperature: float = 0.0

    # Moving averages (for smoothing)
    avg_loss: float = 0.0
    avg_batch_time: float = 0.0

    # Symplectic integrator diagnostics
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    total_hamiltonian_energy: float = 0.0
    energy_drift: float = 0.0  # Drift from initial energy
    energy_drift_percent: float = 0.0


class MetricsLogger:
    """
    Logger for training metrics.

    Handles:
    - Console logging
    - JSON logging
    - Moving averages
    - Periodic summaries
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        log_interval: int = 10,
        smoothing: float = 0.9
    ):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory for JSON logs
            log_interval: Log every N steps
            smoothing: Exponential smoothing factor for moving averages
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = log_interval
        self.smoothing = smoothing

        # History
        self.history: List[Dict] = []

        # Moving averages
        self.ema_loss = 0.0
        self.ema_batch_time = 0.0

        # Timing
        self.step_start_time = None
        self.batch_start_time = None

    def start_step(self):
        """Mark start of training step."""
        self.step_start_time = time.time()

    def start_batch(self):
        """Mark start of batch processing."""
        self.batch_start_time = time.time()

    def compute_metrics(
        self,
        model,
        field,
        optimizer,
        loss_dict: Dict[str, float],
        batch_size: int,
        seq_length: int
    ) -> TrainingMetrics:
        """
        Compute comprehensive metrics from training state.

        Args:
            model: GeometricTransformer model
            field: CognitiveTensorField
            optimizer: PyTorch optimizer
            loss_dict: Dictionary of loss components
            batch_size: Current batch size
            seq_length: Sequence length

        Returns:
            TrainingMetrics with all fields populated
        """
        metrics = TrainingMetrics()

        # Timing (with GPU sync for accurate measurements)
        if self.step_start_time is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            metrics.step_time = time.time() - self.step_start_time

        if self.batch_start_time is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            metrics.batch_time = time.time() - self.batch_start_time

        # Throughput
        metrics.seq_length = seq_length
        if metrics.batch_time > 0:
            metrics.samples_per_second = batch_size / metrics.batch_time
            metrics.tokens_per_second = (batch_size * seq_length) / metrics.batch_time

        # Loss components
        metrics.total_loss = loss_dict.get('total_loss', 0.0)
        metrics.lm_loss = loss_dict.get('lm_loss', 0.0)
        metrics.contrastive_loss = loss_dict.get('contrastive_loss', 0.0)
        metrics.alignment_loss = loss_dict.get('alignment_loss', 0.0)
        metrics.geodesic_cost = loss_dict.get('geodesic_cost', 0.0)
        metrics.field_entropy = loss_dict.get('field_entropy', 0.0)

        # Field state
        if field is not None:
            metrics.field_alpha = field.alpha.item() if hasattr(field, 'alpha') else 0.0
            metrics.field_nu_mean = field.nu.mean().item() if hasattr(field, 'nu') else 0.0
            metrics.field_tau_mean = field.tau.mean().item() if hasattr(field, 'tau') else 0.0

            # Field energy
            if hasattr(field, 'compute_energy'):
                try:
                    metrics.field_energy = field.compute_energy()
                except:
                    pass

            # Field Hamiltonian
            if hasattr(field, 'compute_hamiltonian'):
                try:
                    metrics.field_hamiltonian = field.compute_hamiltonian().item()
                except:
                    pass

            # Symplectic integrator diagnostics (energy conservation)
            if hasattr(field, '_symplectic_diagnostics'):
                diag = field._symplectic_diagnostics
                metrics.kinetic_energy = diag.get('kinetic_energy', 0.0)
                metrics.potential_energy = diag.get('potential_energy', 0.0)
                metrics.total_hamiltonian_energy = diag.get('total_energy', 0.0)

            if hasattr(field, '_energy_drift'):
                metrics.energy_drift = field._energy_drift
            if hasattr(field, '_energy_drift_percent'):
                metrics.energy_drift_percent = field._energy_drift_percent

            # Entropy gradient (for adaptive updates)
            if hasattr(field, 'T') and field.T.requires_grad and field.T.grad is not None:
                metrics.field_entropy_gradient_norm = torch.norm(field.T.grad).item()

        # Gradient statistics
        total_norm = 0.0
        max_grad_val = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_grad_val = max(max_grad_val, p.grad.abs().max().item())
        metrics.grad_norm = total_norm ** 0.5
        metrics.max_grad = max_grad_val

        # Learning rate
        metrics.learning_rate = optimizer.param_groups[0]['lr']

        # Memory (CUDA only)
        if torch.cuda.is_available():
            metrics.memory_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            metrics.memory_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024

        # Geometric weights (if available)
        if hasattr(model, 'geometric_weights'):
            weights = model.geometric_weights
            if weights is not None:
                metrics.weight_wedge = weights[0].item()
                metrics.weight_tensor = weights[1].item()
                metrics.weight_spinor = weights[2].item()

        if hasattr(model, 'temperature'):
            metrics.temperature = model.temperature.item()

        # Computational complexity detection - based on actual architecture
        d_model = model.d_model if hasattr(model, 'd_model') else 512
        n_layers = model.n_layers if hasattr(model, 'n_layers') else 4
        n_attn = getattr(model, 'n_attention_layers', 0)

        # Check for CausalField/BiQuat (O(N log N) via FFT)
        has_causal_field = hasattr(model, 'geometric_stack') or hasattr(model, 'causal_blocks')
        has_attention = n_attn > 0

        if has_causal_field and not has_attention:
            metrics.complexity = "O(N log N)"  # Pure CausalField FFT
        elif has_causal_field and has_attention:
            metrics.complexity = "O(N log N) + O(N²)"  # CausalField + some attention
        else:
            metrics.complexity = "O(N²)"  # Standard transformer

        # Estimate FLOPs per forward pass
        # CausalField: O(N log N) via FFT - roughly 5 * N * log(N) * d per layer
        # BiQuat blocks: ~8 * d² per layer (no FFN, quaternion ops)
        # Attention: 4 * N² * d per layer (QK^T, softmax, V)

        log_n = math.log2(max(seq_length, 2))

        if "O(N log N)" in metrics.complexity and "O(N²)" not in metrics.complexity:
            # Pure CausalField/BiQuat - O(N log N)
            causal_flops = 5 * batch_size * seq_length * log_n * d_model * n_layers
            biquat_flops = 8 * batch_size * seq_length * (d_model ** 2) * n_layers // d_model  # simplified
            total_flops = causal_flops + biquat_flops
        elif "O(N log N)" in metrics.complexity and "O(N²)" in metrics.complexity:
            # Hybrid: CausalField + attention layers
            n_causal = n_layers - n_attn
            causal_flops = 5 * batch_size * seq_length * log_n * d_model * n_causal
            biquat_flops = 8 * batch_size * seq_length * d_model * n_causal
            attn_flops = 4 * batch_size * (seq_length ** 2) * d_model * n_attn
            ffn_flops = 8 * batch_size * seq_length * (d_model ** 2) * n_attn
            total_flops = causal_flops + biquat_flops + attn_flops + ffn_flops
        else:
            # Standard transformer O(N²)
            attn_flops = 4 * batch_size * (seq_length ** 2) * d_model * n_layers
            ffn_flops = 8 * batch_size * seq_length * (d_model ** 2) * n_layers
            total_flops = attn_flops + ffn_flops

        metrics.flops_estimate = total_flops / 1e9  # GFLOPs
        metrics.flops_per_token = total_flops / (batch_size * seq_length) / 1e6  # MFLOPs/token

        if "O(N²)" in metrics.complexity and "O(N log N)" not in metrics.complexity:
            # Pure O(N²): normalize by N to show per-token scaling
            metrics.base_ops_per_token = (total_flops / (batch_size * seq_length * seq_length)) / 1e6
        elif "O(N log N)" in metrics.complexity:
            # O(N log N): normalize by log(N) to show base cost
            metrics.base_ops_per_token = (total_flops / (batch_size * seq_length * log_n)) / 1e6
        else:
            metrics.base_ops_per_token = metrics.flops_per_token

        # Throughput in TFLOP/s
        if metrics.batch_time > 0:
            metrics.tflops_per_sec = (total_flops * 3) / metrics.batch_time / 1e12

        # MFU calculation
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "A100" in device_name:
                theoretical_peak = 312.0
            elif "4090" in device_name:
                theoretical_peak = 165.0
            elif "3090" in device_name:
                theoretical_peak = 71.0
            elif "V100" in device_name:
                theoretical_peak = 125.0
            else:
                theoretical_peak = 50.0
            metrics.mfu_percent = (metrics.tflops_per_sec / theoretical_peak) * 100

        # Update moving averages
        if self.ema_loss == 0:
            self.ema_loss = metrics.total_loss
        else:
            self.ema_loss = self.smoothing * self.ema_loss + (1 - self.smoothing) * metrics.total_loss
        metrics.avg_loss = self.ema_loss

        if self.ema_batch_time == 0:
            self.ema_batch_time = metrics.batch_time
        else:
            self.ema_batch_time = self.smoothing * self.ema_batch_time + (1 - self.smoothing) * metrics.batch_time
        metrics.avg_batch_time = self.ema_batch_time

        return metrics

    def log_step(self, metrics: TrainingMetrics):
        """Log metrics for a training step."""
        print(f"\n{'='*80}")
        print(f"EPOCH {metrics.epoch} | BATCH {metrics.batch} | STEP {metrics.step}")
        print(f"{'='*80}")

        # Timing
        print(f"\nTIMING:")
        print(f"  Batch time:    {metrics.batch_time:.3f}s")
        print(f"  Throughput:    {metrics.samples_per_second:.1f} samples/s | {metrics.tokens_per_second:.1f} tokens/s")

        # Complexity
        print(f"\nCOMPLEXITY: {metrics.complexity}")
        print(f"  FLOPs/batch:   {metrics.flops_estimate:.2f} GFLOPs")
        print(f"  FLOPs/token:   {metrics.flops_per_token:.2f} MFLOPs")
        print(f"  Throughput:    {metrics.tflops_per_sec:.3f} TFLOP/s")
        if metrics.mfu_percent > 0:
            print(f"  MFU:           {metrics.mfu_percent:.1f}%")

        # Losses
        print(f"\nLOSSES:")
        print(f"  Total: {metrics.total_loss:.6f} (avg: {metrics.avg_loss:.6f})")
        print(f"  LM: {metrics.lm_loss:.6f} | Contrastive: {metrics.contrastive_loss:.6f}")
        print(f"  Geodesic: {metrics.geodesic_cost:.6f} | Entropy: {metrics.field_entropy:.6f}")

        # Field state
        print(f"\nFIELD:")
        print(f"  Alpha: {metrics.field_alpha:.4f} | Nu: {metrics.field_nu_mean:.4f} | Tau: {metrics.field_tau_mean:.4f}")
        if metrics.field_hamiltonian != 0:
            print(f"  Hamiltonian: {metrics.field_hamiltonian:.6f}")

        # Symplectic energy conservation
        if metrics.total_hamiltonian_energy != 0.0:
            print(f"\nENERGY CONSERVATION:")
            print(f"  KE: {metrics.kinetic_energy:.6f} | PE: {metrics.potential_energy:.6f}")
            print(f"  Total: {metrics.total_hamiltonian_energy:.6f} | Drift: {metrics.energy_drift_percent:.2f}%")
            if abs(metrics.energy_drift_percent) > 5.0:
                print(f"  WARNING: Energy drift > 5%!")

        # Gradients
        print(f"\nGRADIENTS:")
        print(f"  Norm: {metrics.grad_norm:.4f} | Max: {metrics.max_grad:.4f} | LR: {metrics.learning_rate:.2e}")

        # Memory
        if metrics.memory_allocated_mb > 0:
            print(f"\nMEMORY: {metrics.memory_allocated_mb:.0f} MB allocated")

        print(f"{'='*80}\n")

        # Add to history
        self.history.append({
            'epoch': metrics.epoch,
            'batch': metrics.batch,
            'step': metrics.step,
            'batch_time': metrics.batch_time,
            'total_loss': metrics.total_loss,
            'lm_loss': metrics.lm_loss,
            'grad_norm': metrics.grad_norm,
            'learning_rate': metrics.learning_rate,
            'tokens_per_second': metrics.tokens_per_second,
            'tflops_per_sec': metrics.tflops_per_sec,
            'mfu_percent': metrics.mfu_percent,
            'energy_drift_percent': metrics.energy_drift_percent,
            'memory_mb': metrics.memory_allocated_mb,
        })

    def save_logs(self, filename: str = "training_log.json"):
        """Save metrics history to JSON file."""
        log_path = self.log_dir / filename
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Logs saved to: {log_path}")

    def print_summary(self):
        """Print summary statistics."""
        if len(self.history) == 0:
            print("No metrics to summarize.")
            return

        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)

        losses = [m['total_loss'] for m in self.history]
        times = [m['batch_time'] for m in self.history]

        print(f"\nTotal steps:       {len(self.history)}")
        print(f"Average loss:      {sum(losses)/len(losses):.6f}")
        print(f"Final loss:        {losses[-1]:.6f}")
        print(f"Best loss:         {min(losses):.6f}")
        print(f"Avg batch time:    {sum(times)/len(times):.3f}s")
        print(f"Total time:        {sum(times):.1f}s")
        print("="*80 + "\n")
