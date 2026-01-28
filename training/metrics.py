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
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import json
from pathlib import Path

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
"""

import torch
import time
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
    base_ops_per_token: float = 0.0  # Base ops: O(N^2)→per token*N, O(N)→per token
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
                torch.cuda.synchronize()  # Wait for GPU operations to complete
            metrics.step_time = time.time() - self.step_start_time

        if self.batch_start_time is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for GPU operations to complete
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
        metrics.field_alpha = field.alpha.item() if hasattr(field, 'alpha') else 0.0
        metrics.field_nu_mean = field.nu.mean().item() if hasattr(field, 'nu') else 0.0
        metrics.field_tau_mean = field.tau.mean().item() if hasattr(field, 'tau') else 0.0

        # Field energy
        if hasattr(field, 'compute_energy'):
            metrics.field_energy = field.compute_energy()
        # Field Hamiltonian (energy)
        if hasattr(field, 'compute_hamiltonian'):
            metrics.field_hamiltonian = field.compute_hamiltonian().item()
        
        # Symplectic integrator diagnostics (energy conservation)
        if hasattr(field, '_symplectic_diagnostics'):
            diag = field._symplectic_diagnostics
            metrics.kinetic_energy = diag.get('kinetic_energy', 0.0)
            metrics.potential_energy = diag.get('potential_energy', 0.0)
            metrics.total_hamiltonian_energy = diag.get('total_energy', 0.0)
        
        if hasattr(field, '_energy_drift'):
            metrics.energy_drift = field._energy_drift
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

        # Computational complexity
        # Detect architecture type
        if hasattr(model, 'geometric_stack'):
            # Geometric Mamba hybrid
            has_mamba = hasattr(model.geometric_stack, 'mamba_encoder')
            has_attention = hasattr(model.geometric_stack, 'attention_layers')
            if has_mamba and has_attention:
                metrics.complexity = "O(N) + O(N^2)"  # Hybrid
            elif has_mamba:
                metrics.complexity = "O(N)"  # Pure Mamba
            else:
                metrics.complexity = "O(N^2)"  # Pure attention
        else:
            metrics.complexity = "O(N^2)"  # Standard transformer

        # Estimate FLOPs per forward pass
        d_model = model.d_model if hasattr(model, 'd_model') else 512
        n_layers = model.n_layers if hasattr(model, 'n_layers') else 4

        # Attention: 4 * batch * seq^2 * d_model (QK^T, softmax, V)
        # FFN: 8 * batch * seq * d_model^2 (2 linear layers with 4*d_model hidden)
        # Mamba: 6 * batch * seq * d_model (linear complexity, but more ops per token)

        if "O(N)" in metrics.complexity and "O(N^2)" not in metrics.complexity:
            # Pure Mamba
            mamba_flops = 6 * batch_size * seq_length * d_model * n_layers
            total_flops = mamba_flops
        elif "O(N)" in metrics.complexity and "O(N^2)" in metrics.complexity:
            # Hybrid: some Mamba layers + some attention layers
            n_mamba = getattr(model, 'n_mamba_layers', n_layers // 2)
            n_attn = getattr(model, 'n_attention_layers', n_layers // 2)
            mamba_flops = 6 * batch_size * seq_length * d_model * n_mamba
            attn_flops = 4 * batch_size * (seq_length ** 2) * d_model * n_attn
            ffn_flops = 8 * batch_size * seq_length * (d_model ** 2) * n_attn
            total_flops = mamba_flops + attn_flops + ffn_flops
        else:
            # Standard transformer O(N^2)
            attn_flops = 4 * batch_size * (seq_length ** 2) * d_model * n_layers
            ffn_flops = 8 * batch_size * seq_length * (d_model ** 2) * n_layers
            total_flops = attn_flops + ffn_flops

        metrics.flops_estimate = total_flops / 1e9  # GFLOPs

        # FLOPs per token (forward pass only)
        metrics.flops_per_token = total_flops / (batch_size * seq_length) / 1e6  # MFLOPs/token

        # Actual ops scaling per token (what matters for complexity)
        # O(N^2): ops scale linearly with N → show ops/(N*token)
        # O(N): ops constant per token → show ops/token
        if "O(N^2)" in metrics.complexity:
            # For O(N^2), normalize by sequence length to show base cost
            metrics.base_ops_per_token = (total_flops / (batch_size * seq_length * seq_length)) / 1e6  # MFLOPs per (token*N)
        else:
            # For O(N), already constant per token
            metrics.base_ops_per_token = metrics.flops_per_token

        # Throughput in TFLOP/s (multiply by 3 for forward + backward + optimizer)
        if metrics.batch_time > 0:
            metrics.tflops_per_sec = (total_flops * 3) / metrics.batch_time / 1e12
        else:
            metrics.tflops_per_sec = 0.0

        # Model FLOPs Utilization (MFU): actual vs theoretical peak
        # A100: ~312 TFLOPS (FP16), RTX 3090: ~71 TFLOPS, RTX 4090: ~165 TFLOPS
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "A100" in device_name:
                theoretical_peak = 312.0  # TFLOPS
            elif "4090" in device_name or "RTX 4090" in device_name:
                theoretical_peak = 165.0
            elif "3090" in device_name or "RTX 3090" in device_name:
                theoretical_peak = 71.0
            elif "V100" in device_name:
                theoretical_peak = 125.0
            else:
                theoretical_peak = 50.0  # Conservative estimate

            metrics.mfu_percent = (metrics.tflops_per_sec / theoretical_peak) * 100
        else:
            metrics.mfu_percent = 0.0

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
        """
        Log metrics for a training step.

        Args:
            metrics: TrainingMetrics to log
        """
        # Console output
        print(f"\n{'='*80}")
        print(f"EPOCH {metrics.epoch} | BATCH {metrics.batch} | STEP {metrics.step}")
        print(f"{'='*80}")

        # Timing
        print(f"\nTIMING:")
        print(f"  Batch time:    {metrics.batch_time:.3f}s")
        print(f"  Step time:     {metrics.step_time:.3f}s")
        print(f"  Throughput:    {metrics.samples_per_second:.1f} samples/s")
        print(f"                 {metrics.tokens_per_second:.1f} tokens/s")

        # Complexity
        print(f"\nCOMPUTATIONAL COMPLEXITY:")
        print(f"  Complexity:      {metrics.complexity}")
        if "O(N^2)" in metrics.complexity:
            print(f"  Base ops/N:      {metrics.base_ops_per_token*1000:.0f} KFLOPs per (token·N)")
            print(f"  @ N={metrics.seq_length}: {metrics.flops_per_token:.2f} MFLOPs/token")
        else:
            print(f"  Ops/token:       {metrics.flops_per_token:.3f} MFLOPs (constant)")
        print(f"  Throughput:      {metrics.tflops_per_sec:.3f} TFLOP/s")
        if metrics.mfu_percent > 0:
            print(f"  MFU:             {metrics.mfu_percent:.1f}%")

        # Losses
        print(f"\nLOSSES:")
        print(f"  Total:         {metrics.total_loss:.6f} (avg: {metrics.avg_loss:.6f})")
        print(f"  LM Loss:       {metrics.lm_loss:.6f}")
        print(f"  Contrastive:   {metrics.contrastive_loss:.6f}")
        print(f"  Alignment:     {metrics.alignment_loss:.6f}")
        print(f"  Geodesic:      {metrics.geodesic_cost:.6f}")
        print(f"  Field Entropy: {metrics.field_entropy:.6f}")

        # Field state
        print(f"\nFIELD STATE:")
        print(f"  Alpha:         {metrics.field_alpha:.6f}")
        print(f"  Nu (mean):     {metrics.field_nu_mean:.6f}")
        print(f"  Tau (mean):    {metrics.field_tau_mean:.6f}")
        print(f"  Hamiltonian:   {metrics.field_hamiltonian:.6f}")
        print(f"  |∇H|:          {metrics.field_entropy_gradient_norm:.6f}")
        
        # Symplectic energy conservation (if applicable)
        if metrics.total_hamiltonian_energy != 0.0:
            print(f"\nENERGY CONSERVATION (Symplectic):")
            print(f"  Kinetic:       {metrics.kinetic_energy:.6f}")
            print(f"  Potential:     {metrics.potential_energy:.6f}")
            print(f"  Total:         {metrics.total_hamiltonian_energy:.6f}")
            print(f"  Drift:         {metrics.energy_drift:.6f} ({metrics.energy_drift_percent:.2f}%)")
            if abs(metrics.energy_drift_percent) > 5.0:
                print(f"  ⚠ WARNING: Energy drift > 5%")

        # Gradients
        print(f"\nGRADIENTS:")
        print(f"  Norm:          {metrics.grad_norm:.6f}")
        print(f"  Max:           {metrics.max_grad:.6f}")
        print(f"  Learning rate: {metrics.learning_rate:.6e}")

        # Geometric weights
        if metrics.weight_wedge != 0.0:
            print(f"\nGEOMETRIC WEIGHTS:")
            print(f"  Wedge:         {metrics.weight_wedge:.4f}")
            print(f"  Tensor:        {metrics.weight_tensor:.4f}")
            print(f"  Spinor:        {metrics.weight_spinor:.4f}")
            print(f"  Temperature:   {metrics.temperature:.4f}")

        # Memory
        if metrics.memory_allocated_mb > 0:
            print(f"\nMEMORY (CUDA):")
            print(f"  Allocated:     {metrics.memory_allocated_mb:.1f} MB")
            print(f"  Reserved:      {metrics.memory_reserved_mb:.1f} MB")

        print(f"{'='*80}\n")

        # Add to history
        metrics_dict = {
            'epoch': metrics.epoch,
            'batch': metrics.batch,
            'step': metrics.step,
            'batch_time': metrics.batch_time,
            'step_time': metrics.step_time,
            'complexity': metrics.complexity,
            'flops_estimate': metrics.flops_estimate,
            'flops_per_token': metrics.flops_per_token,
            'tflops_per_sec': metrics.tflops_per_sec,
            'mfu_percent': metrics.mfu_percent,
            'total_loss': metrics.total_loss,
            'lm_loss': metrics.lm_loss,
            'contrastive_loss': metrics.contrastive_loss,
            'alignment_loss': metrics.alignment_loss,
            'geodesic_cost': metrics.geodesic_cost,
            'field_entropy': metrics.field_entropy,
            'field_alpha': metrics.field_alpha,
            'field_nu_mean': metrics.field_nu_mean,
            'field_tau_mean': metrics.field_tau_mean,
            'field_hamiltonian': metrics.field_hamiltonian,
            'grad_norm': metrics.grad_norm,
            'learning_rate': metrics.learning_rate,
            'samples_per_second': metrics.samples_per_second,
            'tokens_per_second': metrics.tokens_per_second
        }

        self.history.append(metrics_dict)

    def save_logs(self, filename: str = "training_log.json"):
        """
        Save metrics history to JSON file.

        Args:
            filename: Output filename
        """
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
    base_ops_per_token: float = 0.0  # Base ops: O(N^2)→per token*N, O(N)→per token
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
                torch.cuda.synchronize()  # Wait for GPU operations to complete
            metrics.step_time = time.time() - self.step_start_time

        if self.batch_start_time is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for GPU operations to complete
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
        metrics.field_alpha = field.alpha.item() if hasattr(field, 'alpha') else 0.0
        metrics.field_nu_mean = field.nu.mean().item() if hasattr(field, 'nu') else 0.0
        metrics.field_tau_mean = field.tau.mean().item() if hasattr(field, 'tau') else 0.0

        # Field energy
        if hasattr(field, 'compute_energy'):
            metrics.field_energy = field.compute_energy()

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

        # Computational complexity
        # Detect architecture type
        if hasattr(model, 'geometric_stack'):
            # Geometric Mamba hybrid
            has_mamba = hasattr(model.geometric_stack, 'mamba_encoder')
            has_attention = hasattr(model.geometric_stack, 'attention_layers')
            if has_mamba and has_attention:
                metrics.complexity = "O(N) + O(N^2)"  # Hybrid
            elif has_mamba:
                metrics.complexity = "O(N)"  # Pure Mamba
            else:
                metrics.complexity = "O(N^2)"  # Pure attention
        else:
            metrics.complexity = "O(N^2)"  # Standard transformer

        # Estimate FLOPs per forward pass
        d_model = model.d_model if hasattr(model, 'd_model') else 512
        n_layers = model.n_layers if hasattr(model, 'n_layers') else 4

        # Attention: 4 * batch * seq^2 * d_model (QK^T, softmax, V)
        # FFN: 8 * batch * seq * d_model^2 (2 linear layers with 4*d_model hidden)
        # Mamba: 6 * batch * seq * d_model (linear complexity, but more ops per token)

        if "O(N)" in metrics.complexity and "O(N^2)" not in metrics.complexity:
            # Pure Mamba
            mamba_flops = 6 * batch_size * seq_length * d_model * n_layers
            total_flops = mamba_flops
        elif "O(N)" in metrics.complexity and "O(N^2)" in metrics.complexity:
            # Hybrid: some Mamba layers + some attention layers
            n_mamba = getattr(model, 'n_mamba_layers', n_layers // 2)
            n_attn = getattr(model, 'n_attention_layers', n_layers // 2)
            mamba_flops = 6 * batch_size * seq_length * d_model * n_mamba
            attn_flops = 4 * batch_size * (seq_length ** 2) * d_model * n_attn
            ffn_flops = 8 * batch_size * seq_length * (d_model ** 2) * n_attn
            total_flops = mamba_flops + attn_flops + ffn_flops
        else:
            # Standard transformer O(N^2)
            attn_flops = 4 * batch_size * (seq_length ** 2) * d_model * n_layers
            ffn_flops = 8 * batch_size * seq_length * (d_model ** 2) * n_layers
            total_flops = attn_flops + ffn_flops

        metrics.flops_estimate = total_flops / 1e9  # GFLOPs

        # FLOPs per token (forward pass only)
        metrics.flops_per_token = total_flops / (batch_size * seq_length) / 1e6  # MFLOPs/token

        # Actual ops scaling per token (what matters for complexity)
        # O(N^2): ops scale linearly with N → show ops/(N*token)
        # O(N): ops constant per token → show ops/token
        if "O(N^2)" in metrics.complexity:
            # For O(N^2), normalize by sequence length to show base cost
            metrics.base_ops_per_token = (total_flops / (batch_size * seq_length * seq_length)) / 1e6  # MFLOPs per (token*N)
        else:
            # For O(N), already constant per token
            metrics.base_ops_per_token = metrics.flops_per_token

        # Throughput in TFLOP/s (multiply by 3 for forward + backward + optimizer)
        if metrics.batch_time > 0:
            metrics.tflops_per_sec = (total_flops * 3) / metrics.batch_time / 1e12
        else:
            metrics.tflops_per_sec = 0.0

        # Model FLOPs Utilization (MFU): actual vs theoretical peak
        # A100: ~312 TFLOPS (FP16), RTX 3090: ~71 TFLOPS, RTX 4090: ~165 TFLOPS
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "A100" in device_name:
                theoretical_peak = 312.0  # TFLOPS
            elif "4090" in device_name or "RTX 4090" in device_name:
                theoretical_peak = 165.0
            elif "3090" in device_name or "RTX 3090" in device_name:
                theoretical_peak = 71.0
            elif "V100" in device_name:
                theoretical_peak = 125.0
            else:
                theoretical_peak = 50.0  # Conservative estimate

            metrics.mfu_percent = (metrics.tflops_per_sec / theoretical_peak) * 100
        else:
            metrics.mfu_percent = 0.0

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
        """
        Log metrics for a training step.

        Args:
            metrics: TrainingMetrics to log
        """
        # Console output
        print(f"\n{'='*80}")
        print(f"EPOCH {metrics.epoch} | BATCH {metrics.batch} | STEP {metrics.step}")
        print(f"{'='*80}")

        # Timing
        print(f"\nTIMING:")
        print(f"  Batch time:    {metrics.batch_time:.3f}s")
        print(f"  Step time:     {metrics.step_time:.3f}s")
        print(f"  Throughput:    {metrics.samples_per_second:.1f} samples/s")
        print(f"                 {metrics.tokens_per_second:.1f} tokens/s")

        # Complexity
        print(f"\nCOMPUTATIONAL COMPLEXITY:")
        print(f"  Complexity:      {metrics.complexity}")
        if "O(N^2)" in metrics.complexity:
            print(f"  Base ops/N:      {metrics.base_ops_per_token*1000:.0f} KFLOPs per (token·N)")
            print(f"  @ N={metrics.seq_length}: {metrics.flops_per_token:.2f} MFLOPs/token")
        else:
            print(f"  Ops/token:       {metrics.flops_per_token:.3f} MFLOPs (constant)")
        print(f"  Throughput:      {metrics.tflops_per_sec:.3f} TFLOP/s")
        if metrics.mfu_percent > 0:
            print(f"  MFU:             {metrics.mfu_percent:.1f}%")

        # Losses
        print(f"\nLOSSES:")
        print(f"  Total:         {metrics.total_loss:.6f} (avg: {metrics.avg_loss:.6f})")
        print(f"  LM Loss:       {metrics.lm_loss:.6f}")
        print(f"  Contrastive:   {metrics.contrastive_loss:.6f}")
        print(f"  Alignment:     {metrics.alignment_loss:.6f}")
        print(f"  Geodesic:      {metrics.geodesic_cost:.6f}")
        print(f"  Field Entropy: {metrics.field_entropy:.6f}")

        # Field state
        print(f"\nFIELD STATE:")
        print(f"  Alpha:         {metrics.field_alpha:.6f}")
        print(f"  Nu (mean):     {metrics.field_nu_mean:.6f}")
        print(f"  Tau (mean):    {metrics.field_tau_mean:.6f}")
        print(f"  Hamiltonian:   {metrics.field_hamiltonian:.6f}")
        print(f"  |∇H|:          {metrics.field_entropy_gradient_norm:.6f}")

        # Gradients
        print(f"\nGRADIENTS:")
        print(f"  Norm:          {metrics.grad_norm:.6f}")
        print(f"  Max:           {metrics.max_grad:.6f}")
        print(f"  Learning rate: {metrics.learning_rate:.6e}")

        # Geometric weights
        if metrics.weight_wedge != 0.0:
            print(f"\nGEOMETRIC WEIGHTS:")
            print(f"  Wedge:         {metrics.weight_wedge:.4f}")
            print(f"  Tensor:        {metrics.weight_tensor:.4f}")
            print(f"  Spinor:        {metrics.weight_spinor:.4f}")
            print(f"  Temperature:   {metrics.temperature:.4f}")

        # Memory
        if metrics.memory_allocated_mb > 0:
            print(f"\nMEMORY (CUDA):")
            print(f"  Allocated:     {metrics.memory_allocated_mb:.1f} MB")
            print(f"  Reserved:      {metrics.memory_reserved_mb:.1f} MB")

        print(f"{'='*80}\n")

        # Add to history
        metrics_dict = {
            'epoch': metrics.epoch,
            'batch': metrics.batch,
            'step': metrics.step,
            'batch_time': metrics.batch_time,
            'step_time': metrics.step_time,
            'complexity': metrics.complexity,
            'flops_estimate': metrics.flops_estimate,
            'flops_per_token': metrics.flops_per_token,
            'tflops_per_sec': metrics.tflops_per_sec,
            'mfu_percent': metrics.mfu_percent,
            'total_loss': metrics.total_loss,
            'lm_loss': metrics.lm_loss,
            'contrastive_loss': metrics.contrastive_loss,
            'alignment_loss': metrics.alignment_loss,
            'geodesic_cost': metrics.geodesic_cost,
            'field_entropy': metrics.field_entropy,
            'field_alpha': metrics.field_alpha,
            'field_nu_mean': metrics.field_nu_mean,
            'field_tau_mean': metrics.field_tau_mean,
            'field_hamiltonian': metrics.field_hamiltonian,
            'grad_norm': metrics.grad_norm,
            'learning_rate': metrics.learning_rate,
            'samples_per_second': metrics.samples_per_second,
            'tokens_per_second': metrics.tokens_per_second
        }

        self.history.append(metrics_dict)

    def save_logs(self, filename: str = "training_log.json"):
        """
        Save metrics history to JSON file.

        Args:
            filename: Output filename
        """
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

