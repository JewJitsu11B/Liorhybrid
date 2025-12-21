"""
Cognitive Trainer

Handles training loop for the Bayesian cognitive field with geometric attention.

Two training modes:
1. Geometric-only: Freeze embeddings, train geometric weights + field params
2. Full training: Train everything end-to-end

Features:
- Mixed-precision training (AMP)
- Gradient accumulation
- Checkpointing
- Logging (TensorBoard compatible)
- Multi-GPU support (future)
"""

import os

# Enable expandable memory segments to reduce fragmentation
# (PyTorch reads PYTORCH_CUDA_ALLOC_CONF; keep old var for compatibility but prefer CUDA one.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import json
from typing import Dict, Optional, List
import time
import signal
import sys
import math

from .metrics import MetricsLogger, TrainingMetrics


class CognitiveTrainer:
    """
    Trainer for cognitive field + geometric transformer.

    Args:
        model: GeometricTransformer model
        field: CognitiveTensorField instance
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: PyTorch optimizer
        lr_scheduler: Learning rate scheduler (optional)
        device: Training device
        config: Training configuration dict
    """

    def __init__(
        self,
        model: nn.Module,
        field: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[object] = None,
        device: str = 'cuda',
        config: Optional[Dict] = None,
        tokenizer: Optional[object] = None,
        split_info: Optional[Dict] = None
    ):
        from bayesian_cognitive_field.utils.pipeline_audit import audit_file_once
        audit_file_once("trainer", __file__)

        self.model = model
        self.field = field
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device(device)
        self.tokenizer = tokenizer  # For inference
        self.config = config or {}

        # LIoR controller state (kept as device tensors; avoids GPU→CPU sync in hot path).
        self._lior_controller_state: Dict[str, torch.Tensor] = {}

        # Move models to device
        self.model = self.model.to(self.device)
        self.field.T = self.field.T.to(self.device)

        # Invariant: no trainables are created in the trainer. Attach modules before optimizer construction.
        if not hasattr(self.model, 'input_embedding') or self.model.input_embedding is None:
            raise RuntimeError(
                "Model is missing `input_embedding`. Attach `model.input_embedding` before constructing the optimizer."
            )
        self.input_embedding = self.model.input_embedding.to(self.device)

        if not hasattr(self.model, 'lm_head') or self.model.lm_head is None:
            raise RuntimeError(
                "Model is missing `lm_head`. Attach `model.lm_head` before constructing the optimizer."
            )
        self.lm_head = self.model.lm_head.to(self.device)

        self.training_mode = self.config.get('training_mode', 'full')  # 'geometric' or 'full'
        self.use_lior = self.config.get('use_lior', True)  # Use LIoR-based training
        self.max_epochs = self.config.get('max_epochs', 10)
        self.grad_accum_steps = self.config.get('grad_accum_steps', 1)
        self.use_amp = self.config.get('use_amp', True)
        self.clip_grad_norm = self.config.get('clip_grad_norm', 1.0)
        self.log_interval = self.config.get('log_interval', 10)
        self.eval_interval = self.config.get('eval_interval', 250)
        self.save_interval = self.config.get('save_interval', 500)  # Checkpoint every 500 steps
        self.output_dir = Path(self.config.get('output_dir', './checkpoints'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Early stopping
        self.target_loss = self.config.get('target_loss', None)  # Stop when val_loss < this
        self.patience = self.config.get('patience', 3)  # Epochs without improvement
        self.patience_counter = 0

        # Timing debug (toggle-able)
        self.timing_debug = self.config.get('timing_debug', False)
        if self.timing_debug:
            self._enable_timing_debug()

        # Metrics logger
        log_dir = self.output_dir / 'logs'
        self.metrics_logger = MetricsLogger(
            log_dir=str(log_dir),
            log_interval=self.log_interval,
            smoothing=0.9
        )

        # Freeze parameters if geometric-only mode
        if self.training_mode == 'geometric':
            self._freeze_embeddings()

        # Store split info for checkpoint saving (enables val set recreation)
        self.split_info = split_info

        # GPU memory cleanup thread
        self.gpu_cleanup = None
        if self.config.get('enable_gpu_cleanup', True) and torch.cuda.is_available():
            from .gpu_cleanup import GPUCleanupThread
            self.gpu_cleanup = GPUCleanupThread(
                cleanup_interval_seconds=self.config.get('cleanup_interval_seconds', 30.0),
                cleanup_every_n_steps=self.config.get('cleanup_every_n_steps', None),
                min_memory_threshold_mb=self.config.get('cleanup_memory_threshold_mb', 1000.0),
                verbose=self.config.get('verbose', False)
            )

        # Graceful shutdown handler - save checkpoint on Ctrl+C
        self._setup_signal_handler()

        # Compile model after all submodules are attached (embedding, lm_head, etc.)
        # This ensures compiled graph includes late-bound components.
        if config and config.get('use_compile', True):
            try:
                print("Compiling model with torch.compile (first batch will be slow)...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled successfully")
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}), continuing without compilation")

    def _setup_signal_handler(self):
        """Setup graceful shutdown handler for Ctrl+C."""
        def _handle_interrupt(signum, frame):
            print("\n" + "="*60)
            print("[!] Interrupt received (Ctrl+C)")

            # Run validation if val_loader exists
            if self.val_loader is not None:
                print("[!] Running validation before exit...")
                try:
                    self.model.eval()
                    val_metrics = self.evaluate()
                    print(f"[!] Validation Loss: {val_metrics.get('val_loss', 'N/A'):.4f}")
                    if 'val_lm_loss' in val_metrics:
                        ppl = math.exp(min(val_metrics['val_lm_loss'], 20))
                        print(f"[!] Validation Perplexity: {ppl:.2f}")
                    self.model.train()
                except Exception as e:
                    print(f"[!] Validation failed: {e}")

            print(f"[!] Saving checkpoint at step {self.global_step}...")
            try:
                self.save_checkpoint(f'interrupted_step_{self.global_step}.pt')
                print("[!] Checkpoint saved successfully!")
            except Exception as e:
                print(f"[!] Failed to save checkpoint: {e}")
            print("="*60)
            sys.exit(0)

        # Only set handler on main thread (Windows compatibility)
        try:
            signal.signal(signal.SIGINT, _handle_interrupt)
        except ValueError:
            # Not main thread, skip signal handler
            pass

    def _freeze_embeddings(self):
        """
        Freeze embedding layers for geometric-only training.

        Only trains:
        - Geometric weights in attention layers
        - Field parameters (if adaptive_learning=True)
        """
        print("Freezing embeddings for geometric-only training")

        # Freeze input embeddings
        for param in self.input_embedding.parameters():
            param.requires_grad = False

        # Freeze field-to-KV embeddings
        if hasattr(self.model, 'field_to_kv'):
            for param in self.model.field_to_kv.parameters():
                param.requires_grad = False

        # Keep geometric weights trainable
        for name, param in self.model.named_parameters():
            if 'geometric_weights' in name or 'temperature' in name:
                param.requires_grad = True
                print(f"  Trainable: {name}")

        # Field parameters
        if self.field.config.adaptive_learning:
            self.field.alpha.requires_grad = True
            self.field.nu.requires_grad = True
            self.field.tau.requires_grad = True
            print("  Trainable: field parameters (alpha, nu, tau)")

    def _enable_timing_debug(self):
        """Enable timing debug on model if supported."""
        if hasattr(self.model, 'set_timing_debug'):
            self.model.set_timing_debug(True)
            print("Timing debug enabled")
        else:
            print("Warning: Model does not support timing debug")

    def _disable_timing_debug(self):
        """Disable timing debug on model if supported."""
        if hasattr(self.model, 'set_timing_debug'):
            self.model.set_timing_debug(False)
            print("Timing debug disabled")

    def set_timing_debug(self, enabled: bool):
        """Toggle timing debug at runtime."""
        self.timing_debug = enabled
        if enabled:
            self._enable_timing_debug()
        else:
            self._disable_timing_debug()

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            metrics: Dict of training metrics
        """
        from bayesian_cognitive_field.utils.pipeline_audit import audit_file_once
        audit_file_once("train_epoch", __file__)

        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'lm_loss': [],
            'contrastive_loss': [],
            'alignment_loss': [],
            'field_reg': []
        }

        # Simple iteration without progress bar (preserves history for trending)
        print(f"\n{'='*80}")
        print(f"EPOCH {self.epoch + 1}/{self.max_epochs}")
        print(f"{'='*80}\n")

        for step, batch in enumerate(self.train_loader):
            # Start timing
            self.metrics_logger.start_batch()

            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Get batch info
            batch_size = batch['input_ids'].shape[0] if 'input_ids' in batch else 1
            seq_length = batch['input_ids'].shape[1] if 'input_ids' in batch else 1

            # Forward pass
            self.metrics_logger.start_step()
            t_fwd_start = time.perf_counter()
            loss, loss_dict = self.training_step(batch)
            t_fwd_end = time.perf_counter()

            # Backward pass
            t_bwd_start = time.perf_counter()
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            t_bwd_end = time.perf_counter()

            # Store timing for live display
            loss_dict['fwd_ms'] = (t_fwd_end - t_fwd_start) * 1000
            loss_dict['bwd_ms'] = (t_bwd_end - t_bwd_start) * 1000

            def _as_float(v) -> float:
                if isinstance(v, torch.Tensor):
                    return float(v.detach().float().item())
                return float(v)

            # Gradient accumulation
            if (step + 1) % self.grad_accum_steps == 0:
                # NaN/Inf gradient protection with diagnostic tracking
                nan_count = 0
                total_params = 0
                nan_params = []
                stable_params = []

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        total_params += 1
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            nan_count += 1
                            nan_params.append(name)
                            param.grad = torch.zeros_like(param.grad)
                        else:
                            stable_params.append(name)

                # Log stable AND nan params on first step to diagnose gradient flow
                if self.global_step == 0:
                    print(f"\n=== GRADIENT FLOW DIAGNOSTIC (step 0) ===")
                    print(f"Stable: {len(stable_params)}/{total_params}, NaN: {nan_count}/{total_params}")
                    print(f"\nStable params ({len(stable_params)}):")
                    for p in stable_params:
                        print(f"  + {p}")
                    print(f"\nFirst 20 NaN params (of {nan_count}):")
                    for p in nan_params[:20]:
                        print(f"  - {p}")
                    print(f"=== END DIAGNOSTIC ===\n")

                # Warn if excessive instability
                if nan_count > 0 and nan_count / max(total_params, 1) > 0.1:
                    print(f"WARNING: Excessive NaN/Inf gradients ({nan_count}/{total_params} params at step {self.global_step})")

                # POWER-LAW TEMPORAL DAMPING (Non-Markovian Memory)
                # Based on Causal Informational Field Theory (CI8 framework)
                # S_t = (t+1)^(-α) encodes long-range temporal causality
                # This implements non-local memory: past gradients retain influence
                # indefinitely via power-law decay, not exponential (Markovian) forgetting
                if self.config.get('use_power_law_damping', True):
                    alpha = self.config.get('power_law_alpha', 0.6)
                    power_scale = (self.global_step + 1) ** (-alpha)

                    # Apply to all model parameters
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(power_scale)

                    # Apply to field adaptive parameters if they exist
                    if hasattr(self.field, 'alpha') and hasattr(self.field.alpha, 'grad'):
                        if self.field.alpha.grad is not None:
                            self.field.alpha.grad.mul_(power_scale)
                    if hasattr(self.field, 'nu') and hasattr(self.field.nu, 'grad'):
                        if self.field.nu.grad is not None:
                            self.field.nu.grad.mul_(power_scale)
                    if hasattr(self.field, 'tau') and hasattr(self.field.tau, 'grad'):
                        if self.field.tau.grad is not None:
                            self.field.tau.grad.mul_(power_scale)

                # Clip gradients (after power-law damping)
                if self.clip_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_grad_norm
                    )

                # Comprehensive logging (BEFORE zero_grad so gradients are available)
                if self.global_step % self.log_interval == 0:
                    # Compute comprehensive metrics
                    loss_dict_f = {k: _as_float(v) for k, v in loss_dict.items()}
                    loss_dict_f['total_loss'] = float(loss.detach().float().item())
                    metrics = self.metrics_logger.compute_metrics(
                        model=self.model,
                        field=self.field,
                        optimizer=self.optimizer,
                        loss_dict=loss_dict_f,
                        batch_size=batch_size,
                        seq_length=seq_length
                    )

                    # Set progress info
                    metrics.epoch = self.epoch
                    metrics.batch = step
                    metrics.step = self.global_step

                    # Log comprehensive metrics
                    self.metrics_logger.log_step(metrics)

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.global_step += 1

                # Notify GPU cleanup thread of step completion
                if self.gpu_cleanup is not None:
                    self.gpu_cleanup.notify_step()

            # Track metrics and update progress bar EVERY step
            loss_dict_f = {k: _as_float(v) for k, v in loss_dict.items()}
            loss_val = float(loss.detach().float().item())  # Sync for display (minimal overhead)
            epoch_losses.append(loss_val)

            # Build live stats display
            live_stats = {
                'loss': f'{loss_val:.4f}',
                'step': self.global_step
            }

            # Add component losses if available
            if 'lm_loss' in loss_dict_f:
                live_stats['lm'] = f'{loss_dict_f["lm_loss"]:.3f}'
                # Perplexity = exp(cross_entropy_loss)
                ppl = math.exp(min(loss_dict_f["lm_loss"], 20))  # Cap at exp(20) to avoid overflow
                live_stats['ppl'] = f'{ppl:.1f}'
            if 'contrastive_loss' in loss_dict_f:
                live_stats['cont'] = f'{loss_dict_f["contrastive_loss"]:.3f}'
            if 'geodesic_cost' in loss_dict_f:
                live_stats['geo'] = f'{loss_dict_f["geodesic_cost"]:.3f}'
            if 'field_entropy' in loss_dict_f:
                live_stats['ent'] = f'{loss_dict_f["field_entropy"]:.3f}'

            # Add timing info
            if 'fwd_ms' in loss_dict_f and 'bwd_ms' in loss_dict_f:
                live_stats['fwd'] = f'{loss_dict_f["fwd_ms"]:.0f}ms'
                live_stats['bwd'] = f'{loss_dict_f["bwd_ms"]:.0f}ms'

            # Print stats on new line (preserves history for trending)
            stats_str = " | ".join([f"{k}={v}" for k, v in live_stats.items()])
            print(f"Step {self.global_step:5d} | {stats_str}", flush=True)

            # Store metrics for logging
            for key, value in loss_dict_f.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)

            # Evaluation
            if self.val_loader is not None and self.global_step % self.eval_interval == 0:
                val_metrics = self.evaluate()
                print(f"\nValidation: Loss = {val_metrics['val_loss']:.6f}")
                self.model.train()

            # Checkpointing
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

        # Epoch summary (convert any remaining tensors to floats)
        epoch_losses_cpu = [x.item() if isinstance(x, torch.Tensor) else x for x in epoch_losses]
        summary = {
            'train_loss': sum(epoch_losses_cpu) / len(epoch_losses_cpu) if epoch_losses_cpu else 0.0
        }
        for key, values in epoch_metrics.items():
            if len(values) > 0:
                summary[f'train_{key}'] = sum(values) / len(values)

        return summary

    def training_step(self, batch: Dict) -> tuple:
        """
        Single training step.

        Args:
            batch: Training batch

        Returns:
            loss: Total loss (scalar tensor)
            loss_dict: Dict of loss components (tensors or floats; convert for logging outside forward)
        """
        from bayesian_cognitive_field.utils.pipeline_audit import audit_file_once
        audit_file_once("training_step", __file__)

        # Use LIoR training if enabled
        if self.use_lior:
            return self._training_step_lior(batch)

        # Evolve field (physics simulation)
        # Field evolution happens independently of input
        if self.config.get('evolve_field_during_training', True):
            self.field.evolve_step()

        # Get modality (handle both string and list)
        modality = batch.get('modality', 'text')
        if isinstance(modality, list):
            modality = modality[0]  # All items in batch have same modality

        # Embed inputs using input embedding layer
        if modality == 'text':
            Q_input = self.input_embedding(batch['input_ids'], modality='text')
        elif modality == 'image':
            Q_input = self.input_embedding(batch['image'], modality='image')
        elif modality == 'video':
            Q_input = self.input_embedding(batch['video'], modality='video')
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # Cast embedding output to model dtype (embeddings always output FP32)
        target_dtype = next(self.model.parameters()).dtype
        Q_input = Q_input.to(target_dtype)

        # Forward through geometric transformer
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        # ALWAYS run diagnostic on first step to catch NaN issues
        # Also run every 50 steps if diagnose_nan is enabled
        run_diagnose = (self.global_step == 0) or (self.config.get('diagnose_nan', False) and self.global_step % 50 == 0)
        with torch.amp.autocast(device_type, enabled=self.use_amp):
            output, attn_weights = self.model(
                Q_input,
                self.field.T,
                time=self.field.t,
                diagnose=run_diagnose
            )

            # Project to vocabulary (for language modeling)
            logits = self.model.lm_head(output)

            # Compute losses
            from .losses import combined_loss

            outputs_dict = {
                'logits': logits,
                'text_embedding': output.mean(dim=1)  # Pool for contrastive
            }

            loss, metrics = combined_loss(
                outputs_dict,
                batch,
                loss_weights=self.config.get('loss_weights'),
                field_state=self.field.T
            )

        return loss, metrics

    def _training_step_lior(self, batch: Dict) -> tuple:
        """
        LIoR-based training step with geodesic learning.

        Args:
            batch: Training batch

        Returns:
            loss: Total LIoR loss (scalar tensor)
            loss_dict: Dict of loss components (tensors or floats; convert for logging outside forward)
        """
        from bayesian_cognitive_field.utils.pipeline_audit import audit_file_once
        audit_file_once("training_step_lior", __file__)

        from .lior_trainer import lior_loss, update_adaptive_parameters

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

        # Cast embedding output to model dtype (embeddings always output FP32)
        target_dtype = next(self.model.parameters()).dtype
        Q_input = Q_input.to(target_dtype)

        # Forward through geometric transformer
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type, enabled=self.use_amp):
            output, attn_weights = self.model(
                Q_input,
                self.field.T,
                time=self.field.t,
                diagnose=False
            )

            # Project to vocabulary
            logits = self.model.lm_head(output)

            # Compute LIoR loss with geodesic cost
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
                weights=self.config.get('lior_loss_weights', {
                    'lm': 1.0,
                    'geodesic': 0.1,
                    'contrastive': 0.01,
                    'field_entropy': 0.001
                }),
                controller=self.config.get('lior_controller', {
                    'enabled': True,
                    'mode': 'lm_only',
                    'k': 1.0,
                    'beta': 0.98,
                    'g_min': 0.1,
                    'g_max': 1.0
                }),
                controller_state=self._lior_controller_state
            )

        # Update adaptive parameters via entropy gradient (separate from main backprop)
        if self.field.config.adaptive_learning:
            update_adaptive_parameters(
                field=self.field,
                learning_rate_alpha=self.config.get('lr_alpha', 1e-4),
                learning_rate_nu=self.config.get('lr_nu', 1e-5),
                learning_rate_tau=self.config.get('lr_tau', 1e-5)
            )

        return loss, loss_dict

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Returns:
            metrics: Dict of validation metrics
        """
        self.model.eval()
        val_losses = []
        val_metrics = {
            'lm_loss': [],
            'contrastive_loss': [],
            'alignment_loss': []
        }

        print(f"\nValidating...")
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass (use same precision as training)
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type, enabled=self.use_amp):
                modality = batch.get('modality', 'text')
                if isinstance(modality, list):
                    modality = modality[0]

                if modality == 'text':
                    Q_input = self.input_embedding(batch['input_ids'], modality='text')
                elif modality == 'image':
                    Q_input = self.input_embedding(batch['image'], modality='image')
                elif modality == 'video':
                    Q_input = self.input_embedding(batch['video'], modality='video')

                # Cast embedding output to model dtype (embeddings always output FP32)
                target_dtype = next(self.model.parameters()).dtype
                Q_input = Q_input.to(target_dtype)

                output, _ = self.model(Q_input, self.field.T, time=self.field.t)

                # Compute loss (inside autocast disabled context)
                logits = self.model.lm_head(output)

                from .losses import combined_loss

                outputs_dict = {
                    'logits': logits,
                    'text_embedding': output.mean(dim=1)
                }

                loss, metrics = combined_loss(
                    outputs_dict,
                    batch,
                    field_state=None  # No field reg during eval
                )

            val_losses.append(float(loss.detach().float().item()))
            for key, value in metrics.items():
                if key in val_metrics:
                    if isinstance(value, torch.Tensor):
                        val_metrics[key].append(float(value.detach().float().item()))
                    else:
                        val_metrics[key].append(float(value))

        # Compute averages
        summary = {
            'val_loss': sum(val_losses) / len(val_losses) if len(val_losses) > 0 else 0.0
        }
        for key, values in val_metrics.items():
            if len(values) > 0:
                summary[f'val_{key}'] = sum(values) / len(values)

        return summary

    def train(self):
        """
        Main training loop.
        """
        # Start GPU cleanup thread
        if self.gpu_cleanup is not None:
            self.gpu_cleanup.start()

        print("=" * 70)
        print(f"Starting training: {self.training_mode} mode")
        print(f"Epochs: {self.max_epochs}")
        print("=" * 70)

        # RESOURCE UTILIZATION
        print("=" * 70)
        print("RESOURCE UTILIZATION")
        print("=" * 70)
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            gpu_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
            model_device = next(self.model.parameters()).device
            field_device = self.field.T.device

            print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"Model device: {model_device}")
            print(f"Field device: {field_device}")
        else:
            print(f"DEVICE: CPU (PyTorch cannot see GPU)")
            print(f"Your PyTorch is CPU-only. Reinstall with CUDA support.")

        print(f"DataLoader workers: {self.train_loader.num_workers}")
        print(f"Batch size: {self.train_loader.batch_size}")

        # Handle both regular and streaming datasets
        try:
            num_batches = len(self.train_loader)
            print(f"Batches per epoch: {num_batches}")
        except TypeError:
            # IterableDataset doesn't support len()
            print(f"Batches per epoch: Unknown (streaming dataset)")

        print(f"Mixed precision: {self.use_amp}")
        print(f"Output: {self.output_dir}")
        print("=" * 70)

        for epoch in range(self.max_epochs):
            self.epoch = epoch

            # Reset DPR K/V from evolved field state at start of each epoch
            # This reinitializes K/V parameters based on current field state
            if hasattr(self.model, 'geometric_stack') and hasattr(self.model.geometric_stack, 'reset_kv_from_field'):
                self.model.geometric_stack.reset_kv_from_field(self.field.T)

            # Train epoch
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['train_loss'])

            # Validate
            if self.val_loader is not None:
                val_metrics = self.evaluate()
                self.val_losses.append(val_metrics['val_loss'])

                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pt')

                print(f"\nEpoch {epoch + 1}/{self.max_epochs}:")
                print(f"  Train loss: {train_metrics['train_loss']:.4f}")
                print(f"  Val loss: {val_metrics['val_loss']:.4f}")
            else:
                print(f"\nEpoch {epoch + 1}/{self.max_epochs}:")
                print(f"  Train loss: {train_metrics['train_loss']:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Stop GPU cleanup thread
        if self.gpu_cleanup is not None:
            self.gpu_cleanup.stop()

        # Save logs and print summary
        self.metrics_logger.save_logs(filename=f"training_log_epoch_{self.max_epochs}.json")
        self.metrics_logger.print_summary()

    def save_checkpoint(self, filename: str):
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        from bayesian_cognitive_field.utils.pipeline_audit import audit_file_once
        audit_file_once("save_checkpoint", __file__)

        # Get current loss values
        train_loss = self.train_losses[-1] if self.train_losses else None
        val_loss = self.val_losses[-1] if self.val_losses else None
        lior_loss = getattr(self, 'last_lior_loss', None)
        field_norm = torch.norm(self.field.T).item() if hasattr(self.field, 'T') else None

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lior_loss': lior_loss,
            'field_norm': field_norm,
            'model_state_dict': self.model.state_dict(),
            # Canonical field snapshot for inference/resume.
            'field_state_dict': self.field.state_dict(),
            'field_state': {
                'T': self.field.T,
                'alpha': self.field.alpha,
                'nu': self.field.nu,
                'tau': self.field.tau,
                't': self.field.t,
                'step_count': self.field.step_count,
                'spatial_size': getattr(self.field, 'spatial_size', None),
                'tensor_dim': getattr(self.field, 'tensor_dim', None),
            },
            'model_config': {
                'd_model': getattr(self.model, 'd_model', None),
                'n_layers': getattr(self.model, 'n_layers', None),
                'field_dim': getattr(self.model, 'field_dim', None),
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'lior_controller_state': {k: v.detach() for k, v in self._lior_controller_state.items()},
            'config': self.config
        }

        # Save lm_head for inference/resume
        if hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
            checkpoint['lm_head_state_dict'] = self.model.lm_head.state_dict()
            print("✓ LM head weights saved to checkpoint")

        # Save input_embedding for inference
        if hasattr(self, 'input_embedding') and self.input_embedding is not None:
            checkpoint['input_embedding_state_dict'] = self.input_embedding.state_dict()
            print("✓ Input embedding weights saved to checkpoint")

        # Save tokenizer for inference (required to convert text↔tokens)
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            # Save tokenizer vocab mappings
            checkpoint['tokenizer'] = {
                'vocab_size': self.tokenizer.vocab_size,
                'vocab': self.tokenizer.vocab,
                'inverse_vocab': self.tokenizer.inverse_vocab
            }
            print("✓ Tokenizer saved to checkpoint")

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save split info for recreating validation set from checkpoint
        if self.split_info is not None:
            checkpoint['split_info'] = self.split_info
            print("✓ Split info saved to checkpoint (enables val set recreation)")

        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        Load training checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        from bayesian_cognitive_field.utils.pipeline_audit import audit_file_once
        audit_file_once("load_checkpoint", __file__)

        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore field state
        if 'field_state_dict' in checkpoint:
            self.field.load_state_dict(checkpoint['field_state_dict'])
        else:
            field_state = checkpoint['field_state']
            self.field.T = field_state['T']
            self.field.alpha = field_state['alpha']
            self.field.nu = field_state['nu']
            self.field.tau = field_state['tau']
            self.field.t = field_state['t']
            self.field.step_count = field_state['step_count']

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']

        if 'scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore LIoR controller EMA state if present
        if 'lior_controller_state' in checkpoint:
            self._lior_controller_state = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in checkpoint['lior_controller_state'].items()
            }

        # Restore lm_head if it was saved
        if 'lm_head_state_dict' in checkpoint:
            if not hasattr(self.model, 'lm_head') or self.model.lm_head is None:
                raise RuntimeError("Checkpoint has lm_head weights but model.lm_head is missing.")
            self.model.lm_head.load_state_dict(checkpoint['lm_head_state_dict'])
            print("✓ LM head weights restored from checkpoint")

        # Restore input_embedding if it was saved
        if 'input_embedding_state_dict' in checkpoint:
            if not hasattr(self.model, 'input_embedding') or self.model.input_embedding is None:
                raise RuntimeError("Checkpoint has input_embedding weights but model.input_embedding is missing.")
            self.model.input_embedding.load_state_dict(checkpoint['input_embedding_state_dict'])
            print("✓ Input embedding weights restored from checkpoint")

        # Restore tokenizer if it was saved
        if 'tokenizer' in checkpoint:
            from ..training import CognitiveTokenizer
            tokenizer_data = checkpoint['tokenizer']
            self.tokenizer = CognitiveTokenizer(vocab_size=tokenizer_data['vocab_size'])
            self.tokenizer.vocab = tokenizer_data['vocab']
            self.tokenizer.inverse_vocab = tokenizer_data['inverse_vocab']
            print("✓ Tokenizer restored from checkpoint")

        print(f"Checkpoint loaded: {filepath}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")

