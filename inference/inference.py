"""
Inference Module

Interactive chat interface for running inference on trained models.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, List

from bayesian_cognitive_field.core import CognitiveTensorField, FieldConfig
from bayesian_cognitive_field.inference import (
    GeometricTransformer,
    GeometricTransformerWithMamba,
    DPRKeyValueGenerator
)


class InferenceEngine:
    """
    Inference engine for chat-based interaction with trained models.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize inference engine from checkpoint.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract config
        self.config = checkpoint.get('config', {})
        print(f"✓ Checkpoint loaded: epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")

        # Auto-detect model architecture from state_dict
        model_state = checkpoint.get('model_state_dict', {})
        has_mamba = any('mamba_encoder' in key for key in model_state.keys())
        has_geometric_stack = any('geometric_stack' in key for key in model_state.keys())

        # Override config with detected architecture
        if has_mamba or has_geometric_stack:
            self.config['use_mamba'] = True
            print("✓ Detected Mamba architecture from checkpoint")
        else:
            self.config['use_mamba'] = False
            print("✓ Detected standard Transformer architecture from checkpoint")

        # Infer d_model from state_dict if not in config
        if 'd_model' not in self.config or self.config['d_model'] is None:
            # Look for temporal_projection or output_projection to infer d_model
            for key, param in model_state.items():
                if 'temporal_projection.bias' in key:
                    self.config['d_model'] = param.shape[0]
                    print(f"✓ Inferred d_model={param.shape[0]} from checkpoint")
                    break
                elif 'output_projection.bias' in key:
                    self.config['d_model'] = param.shape[0]
                    print(f"✓ Inferred d_model={param.shape[0]} from checkpoint")
                    break

        # Infer n_layers from state_dict if not in config
        if 'n_layers' not in self.config or self.config['n_layers'] is None:
            # Count mamba layers
            max_layer = -1
            for key in model_state.keys():
                if 'mamba_encoder.layers.' in key:
                    layer_num = int(key.split('mamba_encoder.layers.')[1].split('.')[0])
                    max_layer = max(max_layer, layer_num)
            if max_layer >= 0:
                self.config['n_layers'] = max_layer + 1
                print(f"✓ Inferred n_layers={max_layer + 1} from checkpoint")

        # Initialize field
        field_config = FieldConfig(
            spatial_size=tuple(self.config.get('spatial_size', [8, 8])),
            tensor_dim=self.config.get('field_dim', 16),
            device=device
        )
        self.field = CognitiveTensorField(field_config)

        # Load field state if available
        if 'field_state_dict' in checkpoint:
            self.field.load_state_dict(checkpoint['field_state_dict'])
            print("✓ Field state loaded")

        # Initialize model with detected/inferred parameters
        use_mamba = self.config.get('use_mamba', False)
        d_model = self.config.get('d_model', 512)
        n_layers = self.config.get('n_layers', 6)
        n_heads = self.config.get('n_heads', 8)
        field_dim = self.config.get('field_dim', 16)
        vocab_size = self.config.get('vocab_size', 32000)

        print(f"Model config: use_mamba={use_mamba}, d_model={d_model}, n_layers={n_layers}, field_dim={field_dim}")

        if use_mamba:
            self.model = GeometricTransformerWithMamba(
                d_model=d_model,
                n_mamba_layers=n_layers,
                n_attention_layers=2,
                n_heads=n_heads,
                field_dim=field_dim,
                use_dpr=self.config.get('use_dpr', True),
                use_positional_encoding=True,
                use_temporal_encoding=True
            )
            print(f"✓ Geometric Mamba model initialized")
        else:
            self.model = GeometricTransformer(
                field_dim=field_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                use_positional_encoding=True,
                use_temporal_encoding=True
            )
            print(f"✓ Geometric Transformer model initialized")

        # Load model weights
        if 'model_state_dict' in checkpoint:
            # Filter out dynamically created parameters that don't exist yet
            state_dict = checkpoint['model_state_dict']

            # K_learned and V_learned are dynamically created on first forward pass
            # They should be filtered out during loading
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not k.endswith(('K_learned', 'V_learned'))
            }

            # Load with strict=False to allow missing/extra keys
            missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)

            if missing_keys:
                print(f"WARNING: Missing keys (will be initialized): {missing_keys}")
            if unexpected_keys:
                print(f"WARNING: Unexpected keys (ignored): {unexpected_keys}")

            print("Model weights loaded")

        # Initialize language modeling head (d_model -> vocab_size)
        import torch.nn as nn
        from bayesian_cognitive_field.training.datasets import CognitiveTokenizer

        self.lm_head = nn.Linear(d_model, vocab_size)

        # Load lm_head weights from checkpoint if available
        if 'lm_head_state_dict' in checkpoint:
            self.lm_head.load_state_dict(checkpoint['lm_head_state_dict'])
            print(f"LM head loaded from checkpoint (vocab_size={vocab_size})")
        else:
            print(f"WARNING: LM head not in checkpoint - using random initialization (vocab_size={vocab_size})")

        # Initialize tokenizer for decoding
        self.tokenizer = CognitiveTokenizer(vocab_size=vocab_size)

        # Move model and lm_head to device (field is already on correct device from config)
        self.model = self.model.to(device)
        self.lm_head = self.lm_head.to(device)

        # Set to eval mode
        self.model.eval()
        self.lm_head.eval()

        print(f"✓ Inference engine ready on {device}")

        # Initialize DPR if available
        self.use_dpr = self.config.get('use_dpr', True)
        if self.use_dpr:
            try:
                self.dpr_encoder = DPRKeyValueGenerator(
                    d_model=d_model,
                    freeze_encoders=True,
                    use_pretrained=True
                )
                self.dpr_encoder = self.dpr_encoder.to(device)
                self.dpr_encoder.eval()
                print("✓ DPR encoder initialized")
            except Exception as e:
                print(f"⚠ DPR encoder failed to load: {e}")
                self.use_dpr = False
                self.dpr_encoder = None
        else:
            self.dpr_encoder = None

    def generate(
        self,
        input_text: str,
        max_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response to input text.

        Args:
            input_text: Input prompt/query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        with torch.no_grad():
            # Encode input to get embeddings
            if self.use_dpr and self.dpr_encoder is not None:
                # Use DPR to generate query embeddings
                Q = self.dpr_encoder.generate_q(
                    input_text=input_text,
                    batch_size=1
                )
            else:
                # Fallback: use random embeddings
                d_model = self.config.get('d_model', 512)
                Q = torch.randn(1, 1, d_model, device=self.device)

            # Forward pass through model
            # GeometricTransformerWithMamba expects (x, field_state, time, mask)
            try:
                output, _ = self.model(
                    x=Q,
                    field_state=self.field.T,
                    time=self.field.t,
                    attention_mask=None
                )
            except Exception as e:
                print(f"Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
                return "[ERROR: Model forward pass failed]"

            # Project embeddings to vocabulary logits
            logits = self.lm_head(output)  # (batch, seq_len, vocab_size)

            # Generate tokens (greedy decoding for now)
            # Take the last position in sequence (next token prediction)
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
            next_token_id = torch.argmax(next_token_logits, dim=-1)  # (batch,)

            # Decode token ID to text
            token_id = next_token_id.item()
            decoded_text = self.tokenizer.decode([token_id])

            # Show debug info with full stats
            q_norm = Q.norm().item()
            q_mean = Q.mean().item()
            q_std = Q.std().item()
            output_norm = output.norm().item()
            output_mean = output.mean().item()
            logits_max = next_token_logits.max().item()
            logits_min = next_token_logits.min().item()
            logits_mean = next_token_logits.mean().item()

            response = f"[Input] norm={q_norm:.4f}, mean={q_mean:.4f}, std={q_std:.4f}\n"
            response += f"[Output] norm={output_norm:.4f}, mean={output_mean:.4f}\n"
            response += f"[Logits] max={logits_max:.2f}, min={logits_min:.2f}, mean={logits_mean:.2f}\n"
            response += f"[Token] {token_id} -> '{decoded_text}'"

            # Update field with new observation
            try:
                self.field.evolve_step()
            except Exception as e:
                print(f"Warning: Field evolution failed: {e}")

            return response

    def chat(self):
        """
        Interactive chat loop.
        """
        print("\n" + "=" * 70)
        print("INFERENCE CHAT MODE")
        print("=" * 70)
        print("Type your prompts below. Commands:")
        print("  /exit, /quit - Exit chat")
        print("  /reset - Reset field state")
        print("  /info - Show model info")
        print("=" * 70)

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['/exit', '/quit']:
                    print("\nExiting chat...")
                    break

                if user_input.lower() == '/reset':
                    # Reinitialize field tensor
                    self.field.T = self.field._initialize_field()
                    self.field.t = 0.0
                    self.field.step_count = 0
                    self.field.history.clear()
                    print("✓ Field state reset")
                    continue

                if user_input.lower() == '/info':
                    self._print_info()
                    continue

                # Generate response
                print("\nModel: ", end="", flush=True)
                response = self.generate(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nChat interrupted. Returning to menu...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                continue

    def _print_info(self):
        """Print model and field information."""
        print("\n" + "=" * 70)
        print("MODEL INFO")
        print("=" * 70)
        print(f"Checkpoint: {Path(self.checkpoint_path).name}")
        print(f"Device: {self.device}")
        print(f"Architecture: {'Geometric Mamba' if self.config.get('use_mamba') else 'Transformer'}")
        print(f"d_model: {self.config.get('d_model')}")
        print(f"n_layers: {self.config.get('n_layers')}")
        print(f"n_heads: {self.config.get('n_heads')}")
        print(f"Field dim: {self.config.get('field_dim')}")
        print(f"Spatial size: {self.config.get('spatial_size')}")
        print(f"DPR enabled: {self.use_dpr}")

        # Field statistics
        field_norm = self.field.T.norm().item()
        field_mean = self.field.T.abs().mean().item()
        print(f"\nField norm: {field_norm:.4f}")
        print(f"Field mean: {field_mean:.4f}")
        print("=" * 70)


def load_checkpoint_with_gui() -> Optional[str]:
    """
    Open GUI file dialog to select checkpoint.

    Returns:
        Path to selected checkpoint, or None if cancelled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Create root window (hidden)
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        # Open file dialog
        checkpoint_path = filedialog.askopenfilename(
            title='Select Checkpoint File',
            filetypes=[
                ('PyTorch Checkpoint', '*.pt *.pth'),
                ('All Files', '*.*')
            ],
            initialdir='./checkpoints'
        )

        root.destroy()

        return checkpoint_path if checkpoint_path else None

    except ImportError:
        print("ERROR: tkinter not available. Please install it or provide checkpoint path manually.")
        return None


def main():
    """Main entry point for inference mode."""
    print("\n" + "=" * 70)
    print("BAYESIAN COGNITIVE FIELD - INFERENCE MODE")
    print("=" * 70)

    # Get checkpoint path
    print("\n1. Select checkpoint")
    print("  [1] Browse with GUI")
    print("  [2] Enter path manually")

    choice = input("\nChoice [1-2]: ").strip()

    if choice == '1':
        checkpoint_path = load_checkpoint_with_gui()
        if not checkpoint_path:
            print("No checkpoint selected. Exiting.")
            return
    else:
        checkpoint_path = input("\nCheckpoint path: ").strip()

    # Validate path
    if not Path(checkpoint_path).exists():
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        return

    # Initialize inference engine
    try:
        engine = InferenceEngine(checkpoint_path)
    except Exception as e:
        print(f"\nERROR: Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return

    # Start chat
    engine.chat()


if __name__ == '__main__':
    main()
