"""
Inference Module

Interactive chat interface for running inference on trained models.
"""

import math
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
from pathlib import Path

from Liorhybrid.core import CognitiveTensorField, FieldConfig
from Liorhybrid.inference import (
    GeometricTransformer,
    GeometricTransformerWithMamba,
)
from inference.input_adapters import load_text_from_source
from Liorhybrid.training.tokenizer import CognitiveTokenizer
from Liorhybrid.training.embeddings import MultimodalEmbedding


class InferenceEngine:
    """
    Inference engine for chat-based interaction with trained models.

    Includes SDM (Sparse Distributed Memory) for associative memory retrieval.
    Memory can be used to augment context during generation with retrieved
    similar past contexts.
    
    Integration notes:
    - SDM uses cosine similarity for content-addressable retrieval
    - Input embeddings can be used as query addresses
    - Retrieved values can augment generation context
    - Future: entropy-gated retrieval based on field state
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        enable_memory: bool = False,
        memory_capacity: int = 2048
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.enable_memory = enable_memory

        print(f"Loading checkpoint from: {checkpoint_path}")
        # weights_only=False needed for PyTorch 2.6+ which changed default
        # WARNING: Only load checkpoints from trusted sources due to pickle security risks
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Validate checkpoint schema
        from Liorhybrid.training.checkpoint_validator import validate_checkpoint_schema
        try:
            validate_checkpoint_schema(checkpoint, strict=False)
        except Exception as e:
            print(f"⚠ Checkpoint validation warning: {e}")
            print("⚠ Loading checkpoint anyway - may cause errors if incompatible")
            print()

        self.config = checkpoint.get('config', {})
        print(f"✓ Checkpoint loaded: epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")

        model_state = checkpoint.get('model_state_dict', {})
        has_mamba = any('mamba_encoder' in key for key in model_state.keys())
        has_geometric_stack = any('geometric_stack' in key for key in model_state.keys())

        self.config['use_mamba'] = bool(has_mamba or has_geometric_stack)

        if self.config['use_mamba']:
            print("✓ Detected Mamba architecture from checkpoint")
        else:
            print("✓ Detected standard Transformer architecture from checkpoint")

        # Infer key shapes if missing
        if 'd_model' not in self.config or self.config['d_model'] is None:
            for key, param in model_state.items():
                if 'temporal_projection.bias' in key or 'output_projection.bias' in key:
                    self.config['d_model'] = int(param.shape[0])
                    print(f"✓ Inferred d_model={self.config['d_model']} from checkpoint")
                    break

        if 'n_layers' not in self.config or self.config['n_layers'] is None:
            max_layer = -1
            for key in model_state.keys():
                if 'mamba_encoder.layers.' in key:
                    layer_num = int(key.split('mamba_encoder.layers.')[1].split('.')[0])
                    max_layer = max(max_layer, layer_num)
            if max_layer >= 0:
                self.config['n_layers'] = max_layer + 1
                print(f"✓ Inferred n_layers={self.config['n_layers']} from checkpoint")

        d_model = int(self.config.get('d_model', 512))
        n_layers = int(self.config.get('n_layers', 6))
        n_heads = int(self.config.get('n_heads', 8))
        field_dim = int(self.config.get('field_dim', 16))
        vocab_size = int(self.config.get('vocab_size', 32000))
        max_seq_len = int(self.config.get('max_seq_len', 512))

        # Field
        field_config = FieldConfig(
            spatial_size=tuple(self.config.get('spatial_size', [8, 8])),
            tensor_dim=field_dim,
            device=device
        )
        self.field = CognitiveTensorField(field_config)

        field_state_dict = checkpoint.get('field_state_dict')
        field_state_loaded = False
        if field_state_dict is not None:
            if isinstance(field_state_dict, (list, tuple)) and len(field_state_dict) == 2:
                kind, data = field_state_dict
                if kind == "state_dict" and isinstance(data, dict):
                    field_state_dict = data
                elif kind == "tensor" and torch.is_tensor(data):
                    self.field.T = data.to(device)
                    field_state_dict = None
                    field_state_loaded = True
                elif kind == "snapshot" and isinstance(data, dict):
                    field_state_dict = data
                else:
                    field_state_dict = None
            if isinstance(field_state_dict, dict):
                try:
                    self.field.load_state_dict(field_state_dict, strict=False)
                    print("✓ Field state loaded")
                    field_state_loaded = True
                except Exception as e:
                    print(f"WARNING: field_state_dict load failed ({e})")

        if not field_state_loaded:
            field_state = checkpoint.get('field_state')
            if isinstance(field_state, dict) and 'T' in field_state:
                self.field.T = field_state['T'].to(device)
                print("✓ Field tensor loaded")

        # Model
        if bool(self.config.get('use_mamba', False)):
            self.model = GeometricTransformerWithMamba(
                d_model=d_model,
                n_mamba_layers=n_layers,
                n_attention_layers=int(self.config.get('n_attention_layers', 2)),
                n_heads=n_heads,
                field_dim=field_dim,
                use_dpr=False,  # DPR is intentionally disabled in inference
                use_positional_encoding=True,
                use_temporal_encoding=True
            )
            print("✓ Geometric Mamba model initialized")
        else:
            self.model = GeometricTransformer(
                field_dim=field_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                use_positional_encoding=True,
                use_temporal_encoding=True
            )
            print("✓ Geometric Transformer model initialized")

        # Weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not k.endswith(('K_learned', 'V_learned'))
            }
            self.model.load_state_dict(filtered_state_dict, strict=False)
            print("Model weights loaded")

        # Tokenizer + embeddings + LM head
        self.tokenizer = CognitiveTokenizer(vocab_size=vocab_size)

        self.input_embedding = MultimodalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len
        )
        if 'input_embedding_state_dict' in checkpoint:
            self.input_embedding.load_state_dict(checkpoint['input_embedding_state_dict'])
            print("✓ Input embedding loaded from checkpoint")
        else:
            print("WARNING: input_embedding not in checkpoint - using random initialization")

        self.lm_head = nn.Linear(d_model, vocab_size)
        if 'lm_head_state_dict' in checkpoint:
            self.lm_head.load_state_dict(checkpoint['lm_head_state_dict'])
            print(f"✓ LM head loaded from checkpoint (vocab_size={vocab_size})")
        else:
            print(f"WARNING: LM head not in checkpoint - using random initialization (vocab_size={vocab_size})")

        self.model = self.model.to(device).eval()
        self.input_embedding = self.input_embedding.to(device).eval()
        self.lm_head = self.lm_head.to(device).eval()

        # Initialize SDM memory if enabled
        self.memory = None
        if enable_memory:
            from Liorhybrid.inference.sdm_memory import SDMMemory
            self.memory = SDMMemory(
                capacity=memory_capacity,
                address_dim=d_model,
                value_dim=d_model,
                device=device,
                similarity_threshold=0.5  # Only retrieve fairly similar memories
            )
            print(f"✓ SDM memory initialized (capacity={memory_capacity}, dim={d_model})")

        # Entropy-order controls (collapse-only gating)
        self.nu_inference = float(self.config.get("nu_inference", 1.0))
        self.selector = str(self.config.get("selector", "softmax")).lower()

        print(f"✓ Inference engine ready on {device}")

    def _encode_text_to_ids(self, text: str) -> torch.Tensor:
        max_len = int(self.config.get('max_seq_len', 512))
        ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=max_len)
        if not ids:
            ids = [self.tokenizer.text_token_id, self.tokenizer.eos_token_id]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    @staticmethod
    def _entropy_from_probs(p: torch.Tensor) -> torch.Tensor:
        p = p.clamp(min=1.0e-8)
        return -(p * torch.log(p)).sum(dim=-1)

    def _entropy_gate(self, scores: torch.Tensor, *, entropy: torch.Tensor, nu: float, tau: float) -> torch.Tensor:
        if tau <= 0:
            return scores
        gate = torch.exp(-torch.pow(entropy, nu) / float(tau))
        if gate.dim() == 0:
            gate = gate.view(1, 1)
        elif gate.dim() == 1:
            gate = gate.view(-1, 1)
        return scores * gate

    def _selector_probs(self, scores: torch.Tensor, *, selector: str, tau: float) -> torch.Tensor:
        if selector == "softmax":
            return torch.softmax(scores / float(max(tau, 1.0e-8)), dim=-1)
        if selector == "bornmax":
            energy = torch.square(scores)
            return torch.softmax(energy / float(max(tau, 1.0e-8)), dim=-1)
        if selector == "gibbsmax":
            return torch.softmax(scores / float(max(tau, 1.0e-8)), dim=-1)
        raise ValueError(f"Unknown selector: {selector} (expected softmax|bornmax|gibbsmax)")

    def generate(
        self,
        input_text: str,
        max_tokens: int = 64,
        temperature: float = 0.7
    ) -> str:
        prompt = (input_text or "").strip()
        if not prompt:
            return ""

        max_seq_len = int(self.config.get('max_seq_len', 512))
        eos_id = int(getattr(self.tokenizer, "eos_token_id", 1))
        nu = float(self.nu_inference)
        tau = float(temperature) if temperature is not None else 0.7
        selector = self.selector

        with torch.no_grad():
            input_ids = self._encode_text_to_ids(prompt)
            generated_ids: List[int] = []

            for _ in range(int(max_tokens)):
                if input_ids.shape[1] > max_seq_len:
                    input_ids = input_ids[:, -max_seq_len:]

                # TODO (memory): replace/augment x with SDM retrieval.
                # Example contract (not implemented):
                #   retrieved, confidence = self.memory.query(x, field_state=self.field.T)
                #   if gate_allows_retrieval: x = mix(x, retrieved)
                x = self.input_embedding(input_ids, modality='text')

                output, _ = self.model(
                    x=x,
                    field_state=self.field.T,
                    time=self.field.t,
                    attention_mask=None
                )

                logits = self.lm_head(output)[:, -1, :]  # [1, V]

                p = self._selector_probs(logits, selector=selector, tau=tau)
                H = self._entropy_from_probs(p).mean()  # scalar entropy estimate

                gated_logits = self._entropy_gate(logits, entropy=H, nu=nu, tau=tau)
                p2 = self._selector_probs(gated_logits, selector=selector, tau=tau)

                next_id_t = torch.multinomial(p2, num_samples=1).squeeze(1)
                next_id = int(next_id_t.item())

                generated_ids.append(next_id)

                if next_id == eos_id:
                    break

                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_id]], device=input_ids.device, dtype=input_ids.dtype)],
                    dim=1
                )

                # Dynamics step (kept identical to training semantics)
                self.field.evolve_step()

            continuation = self.tokenizer.decode(generated_ids)
            return prompt + continuation

    def chat(self):
        """Interactive chat loop."""
        print("\n" + "=" * 70)
        print("  INFERENCE CHAT MODE")
        print("=" * 70)
        print("Type your prompts below. Enter 'quit' or 'exit' to stop.")
        print("")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("\nExiting chat mode.")
                    break

                response = self.generate(user_input)
                print(f"\nModel: {response}\n")

            except KeyboardInterrupt:
                print("\n\nExiting chat mode.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue


def load_checkpoint_with_gui() -> str:
    """Open a file dialog to select a checkpoint file."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()  # Hide the main window

        checkpoint_path = filedialog.askopenfilename(
            title="Select Checkpoint",
            filetypes=[
                ("PyTorch Checkpoints", "*.pt *.pth"),
                ("All Files", "*.*")
            ]
        )

        root.destroy()
        return checkpoint_path if checkpoint_path else ""

    except ImportError:
        print("tkinter not available. Please enter path manually.")
        return ""
    except Exception as e:
        print(f"GUI error: {e}. Please enter path manually.")
        return ""
