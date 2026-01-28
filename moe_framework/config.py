"""
Configuration for MoE Framework

Defines all configuration parameters for the mixture-of-experts system.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MoEConfig:
    """
    Configuration for Mixture-of-Experts system.
    
    This configuration controls all aspects of the MoE framework including:
    - Model dimensions and architecture
    - Expert configuration and specialization
    - Gating mechanism parameters
    - Librarian deduplication settings
    - Knowledge graph parameters
    - Optimization flags (CUDA-safe)
    """
    
    # Model dimensions
    input_dim: int = 512
    """Input dimension for expert modules"""
    
    hidden_dim: int = 2048
    """Hidden dimension for expert processing"""
    
    output_dim: int = 512
    """Output dimension for expert modules"""
    
    # Expert configuration
    num_experts: int = 32
    """Total number of expert modules"""
    
    top_k_experts: int = 3
    """Number of experts to activate per input (sparse activation)"""
    
    expert_specializations: Optional[List[str]] = None
    """List of specialization names for each expert (optional)"""
    
    # Gating configuration
    gating_type: str = 'topk'
    """Type of gating mechanism: 'topk', 'dense', or 'noisy_topk'"""
    
    load_balance_weight: float = 0.01
    """Weight for load balancing auxiliary loss"""
    
    num_gating_heads: int = 8
    """Number of attention heads in supervisor gating"""
    
    # Librarian configuration
    dedup_threshold: float = 0.85
    """Similarity threshold for deduplication (0-1)"""
    
    # Knowledge graph configuration
    max_kg_nodes: int = 100000
    """Maximum number of nodes in knowledge graph"""
    
    kg_checkpoint_dir: str = './kg_checkpoints'
    """Directory for knowledge graph checkpoints"""
    
    kg_save_interval: int = 1000
    """Save knowledge graph every N node updates"""
    
    use_faiss: bool = True
    """Use FAISS for fast similarity search in knowledge graph"""
    
    # Optimization configuration
    use_compile: bool = True
    """Enable torch.compile for kernel fusion"""
    
    compile_mode: str = 'max-autotune'
    """Torch compile mode: 'default', 'reduce-overhead', 'max-autotune'"""
    
    use_cuda_graph: bool = False
    """Enable CUDA graphs for inference (requires fixed shapes)"""
    
    use_amp: bool = True
    """Enable automatic mixed precision training"""
    
    use_gradient_checkpointing: bool = True
    """Enable gradient checkpointing to save memory"""
    
    # Training configuration
    max_batch_size: int = 32
    """Maximum batch size (for buffer pre-allocation)"""
    
    max_seq_len: int = 512
    """Maximum sequence length (for buffer pre-allocation)"""
    
    dropout: float = 0.1
    """Dropout probability"""
    
    # Device configuration
    device: str = 'cuda'
    """Device to run on: 'cuda' or 'cpu'"""
    
    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Set default expert specializations if not provided
        if self.expert_specializations is None:
            self.expert_specializations = [
                f'expert_{i}' for i in range(self.num_experts)
            ]
        
        # Validation
        assert len(self.expert_specializations) == self.num_experts, \
            f"Number of specializations ({len(self.expert_specializations)}) " \
            f"must match num_experts ({self.num_experts})"
        
        assert self.top_k_experts <= self.num_experts, \
            f"top_k_experts ({self.top_k_experts}) cannot exceed " \
            f"num_experts ({self.num_experts})"
        
        assert 0 < self.dedup_threshold <= 1.0, \
            f"dedup_threshold must be in (0, 1], got {self.dedup_threshold}"
        
        assert self.gating_type in ['topk', 'dense', 'noisy_topk'], \
            f"Invalid gating_type: {self.gating_type}"
        
        assert self.compile_mode in ['default', 'reduce-overhead', 'max-autotune', 
                                     'max-autotune-no-cudagraphs'], \
            f"Invalid compile_mode: {self.compile_mode}"
        
        # Warnings for potentially incompatible settings
        if self.use_cuda_graph and self.compile_mode == 'max-autotune':
            # max-autotune includes cudagraphs, so explicit use_cuda_graph not needed
            pass
        
        if self.use_gradient_checkpointing and self.use_cuda_graph:
            # CUDA graphs don't work well with gradient checkpointing
            import warnings
            warnings.warn(
                "CUDA graphs are not compatible with gradient checkpointing. "
                "use_cuda_graph will be ignored during training."
            )
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_experts': self.num_experts,
            'top_k_experts': self.top_k_experts,
            'expert_specializations': self.expert_specializations,
            'gating_type': self.gating_type,
            'load_balance_weight': self.load_balance_weight,
            'num_gating_heads': self.num_gating_heads,
            'dedup_threshold': self.dedup_threshold,
            'max_kg_nodes': self.max_kg_nodes,
            'kg_checkpoint_dir': self.kg_checkpoint_dir,
            'kg_save_interval': self.kg_save_interval,
            'use_faiss': self.use_faiss,
            'use_compile': self.use_compile,
            'compile_mode': self.compile_mode,
            'use_cuda_graph': self.use_cuda_graph,
            'use_amp': self.use_amp,
            'use_gradient_checkpointing': self.use_gradient_checkpointing,
            'max_batch_size': self.max_batch_size,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'device': self.device,
        }
