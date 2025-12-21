"""
Training Infrastructure

Provides tokenization, embeddings, datasets, and training loops for
the Bayesian cognitive field with geometric attention.

Supports two training modes:
1. Geometric-only: Train geometric weights, freeze embeddings
2. Full training: Train everything end-to-end (embeddings + geometric + field)

Multimodal support:
- Text: BPE tokenization, learned embeddings
- Images: ViT-style patch embeddings
- Video: Frame sampling + temporal encoding
"""

from .tokenizer import CognitiveTokenizer
from .embeddings import MultimodalEmbedding
from .datasets import (
    TextDataset,
    ChunkedTextDataset,
    ImageTextDataset,
    VideoTextDataset,
    MultimodalDataset
)
from .losses import (
    language_modeling_loss,
    contrastive_loss,
    multimodal_alignment_loss
)
from .trainer import CognitiveTrainer
from .metrics import TrainingMetrics, MetricsLogger
from .lior_trainer import (
    compute_geodesic_cost,
    compute_field_entropy,
    update_adaptive_parameters,
    lior_loss,
    LIoRTrainingMixin
)
from .file_reader import (
    UniversalFileReader,
    open_file_dialog,
    open_multiple_files_dialog,
    read_file_with_dialog
)
from .biquat_optimizer import BiquatOptimizer
from .lior_optimizer import LIoROptimizer, LIoRManifoldOptimizer
from .gpu_cleanup import (
    GPUCleanupThread,
    cleanup_gpu_memory,
    enable_expandable_segments,
    check_cuda_alloc_conf
)
from .checkpoint_utils import (
    inspect_checkpoint,
    print_checkpoint_summary,
    compare_checkpoints,
    find_best_checkpoint,
    get_split_info,
    recreate_splits,
    run_validation_from_checkpoint
)

__all__ = [
    'CognitiveTokenizer',
    'MultimodalEmbedding',
    'TextDataset',
    'ChunkedTextDataset',
    'ImageTextDataset',
    'VideoTextDataset',
    'MultimodalDataset',
    'language_modeling_loss',
    'contrastive_loss',
    'multimodal_alignment_loss',
    'CognitiveTrainer',
    'TrainingMetrics',
    'MetricsLogger',
    'compute_geodesic_cost',
    'compute_field_entropy',
    'update_adaptive_parameters',
    'lior_loss',
    'LIoRTrainingMixin',
    'UniversalFileReader',
    'open_file_dialog',
    'open_multiple_files_dialog',
    'read_file_with_dialog',
    'BiquatOptimizer',
    # LIoR optimizers
    'LIoROptimizer',
    'LIoRManifoldOptimizer',
    # GPU cleanup
    'GPUCleanupThread',
    'cleanup_gpu_memory',
    'enable_expandable_segments',
    'check_cuda_alloc_conf',
    # Checkpoint utilities
    'inspect_checkpoint',
    'print_checkpoint_summary',
    'compare_checkpoints',
    'find_best_checkpoint',
    'get_split_info',
    'recreate_splits',
    'run_validation_from_checkpoint',
]
