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
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

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
try:
    import torch as _trainer2_torch
    _trainer2_cuda_ok = _trainer2_torch.cuda.is_available()
except Exception as _trainer2_exc:
    _trainer2_torch = None
    _trainer2_cuda_ok = False
    _trainer2_import_error = _trainer2_exc

if _trainer2_cuda_ok:
    from . import trainer2
else:
    if _trainer2_torch is None:
        _trainer2_reason = (
            "trainer2 unavailable: torch import failed "
            f"({str(_trainer2_import_error)})."
        )
    else:
        _trainer2_reason = (
            "trainer2 is CUDA-only and requires torch.cuda.is_available() == True. "
            "Import skipped on a CPU-only host."
        )

    class _Trainer2Unavailable:
        def __getattr__(self, name):
            raise RuntimeError(_trainer2_reason)

        def __repr__(self):
            return f"<trainer2 unavailable: {_trainer2_reason}>"

    trainer2 = _Trainer2Unavailable()
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
    'trainer2',
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
