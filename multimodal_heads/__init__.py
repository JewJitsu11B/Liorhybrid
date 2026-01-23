"""
Multimodal Head Modules

Standalone modules for audio, image, video, and cross-modal processing
using the physics framework (ComplexMetricTensor, LiorKernel, CognitiveManifold).

Components:
- AudioCausalHead: Audio processing with LiorKernel temporal memory
- ImageManifoldHead: Image processing with geodesic spatial geometry
- IRVideoHead: Infrared video processing (700-2500nm)
- VisibleVideoHead: Visible spectrum video processing (380-700nm)
- UVVideoHead: Ultraviolet video processing (10-380nm)
- MultispectralVideoFusion: Fuses IR, visible, and UV video
- CrossModalFusion: Multimodal fusion via complex metrics
- RetrievalHead: Geodesic-based retrieval
- TimeSeriesHead: Time series with full LiorKernel capabilities
- GraphReasoningHead: Graph reasoning with parallel transport
- ControlHead: RL/robotics control using Hamiltonian structure
"""

from .audio_head import AudioCausalHead
from .image_head import ImageManifoldHead
from .video_heads import (
    IRVideoHead,
    VisibleVideoHead,
    UVVideoHead,
    MultispectralVideoFusion
)
from .cross_modal import CrossModalFusion
from .retrieval_head import RetrievalHead
from .timeseries_head import TimeSeriesHead
from .graph_reasoning import GraphReasoningHead
from .control_head import ControlHead

__all__ = [
    'AudioCausalHead',
    'ImageManifoldHead',
    'IRVideoHead',
    'VisibleVideoHead',
    'UVVideoHead',
    'MultispectralVideoFusion',
    'CrossModalFusion',
    'RetrievalHead',
    'TimeSeriesHead',
    'GraphReasoningHead',
    'ControlHead',
]

__version__ = '1.0.0'
