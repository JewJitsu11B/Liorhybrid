"""
Mixture-of-Experts (MoE) Framework

A sophisticated MoE framework with:
- Sparse expert activation
- Supervisor-based gating
- Expert constellation coordination
- Librarian deduplication
- Knowledge graph integration
- CUDA-safe optimizations
"""

from .config import MoEConfig
from .expert import BaseExpert
from .supervisor import SupervisorGating
from .constellation import ExpertConstellation
from .librarian import LibrarianCurator
from .knowledge_graph import KnowledgeGraph, PersistentKnowledgeGraph, FastRetrievalKG
from .moe_system import MixtureOfExpertsSystem, FullyOptimizedMoESystem

__all__ = [
    'MoEConfig',
    'BaseExpert',
    'SupervisorGating',
    'ExpertConstellation',
    'LibrarianCurator',
    'KnowledgeGraph',
    'PersistentKnowledgeGraph',
    'FastRetrievalKG',
    'MixtureOfExpertsSystem',
    'FullyOptimizedMoESystem',
]

__version__ = '1.0.0'
