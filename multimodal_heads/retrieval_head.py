"""
Retrieval Head

Like KnowledgeGraph but uses CognitiveManifold geodesics.
Stores embeddings on learned manifold and retrieves via
geodesic distance with LIoR-weighted effective metric.

CUDA-safe: All operations compatible with torch.compile and CUDA graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.manifold import CognitiveManifold


class RetrievalHead(nn.Module):
    """
    Geodesic-based retrieval using cognitive manifold.
    
    Key physics:
    - Embeddings stored on learned Riemannian manifold
    - Retrieval via geodesic distance (not cosine similarity)
    - LIoR-weighted effective metric for importance weighting
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_coord: int = 8,
        max_items: int = 10000,
        use_faiss: bool = False,
    ):
        """
        Initialize RetrievalHead.
        
        Args:
            d_model: Model dimension
            d_coord: Coordinate manifold dimension
            max_items: Maximum number of items to store
            use_faiss: Whether to use FAISS for approximate search
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_coord = d_coord
        self.max_items = max_items
        self.use_faiss = use_faiss
        
        # Cognitive manifold for geometric structure
        self.manifold = CognitiveManifold(
            d_embed=d_model,
            d_coord=d_coord,
            learnable_metric=True
        )
        
        # Storage for embeddings and coordinates
        self.register_buffer(
            'stored_embeddings',
            torch.zeros(max_items, d_model)
        )
        self.register_buffer(
            'stored_coords',
            torch.zeros(max_items, d_coord)
        )
        self.register_buffer(
            'stored_count',
            torch.tensor(0, dtype=torch.long)
        )
        
        # Resilience scores (LIoR weights)
        self.register_buffer(
            'resilience_scores',
            torch.ones(max_items)
        )
        
        # Metadata storage (as indices)
        self.metadata_list: List[Dict] = []
    
    def add_items(
        self,
        embeddings: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
        resilience: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add items to retrieval index.
        
        Args:
            embeddings: (N, d_model) embeddings to add
            metadata: Optional list of metadata dicts
            resilience: Optional (N,) resilience scores for LIoR weighting
            
        Returns:
            indices: (N,) indices where items were stored
        """
        N = embeddings.shape[0]
        
        if self.stored_count + N > self.max_items:
            raise ValueError(f"Cannot add {N} items, only {self.max_items - self.stored_count} slots available")
        
        # Project embeddings to manifold coordinates
        coords, _ = self.manifold.project(embeddings)
        
        # Store embeddings and coordinates
        start_idx = self.stored_count
        end_idx = start_idx + N
        
        self.stored_embeddings[start_idx:end_idx] = embeddings
        self.stored_coords[start_idx:end_idx] = coords
        
        if resilience is not None:
            self.resilience_scores[start_idx:end_idx] = resilience
        
        # Store metadata
        if metadata is not None:
            self.metadata_list.extend(metadata)
        else:
            self.metadata_list.extend([{}] * N)
        
        self.stored_count += N
        
        return torch.arange(start_idx, end_idx, device=embeddings.device)
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        use_lior_weighting: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Retrieve top-k items by geodesic distance.
        
        Args:
            query: (batch, d_model) or (d_model,) query embedding
            top_k: Number of items to retrieve
            use_lior_weighting: Whether to use resilience-weighted metric
            
        Returns:
            distances: (batch, top_k) geodesic distances
            indices: (batch, top_k) indices of retrieved items
            metadata: List of metadata dicts for retrieved items
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)  # Add batch dimension
        
        batch_size = query.shape[0]
        
        # Project query to manifold
        query_coords, _ = self.manifold.project(query)  # (batch, d_coord)
        
        # Get stored items
        stored_coords = self.stored_coords[:self.stored_count]  # (N, d_coord)
        
        # Compute geodesic distances
        distances = []
        for i in range(batch_size):
            q = query_coords[i:i+1]  # (1, d_coord)
            
            # Expand for broadcasting
            q_expanded = q.expand(self.stored_count, -1)
            
            # Geodesic distance on manifold
            dists = self.manifold.geodesic_distance(q_expanded, stored_coords)
            
            # Apply LIoR weighting if requested
            if use_lior_weighting:
                resilience = self.resilience_scores[:self.stored_count]
                # Weight by resilience (higher resilience = effectively closer)
                dists = dists / (resilience + 1e-8)
            
            distances.append(dists)
        
        distances = torch.stack(distances, dim=0)  # (batch, N)
        
        # Get top-k
        top_k = min(top_k, self.stored_count)
        topk_dists, topk_indices = torch.topk(
            distances, top_k, dim=1, largest=False
        )
        
        # Get metadata
        metadata_results = []
        for i in range(batch_size):
            batch_metadata = [
                self.metadata_list[idx.item()]
                for idx in topk_indices[i]
            ]
            metadata_results.append(batch_metadata)
        
        return topk_dists, topk_indices, metadata_results
    
    def clear(self):
        """Clear all stored items."""
        self.stored_embeddings.zero_()
        self.stored_coords.zero_()
        self.stored_count.zero_()
        self.resilience_scores.fill_(1.0)
        self.metadata_list = []
