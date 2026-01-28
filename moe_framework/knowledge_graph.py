"""
Knowledge Graph Module

Persistent knowledge graph for storing and retrieving expert insights.
CUDA-safe with GPU-accelerated similarity search.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import time


class KnowledgeGraph:
    """
    Persistent knowledge graph for storing expert insights.
    
    Structure:
    - Nodes: Expert insights, concepts, relationships
    - Edges: Semantic connections, causal links
    
    CUDA-Safe: GPU-accelerated similarity search.
    """
    
    def __init__(self, embedding_dim: int, max_nodes: int = 100000, device='cuda'):
        self.embedding_dim = embedding_dim
        self.max_nodes = max_nodes
        self.device = device
        
        # Node storage (CUDA tensors)
        self.node_embeddings = torch.zeros(
            max_nodes, embedding_dim,
            dtype=torch.float32, device=device
        )
        
        self.node_metadata = []  # CPU list for metadata
        self.num_nodes = 0
        
        # Edge storage (sparse)
        self.edge_indices = []  # List of (source, target)
        self.edge_weights = []
        self.edge_types = []
        
        # FAISS index for fast search
        self.use_faiss = False
        self._init_faiss()
    
    def _init_faiss(self):
        """Initialize FAISS index if available."""
        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            if torch.cuda.is_available():
                self.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index)
            self.use_faiss = True
        except ImportError:
            pass
    
    def add_or_update_node(self, embedding: torch.Tensor, node_type: str, metadata: Dict) -> int:
        """
        Add new node or update existing similar node.
        
        Args:
            embedding: Node embedding vector
            node_type: Node type
            metadata: Additional metadata
            
        Returns:
            node_id: ID of created or updated node
        """
        # Normalize embedding
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        # Check for existing similar node
        if self.num_nodes > 0:
            similarities = F.cosine_similarity(
                embedding.unsqueeze(0),
                self.node_embeddings[:self.num_nodes],
                dim=-1
            )
            max_sim, max_idx = similarities.max(dim=0)
            
            # Update if very similar (>0.95)
            if max_sim > 0.95:
                alpha = 0.7
                self.node_embeddings[max_idx] = F.normalize(
                    alpha * embedding + (1 - alpha) * self.node_embeddings[max_idx],
                    p=2, dim=-1
                )
                self.node_metadata[max_idx].update(metadata)
                return max_idx.item()
        
        # Add new node
        if self.num_nodes >= self.max_nodes:
            raise RuntimeError("Knowledge graph is full")
        
        node_id = self.num_nodes
        self.node_embeddings[node_id] = embedding
        self.node_metadata.append({'type': node_type, **metadata})
        self.num_nodes += 1
        
        # Update FAISS index
        if self.use_faiss:
            import faiss
            self.faiss_index.add(embedding.cpu().numpy().reshape(1, -1))
        
        return node_id
    
    def find_similar_nodes(self, query_embedding: torch.Tensor, top_k: int = 5,
                          threshold: float = 0.7) -> List[Tuple[int, float]]:
        """Find similar nodes using efficient similarity search."""
        query_embedding = F.normalize(query_embedding, p=2, dim=-1)
        
        if self.num_nodes == 0:
            return []
        
        # Compute similarities
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.node_embeddings[:self.num_nodes],
            dim=-1
        )
        
        # Get top-k
        top_scores, top_indices = torch.topk(
            similarities, k=min(top_k, self.num_nodes)
        )
        
        results = [
            (int(idx), float(score))
            for idx, score in zip(top_indices, top_scores)
            if score >= threshold
        ]
        
        return results
    
    def add_edge(self, source_id: int, target_id: int, weight: float, edge_type: str) -> None:
        """Add edge between two nodes."""
        self.edge_indices.append((source_id, target_id))
        self.edge_weights.append(weight)
        self.edge_types.append(edge_type)
    
    def save(self, path: str) -> None:
        """Save knowledge graph to disk."""
        torch.save({
            'node_embeddings': self.node_embeddings[:self.num_nodes].cpu(),
            'node_metadata': self.node_metadata,
            'num_nodes': self.num_nodes,
            'edge_indices': self.edge_indices,
            'edge_weights': self.edge_weights,
            'edge_types': self.edge_types,
        }, path)
    
    def load(self, path: str) -> None:
        """Load knowledge graph from disk."""
        checkpoint = torch.load(path)
        self.num_nodes = checkpoint['num_nodes']
        self.node_embeddings[:self.num_nodes] = checkpoint['node_embeddings'].to(self.device)
        self.node_metadata = checkpoint['node_metadata']
        self.edge_indices = checkpoint['edge_indices']
        self.edge_weights = checkpoint['edge_weights']
        self.edge_types = checkpoint['edge_types']


class PersistentKnowledgeGraph(KnowledgeGraph):
    """Knowledge graph with automatic checkpointing."""
    
    def __init__(self, *args, checkpoint_dir='./kg_checkpoints', save_interval=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_interval = save_interval
        self.save_counter = 0
    
    def add_or_update_node(self, *args, **kwargs):
        """Add node with auto-save."""
        node_id = super().add_or_update_node(*args, **kwargs)
        
        self.save_counter += 1
        if self.save_counter >= self.save_interval:
            self.incremental_save()
            self.save_counter = 0
        
        return node_id
    
    def incremental_save(self):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'kg_checkpoint_{time.time()}.pt'
        self.save(str(checkpoint_path))
        
        # Keep only last 5 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('kg_checkpoint_*.pt'))
        for old_ckpt in checkpoints[:-5]:
            old_ckpt.unlink()


class FastRetrievalKG(KnowledgeGraph):
    """Knowledge graph with optimized retrieval."""
    pass  # Future: Add advanced indexing
