"""
Librarian Curator Module

Deduplicates expert reports and integrates into knowledge graph.
CUDA-safe with efficient similarity computations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class LibrarianCurator(nn.Module):
    """
    Librarian module for deduplication and knowledge graph integration.
    
    Responsibilities:
    1. Deduplicate overlapping expert reports
    2. Integrate new insights into knowledge graph
    3. Update nodes, edges, and relationships
    
    CUDA-Safe: Uses efficient similarity computations.
    """
    
    def __init__(self, embedding_dim: int, similarity_threshold: float = 0.85):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Deduplication encoder
        self.dedup_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Integration network for knowledge graph
        self.kg_integrator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def deduplicate_reports(self, draft_reports: List[Dict]) -> List[Dict]:
        """
        Deduplicate expert reports based on semantic similarity.
        
        Args:
            draft_reports: List of draft reports from experts
            
        Returns:
            deduplicated_reports: Non-redundant reports
        """
        if len(draft_reports) <= 1:
            return draft_reports
        
        # Extract summaries
        summaries = torch.stack([r['summary'] for r in draft_reports])
        
        # Encode for similarity comparison
        encoded = self.dedup_encoder(summaries)
        encoded = F.normalize(encoded, p=2, dim=-1)
        
        # Compute pairwise similarity
        similarity_matrix = torch.mm(encoded, encoded.t())
        
        # Identify duplicates (upper triangular, exclude diagonal)
        triu_indices = torch.triu_indices(len(draft_reports), len(draft_reports), offset=1)
        
        # Mark duplicates
        is_duplicate = torch.zeros(len(draft_reports), dtype=torch.bool, device=summaries.device)
        for idx in range(len(triu_indices[0])):
            i, j = triu_indices[0][idx].item(), triu_indices[1][idx].item()
            if similarity_matrix[i, j] > self.similarity_threshold:
                # Mark second report as duplicate
                is_duplicate[j] = True
        
        # Filter out duplicates
        deduplicated_reports = [
            report for idx, report in enumerate(draft_reports)
            if not is_duplicate[idx]
        ]
        
        return deduplicated_reports
    
    def integrate_to_knowledge_graph(self, reports: List[Dict], knowledge_graph: 'KnowledgeGraph') -> None:
        """
        Integrate deduplicated reports into knowledge graph.
        
        Args:
            reports: Deduplicated expert reports
            knowledge_graph: Target knowledge graph
        """
        for report in reports:
            # Extract key information
            summary = report['summary']
            specialization = report['specialization']
            confidence = report['avg_confidence']
            
            # Create or update node
            node_id = knowledge_graph.add_or_update_node(
                embedding=summary,
                node_type='expert_insight',
                metadata={
                    'specialization': specialization,
                    'confidence': confidence,
                    'expert_id': report['expert_id']
                }
            )
            
            # Add edges to related nodes
            related_nodes = knowledge_graph.find_similar_nodes(
                summary, top_k=5, threshold=0.7
            )
            
            for related_id, similarity in related_nodes:
                if related_id != node_id:
                    knowledge_graph.add_edge(
                        node_id, related_id,
                        weight=similarity,
                        edge_type='semantic_similarity'
                    )
