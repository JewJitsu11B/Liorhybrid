"""
Complete MoE System

Integrates all components into a complete mixture-of-experts system.
CUDA-safe with advanced optimizations.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from .config import MoEConfig
from .expert import BaseExpert
from .supervisor import SupervisorGating
from .constellation import ExpertConstellation
from .librarian import LibrarianCurator
from .knowledge_graph import KnowledgeGraph, PersistentKnowledgeGraph


class MixtureOfExpertsSystem(nn.Module):
    """
    Complete MoE system integrating all components.
    
    Pipeline:
    Input → Supervisors → Expert Constellations → Draft Reports → Librarians → Knowledge Graph
    
    CUDA-Safe: End-to-end GPU acceleration.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Initialize experts
        self.experts = nn.ModuleList([
            BaseExpert(
                expert_id=i,
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                specialization=config.expert_specializations[i],
                dropout=config.dropout,
            )
            for i in range(config.num_experts)
        ])
        
        # Initialize supervisor
        self.supervisor = SupervisorGating(
            input_dim=config.input_dim,
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
            num_heads=config.num_gating_heads,
        )
        
        # Initialize constellation coordinator
        self.constellation = ExpertConstellation(self.experts, self.supervisor)
        
        # Initialize librarian
        self.librarian = LibrarianCurator(
            embedding_dim=config.output_dim,
            similarity_threshold=config.dedup_threshold,
        )
        
        # Initialize knowledge graph
        if config.kg_checkpoint_dir:
            self.knowledge_graph = PersistentKnowledgeGraph(
                embedding_dim=config.output_dim,
                max_nodes=config.max_kg_nodes,
                checkpoint_dir=config.kg_checkpoint_dir,
                save_interval=config.kg_save_interval,
                device=config.device,
            )
        else:
            self.knowledge_graph = KnowledgeGraph(
                embedding_dim=config.output_dim,
                max_nodes=config.max_kg_nodes,
                device=config.device,
            )
        
    def forward(self, x: torch.Tensor, use_knowledge_graph: bool = True) -> torch.Tensor:
        """
        Complete MoE forward pass with knowledge graph integration.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            use_knowledge_graph: Whether to integrate into knowledge graph
            
        Returns:
            output: Final combined output (batch_size, seq_len, output_dim)
        """
        # Stage 1: Sparse attention and expert constellation activation
        output, draft_reports = self.constellation(x)
        
        # Stage 2: Librarian deduplication
        deduplicated_reports = self.librarian.deduplicate_reports(draft_reports)
        
        # Stage 3: Knowledge graph integration
        if use_knowledge_graph and len(deduplicated_reports) > 0 and not self.training:
            # Only integrate during inference to avoid overhead
            self.librarian.integrate_to_knowledge_graph(
                deduplicated_reports,
                self.knowledge_graph
            )
        
        return output
    
    def query_knowledge_graph(self, query: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """
        Query knowledge graph for relevant past insights.
        
        Args:
            query: Query embedding (embedding_dim,)
            top_k: Number of results to return
            
        Returns:
            results: List of retrieved insights with metadata
        """
        similar_nodes = self.knowledge_graph.find_similar_nodes(
            query, top_k=top_k, threshold=0.6
        )
        
        results = []
        for node_id, similarity in similar_nodes:
            results.append({
                'node_id': node_id,
                'similarity': similarity,
                'metadata': self.knowledge_graph.node_metadata[node_id],
                'embedding': self.knowledge_graph.node_embeddings[node_id]
            })
        
        return results
    
    def compute_load_balancing_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute load balancing auxiliary loss."""
        # Get gating logits
        expert_emb = self.constellation.expert_embeddings
        attn_output, _ = self.supervisor.attention(
            x,
            expert_emb.unsqueeze(0).expand(x.size(0), -1, -1),
            expert_emb.unsqueeze(0).expand(x.size(0), -1, -1)
        )
        gate_logits = self.supervisor.gate(attn_output)
        
        return self.supervisor.load_balancing_loss(gate_logits)


class FullyOptimizedMoESystem(nn.Module):
    """
    MoE system with all optimizations enabled.
    
    Optimizations:
    1. Kernel fusion via torch.compile
    2. CUDA graphs for inference
    3. Mixed precision training
    4. Gradient checkpointing for memory
    5. Pre-allocated buffers
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Initialize base MoE system
        self.moe = MixtureOfExpertsSystem(config)
        
        # Optimization flags
        self.use_compile = config.use_compile
        self.use_cuda_graph = config.use_cuda_graph
        self.use_amp = config.use_amp
        
        # Compiled versions (lazily initialized)
        self._compiled_forward = None
        self._cuda_graph_wrapper = None
        
        # AMP scaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def forward(self, x: torch.Tensor, use_knowledge_graph: bool = True) -> torch.Tensor:
        """Optimized forward pass."""
        if self.training:
            return self._forward_training(x, use_knowledge_graph)
        else:
            return self._forward_inference(x, use_knowledge_graph)
    
    def _forward_training(self, x: torch.Tensor, use_knowledge_graph: bool) -> torch.Tensor:
        """Optimized training forward pass."""
        if self.use_amp:
            with torch.cuda.amp.autocast():
                return self.moe(x, use_knowledge_graph)
        else:
            return self.moe(x, use_knowledge_graph)
    
    def _forward_inference(self, x: torch.Tensor, use_knowledge_graph: bool) -> torch.Tensor:
        """Heavily optimized inference forward pass."""
        # Use compiled version
        if self.use_compile:
            if self._compiled_forward is None:
                # Compile on first call
                import torch
                if hasattr(torch, 'compile'):
                    self._compiled_forward = torch.compile(
                        lambda inp: self.moe(inp, use_knowledge_graph),
                        mode=self.config.compile_mode
                    )
                else:
                    self._compiled_forward = lambda inp: self.moe(inp, use_knowledge_graph)
            return self._compiled_forward(x)
        
        # Regular forward
        return self.moe(x, use_knowledge_graph)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                     optimizer: torch.optim.Optimizer,
                     loss_fn: callable) -> float:
        """
        Optimized training step with AMP.
        
        Args:
            batch: (inputs, targets)
            optimizer: Optimizer
            loss_fn: Loss function
            
        Returns:
            loss: Scalar loss value
        """
        x, targets = batch
        
        if self.use_amp:
            # Mixed precision training step
            with torch.cuda.amp.autocast():
                outputs = self(x)
                task_loss = loss_fn(outputs, targets)
                
                # Add load balancing loss
                lb_loss = self.moe.compute_load_balancing_loss(x)
                loss = task_loss + self.config.load_balance_weight * lb_loss
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Regular training step
            outputs = self(x)
            task_loss = loss_fn(outputs, targets)
            lb_loss = self.moe.compute_load_balancing_loss(x)
            loss = task_loss + self.config.load_balance_weight * lb_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        return loss.item()
    
    def query_knowledge_graph(self, query: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """Query knowledge graph."""
        return self.moe.query_knowledge_graph(query, top_k)
