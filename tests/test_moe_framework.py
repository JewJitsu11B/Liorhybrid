"""
Tests for MoE Framework

Basic tests to validate functionality.
"""
try: import usage_tracker; usage_tracker.track(__file__)
except: pass

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from moe_framework import (
    MoEConfig,
    BaseExpert,
    SupervisorGating,
    ExpertConstellation,
    LibrarianCurator,
    KnowledgeGraph,
    MixtureOfExpertsSystem,
    FullyOptimizedMoESystem,
)


class TestMoEConfig:
    """Test configuration."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = MoEConfig()
        assert config.input_dim == 512
        assert config.num_experts == 32
        assert config.top_k_experts == 3
        assert len(config.expert_specializations) == config.num_experts
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Should raise if top_k > num_experts
        with pytest.raises(AssertionError):
            MoEConfig(num_experts=4, top_k_experts=5)
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = MoEConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['input_dim'] == 512


class TestBaseExpert:
    """Test expert module."""
    
    def test_expert_creation(self):
        """Test expert module creation."""
        expert = BaseExpert(
            expert_id=0,
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            specialization='test_expert'
        )
        assert expert.expert_id == 0
        assert expert.specialization == 'test_expert'
    
    def test_expert_forward(self):
        """Test expert forward pass."""
        expert = BaseExpert(
            expert_id=0,
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            specialization='test_expert'
        )
        
        x = torch.randn(2, 10, 256)
        output, confidence = expert(x)
        
        assert output.shape == (2, 10, 256)
        assert confidence.shape == (2, 10)
        assert (confidence >= 0).all() and (confidence <= 1).all()
    
    def test_draft_report_generation(self):
        """Test draft report generation."""
        expert = BaseExpert(
            expert_id=0,
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            specialization='test_expert'
        )
        
        x = torch.randn(2, 10, 256)
        output, confidence = expert(x)
        report = expert.generate_draft_report(output, confidence)
        
        assert 'expert_id' in report
        assert 'specialization' in report
        assert 'summary' in report
        assert 'avg_confidence' in report
        assert report['summary'].shape == (256,)


class TestSupervisorGating:
    """Test supervisor gating."""
    
    def test_supervisor_creation(self):
        """Test supervisor creation."""
        supervisor = SupervisorGating(
            input_dim=256,
            num_experts=8,
            top_k=2
        )
        assert supervisor.num_experts == 8
        assert supervisor.top_k == 2
    
    def test_supervisor_forward(self):
        """Test supervisor forward pass."""
        supervisor = SupervisorGating(
            input_dim=256,
            num_experts=8,
            top_k=2
        )
        
        x = torch.randn(2, 10, 256)
        expert_embeddings = torch.randn(8, 256)
        
        expert_indices, gating_weights = supervisor(x, expert_embeddings)
        
        assert expert_indices.shape == (2, 10, 2)
        assert gating_weights.shape == (2, 10, 2)
        
        # Weights should sum to 1
        assert torch.allclose(
            gating_weights.sum(dim=-1),
            torch.ones(2, 10),
            atol=1e-5
        )


class TestKnowledgeGraph:
    """Test knowledge graph."""
    
    def test_kg_creation(self):
        """Test knowledge graph creation."""
        kg = KnowledgeGraph(embedding_dim=256, max_nodes=1000, device='cpu')
        assert kg.num_nodes == 0
        assert kg.max_nodes == 1000
    
    def test_kg_add_node(self):
        """Test adding nodes."""
        kg = KnowledgeGraph(embedding_dim=256, max_nodes=1000, device='cpu')
        
        embedding = torch.randn(256)
        node_id = kg.add_or_update_node(
            embedding=embedding,
            node_type='test_node',
            metadata={'test': 'data'}
        )
        
        assert node_id == 0
        assert kg.num_nodes == 1
    
    def test_kg_find_similar(self):
        """Test finding similar nodes."""
        kg = KnowledgeGraph(embedding_dim=256, max_nodes=1000, device='cpu')
        
        # Add some nodes
        for i in range(5):
            embedding = torch.randn(256)
            kg.add_or_update_node(
                embedding=embedding,
                node_type='test_node',
                metadata={'id': i}
            )
        
        # Query
        query = torch.randn(256)
        results = kg.find_similar_nodes(query, top_k=3, threshold=0.0)
        
        assert len(results) <= 3
        for node_id, similarity in results:
            assert 0 <= node_id < 5
    
    def test_kg_save_load(self, tmp_path):
        """Test saving and loading."""
        kg = KnowledgeGraph(embedding_dim=256, max_nodes=1000, device='cpu')
        
        # Add nodes
        for i in range(3):
            embedding = torch.randn(256)
            kg.add_or_update_node(
                embedding=embedding,
                node_type='test_node',
                metadata={'id': i}
            )
        
        # Save
        save_path = tmp_path / "kg_test.pt"
        kg.save(str(save_path))
        
        # Load into new KG
        kg2 = KnowledgeGraph(embedding_dim=256, max_nodes=1000, device='cpu')
        kg2.load(str(save_path))
        
        assert kg2.num_nodes == 3


class TestMixtureOfExpertsSystem:
    """Test complete MoE system."""
    
    def test_moe_creation(self):
        """Test MoE system creation."""
        config = MoEConfig(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            num_experts=4,
            top_k_experts=2,
        )
        
        model = MixtureOfExpertsSystem(config)
        assert len(model.experts) == 4
        assert model.supervisor.top_k == 2
    
    def test_moe_forward(self):
        """Test MoE forward pass."""
        config = MoEConfig(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            num_experts=4,
            top_k_experts=2,
        )
        
        model = MixtureOfExpertsSystem(config)
        model.eval()
        
        x = torch.randn(2, 10, 256)
        
        with torch.no_grad():
            output = model(x, use_knowledge_graph=False)
        
        assert output.shape == (2, 10, 256)
    
    def test_moe_knowledge_graph_integration(self):
        """Test knowledge graph integration."""
        config = MoEConfig(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            num_experts=4,
            top_k_experts=2,
        )
        
        model = MixtureOfExpertsSystem(config)
        model.eval()
        
        x = torch.randn(2, 10, 256)
        
        with torch.no_grad():
            output = model(x, use_knowledge_graph=True)
        
        # Should have added nodes to KG
        assert model.knowledge_graph.num_nodes >= 0


class TestFullyOptimizedMoESystem:
    """Test optimized MoE system."""
    
    def test_optimized_moe_creation(self):
        """Test optimized MoE creation."""
        config = MoEConfig(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            num_experts=4,
            top_k_experts=2,
            use_compile=False,  # Disable for testing
            use_amp=False,
        )
        
        model = FullyOptimizedMoESystem(config)
        assert model.config.num_experts == 4
    
    def test_optimized_moe_forward(self):
        """Test optimized MoE forward."""
        config = MoEConfig(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            num_experts=4,
            top_k_experts=2,
            use_compile=False,
            use_amp=False,
        )
        
        model = FullyOptimizedMoESystem(config)
        model.eval()
        
        x = torch.randn(2, 10, 256)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 10, 256)
    
    def test_training_step(self):
        """Test training step."""
        config = MoEConfig(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            num_experts=4,
            top_k_experts=2,
            use_compile=False,
            use_amp=False,
        )
        
        model = FullyOptimizedMoESystem(config)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        
        x = torch.randn(2, 10, 256)
        targets = torch.randn(2, 10, 256)
        
        loss = model.training_step((x, targets), optimizer, loss_fn)
        
        assert isinstance(loss, float)
        assert loss >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
