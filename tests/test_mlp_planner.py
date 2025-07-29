import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp_planner import MLPPlanner


class TestMLPPlanner:
    """Test cases for MLPPlanner."""
    
    def test_mlp_planner_initialization(self):
        """Test MLP planner initialization."""
        input_dim = 64
        output_dim = 32
        hidden_dim = 128
        
        model = MLPPlanner(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )
        
        assert model.input_dim == input_dim
        assert model.output_dim == output_dim
        assert model.hidden_dim == hidden_dim
        assert hasattr(model, 'mlp')
        assert isinstance(model.mlp, nn.Sequential)
    
    def test_mlp_planner_forward_pass(self):
        """Test MLP planner forward pass."""
        input_dim = 64
        output_dim = 32
        batch_size = 4
        
        model = MLPPlanner(input_dim, output_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = model(x)
        
        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_mlp_planner_single_sample(self):
        """Test MLP planner with single sample input."""
        input_dim = 64
        output_dim = 32
        
        model = MLPPlanner(input_dim, output_dim)
        x = torch.randn(input_dim)  # Single sample
        
        output = model(x)
        
        assert output.shape == (1, output_dim)  # Should be expanded to batch size 1
    
    def test_mlp_planner_different_activations(self):
        """Test MLP planner with different activation functions."""
        input_dim = 64
        output_dim = 32
        batch_size = 4
        
        activations = ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'gelu']
        
        for activation in activations:
            model = MLPPlanner(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=activation
            )
            x = torch.randn(batch_size, input_dim)
            
            output = model(x)
            
            assert output.shape == (batch_size, output_dim)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_mlp_planner_without_batch_norm(self):
        """Test MLP planner without batch normalization."""
        input_dim = 64
        output_dim = 32
        batch_size = 4
        
        model = MLPPlanner(
            input_dim=input_dim,
            output_dim=output_dim,
            use_batch_norm=False
        )
        x = torch.randn(batch_size, input_dim)
        
        output = model(x)
        
        assert output.shape == (batch_size, output_dim)
        
        # Check that batch norm layers are None
        for batch_norm in model.batch_norms:
            assert batch_norm is None
    
    def test_mlp_planner_dropout(self):
        """Test that dropout is applied correctly."""
        input_dim = 64
        output_dim = 32
        batch_size = 4
        dropout_rate = 0.5
        
        model = MLPPlanner(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        )
        x = torch.randn(batch_size, input_dim)
        
        # Training mode
        model.train()
        output_train = model(x)
        
        # Evaluation mode
        model.eval()
        output_eval = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval, atol=1e-6)
    
    def test_mlp_planner_get_features(self):
        """Test getting intermediate features from MLP planner."""
        input_dim = 64
        output_dim = 32
        batch_size = 4
        
        model = MLPPlanner(input_dim, output_dim, hidden_dim=128)
        x = torch.randn(batch_size, input_dim)
        
        features = model.get_features(x)
        
        # Check that basic features are present
        expected_keys = ['input', 'output', 'linear_0', 'activation_0']
        if model.use_batch_norm:
            expected_keys.append('batch_norm_0')
        if model.dropout_rate > 0:
            expected_keys.append('dropout_0')
        
        for key in expected_keys:
            assert key in features
        
        # Check feature shapes
        assert features['input'].shape == (batch_size, input_dim)
        assert features['output'].shape == (batch_size, output_dim)
        assert features['linear_0'].shape == (batch_size, model.hidden_dim)
    
    def test_mlp_planner_get_layer_activations(self):
        """Test getting layer activations from MLP planner."""
        input_dim = 64
        output_dim = 32
        batch_size = 4
        
        model = MLPPlanner(input_dim, output_dim, hidden_dim=128)
        x = torch.randn(batch_size, input_dim)
        
        activations = model.get_layer_activations(x)
        
        # Should have activations for each activation layer
        assert len(activations) > 0
        
        # Check activation shapes (should be hidden_dim or hidden_dim//2)
        for activation in activations:
            assert activation.shape[0] == batch_size
            assert activation.shape[1] in [model.hidden_dim, model.hidden_dim // 2]
    
    def test_mlp_planner_parameter_count(self):
        """Test that parameter count is reasonable."""
        input_dim = 64
        output_dim = 32
        hidden_dim = 128
        
        model = MLPPlanner(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )
        
        total_params = model.count_parameters()
        expected_params = (
            input_dim * hidden_dim + hidden_dim +  # First layer
            hidden_dim * hidden_dim + hidden_dim +  # Second layer
            hidden_dim * hidden_dim + hidden_dim +  # Third layer
            hidden_dim * (hidden_dim // 2) + (hidden_dim // 2) +  # Fourth layer
            (hidden_dim // 2) * output_dim + output_dim  # Output layer
        )
        
        # Add batch norm parameters if used
        if model.use_batch_norm:
            expected_params += (hidden_dim * 3 + (hidden_dim // 2)) * 2  # gamma and beta for each batch norm layer
        
        assert total_params == expected_params
    
    def test_mlp_planner_model_summary(self):
        """Test getting model summary."""
        input_dim = 64
        output_dim = 32
        hidden_dim = 128
        dropout_rate = 0.2
        activation = 'relu'
        
        model = MLPPlanner(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        summary = model.get_model_summary()
        
        assert summary['input_dim'] == input_dim
        assert summary['output_dim'] == output_dim
        assert summary['hidden_dim'] == hidden_dim
        assert summary['dropout_rate'] == dropout_rate
        assert summary['activation'] == activation
        assert summary['use_batch_norm'] == True
        assert summary['total_layers'] == 5
        assert summary['total_parameters'] == model.count_parameters()
    
    def test_mlp_planner_gradient_flow(self):
        """Test that gradients flow through the model."""
        input_dim = 64
        output_dim = 32
        batch_size = 4
        
        model = MLPPlanner(input_dim, output_dim)
        x = torch.randn(batch_size, input_dim, requires_grad=True)
        target = torch.randn(batch_size, output_dim)
        
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that model parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter {name}"
    
    def test_mlp_planner_different_input_sizes(self):
        """Test MLP planner with different input sizes."""
        input_dim = 64
        output_dim = 32
        
        model = MLPPlanner(input_dim, output_dim)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, input_dim)
            output = model(x)
            assert output.shape == (batch_size, output_dim)
    
    def test_mlp_planner_device_transfer(self):
        """Test that model can be moved to different devices."""
        if torch.cuda.is_available():
            input_dim = 64
            output_dim = 32
            batch_size = 4
            
            model = MLPPlanner(input_dim, output_dim)
            x = torch.randn(batch_size, input_dim)
            
            # Move to GPU
            model = model.cuda()
            x = x.cuda()
            
            output = model(x)
            assert output.shape == (batch_size, output_dim)
            assert output.device.type == 'cuda'
            
            # Move back to CPU
            model = model.cpu()
            x = x.cpu()
            
            output = model(x)
            assert output.shape == (batch_size, output_dim)
            assert output.device.type == 'cpu'
    
    def test_mlp_planner_weight_initialization(self):
        """Test that weights are properly initialized."""
        input_dim = 64
        output_dim = 32
        
        model = MLPPlanner(input_dim, output_dim)
        
        # Check that weights are not all zero
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert not torch.allclose(param.data, torch.zeros_like(param.data))
                assert not torch.isnan(param.data).any()
                assert not torch.isinf(param.data).any()
    
    def test_mlp_planner_batch_norm_training_eval(self):
        """Test that batch normalization behaves differently in train/eval modes."""
        input_dim = 64
        output_dim = 32
        batch_size = 4
        
        model = MLPPlanner(input_dim, output_dim, use_batch_norm=True)
        x = torch.randn(batch_size, input_dim)
        
        # Training mode
        model.train()
        output_train = model(x)
        
        # Evaluation mode
        model.eval()
        output_eval = model(x)
        
        # Outputs should be different due to batch norm
        assert not torch.allclose(output_train, output_eval, atol=1e-6)
    
    def test_mlp_planner_invalid_activation(self):
        """Test that invalid activation function raises error."""
        input_dim = 64
        output_dim = 32
        
        with pytest.raises(ValueError, match="Unsupported activation function"):
            model = MLPPlanner(
                input_dim=input_dim,
                output_dim=output_dim,
                activation='invalid_activation'
            )
    
    def test_mlp_planner_small_hidden_dim(self):
        """Test MLP planner with small hidden dimension."""
        input_dim = 64
        output_dim = 32
        hidden_dim = 32  # Small hidden dimension
        
        model = MLPPlanner(input_dim, output_dim, hidden_dim=hidden_dim)
        x = torch.randn(4, input_dim)
        
        output = model(x)
        
        assert output.shape == (4, output_dim)
        assert model.hidden_dim == hidden_dim
    
    def test_mlp_planner_large_network(self):
        """Test MLP planner with large network architecture."""
        input_dim = 128
        output_dim = 64
        hidden_dim = 512
        
        model = MLPPlanner(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )
        x = torch.randn(8, input_dim)
        
        output = model(x)
        
        assert output.shape == (8, output_dim)
        assert model.hidden_dim == hidden_dim
        
        # Check parameter count
        total_params = model.count_parameters()
        assert total_params > 100000  # Should have many parameters
    
    def test_mlp_planner_feature_extraction(self):
        """Test that feature extraction works correctly."""
        input_dim = 64
        output_dim = 32
        batch_size = 4
        
        model = MLPPlanner(input_dim, output_dim, hidden_dim=128)
        x = torch.randn(batch_size, input_dim)
        
        # Get features
        features = model.get_features(x)
        
        # Check that features flow correctly through the network
        # Input should be the original input
        assert torch.allclose(features['input'], x)
        
        # Output should match direct forward pass
        direct_output = model(x)
        assert torch.allclose(features['output'], direct_output)
        
        # Check that activations are non-negative for ReLU
        activation_key = 'activation_0'
        if activation_key in features:
            assert torch.all(features[activation_key] >= 0)


if __name__ == '__main__':
    pytest.main([__file__]) 