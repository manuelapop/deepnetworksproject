import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_planner import (
    TransformerPlanner, 
    PositionalEncoding, 
    MultiHeadAttention, 
    TransformerBlock
)


class TestPositionalEncoding:
    """Test cases for PositionalEncoding."""
    
    def test_positional_encoding_shape(self):
        """Test that positional encoding has correct shape."""
        d_model = 64
        max_len = 100
        pe = PositionalEncoding(d_model, max_len)
        
        x = torch.randn(50, 32, d_model)  # (seq_len, batch_size, d_model)
        output = pe(x)
        
        assert output.shape == x.shape
        assert output.shape == (50, 32, d_model)
    
    def test_positional_encoding_values(self):
        """Test that positional encoding adds position information."""
        d_model = 64
        max_len = 100
        pe = PositionalEncoding(d_model, max_len)
        
        x = torch.zeros(10, 1, d_model)
        output = pe(x)
        
        # Output should not be zero (positional encoding added)
        assert not torch.allclose(output, x)
        
        # Different positions should have different encodings
        pos1 = pe(torch.zeros(1, 1, d_model))
        pos2 = pe(torch.zeros(2, 1, d_model))
        assert not torch.allclose(pos1[0], pos2[1])


class TestMultiHeadAttention:
    """Test cases for MultiHeadAttention."""
    
    def test_multi_head_attention_shape(self):
        """Test that multi-head attention has correct output shape."""
        d_model = 64
        num_heads = 8
        batch_size = 4
        seq_len = 10
        
        attention = MultiHeadAttention(d_model, num_heads)
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    def test_multi_head_attention_mask(self):
        """Test that attention mask works correctly."""
        d_model = 64
        num_heads = 8
        batch_size = 2
        seq_len = 10
        
        attention = MultiHeadAttention(d_model, num_heads)
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        # Create mask where first 5 positions are valid, rest are masked
        mask = torch.zeros(batch_size, 1, 1, seq_len)
        mask[:, :, :, :5] = 1
        
        output, attention_weights = attention(query, key, value, mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
        # Attention weights for masked positions should be zero
        assert torch.allclose(attention_weights[:, :, :, 5:], torch.zeros_like(attention_weights[:, :, :, 5:]))
    
    def test_multi_head_attention_heads(self):
        """Test that different heads learn different attention patterns."""
        d_model = 64
        num_heads = 8
        batch_size = 1
        seq_len = 5
        
        attention = MultiHeadAttention(d_model, num_heads)
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention(query, key, value)
        
        # Check that different heads have different attention patterns
        head_weights = attention_weights[0]  # (num_heads, seq_len, seq_len)
        
        # Compare first two heads
        head1 = head_weights[0]
        head2 = head_weights[1]
        
        # They should be different (not identical)
        assert not torch.allclose(head1, head2, atol=1e-6)


class TestTransformerBlock:
    """Test cases for TransformerBlock."""
    
    def test_transformer_block_shape(self):
        """Test that transformer block has correct output shape."""
        d_model = 64
        num_heads = 8
        d_ff = 256
        batch_size = 4
        seq_len = 10
        
        block = TransformerBlock(d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = block(x)
        
        assert output.shape == x.shape
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_transformer_block_residual_connection(self):
        """Test that residual connections work correctly."""
        d_model = 64
        num_heads = 8
        d_ff = 256
        batch_size = 2
        seq_len = 5
        
        block = TransformerBlock(d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Set dropout to 0 for deterministic output
        block.dropout.p = 0.0
        
        output = block(x)
        
        # Output should be different from input due to transformations
        assert not torch.allclose(output, x)
        
        # But should maintain the same shape
        assert output.shape == x.shape


class TestTransformerPlanner:
    """Test cases for TransformerPlanner."""
    
    def test_transformer_planner_initialization(self):
        """Test transformer planner initialization."""
        input_dim = 64
        output_dim = 32
        d_model = 128
        num_heads = 8
        num_layers = 4
        
        model = TransformerPlanner(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        assert model.input_dim == input_dim
        assert model.output_dim == output_dim
        assert model.d_model == d_model
        assert len(model.transformer_blocks) == num_layers
    
    def test_transformer_planner_forward_pass(self):
        """Test transformer planner forward pass."""
        input_dim = 64
        output_dim = 32
        batch_size = 4
        seq_len = 10
        
        model = TransformerPlanner(input_dim, output_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, output_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_transformer_planner_with_sequence_lengths(self):
        """Test transformer planner with variable sequence lengths."""
        input_dim = 64
        output_dim = 32
        batch_size = 3
        seq_len = 10
        
        model = TransformerPlanner(input_dim, output_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        seq_lengths = [7, 5, 10]  # Different sequence lengths
        
        output = model(x, seq_lengths)
        
        assert output.shape == (batch_size, seq_len, output_dim)
    
    def test_transformer_planner_attention_weights(self):
        """Test getting attention weights from transformer planner."""
        input_dim = 64
        output_dim = 32
        batch_size = 2
        seq_len = 8
        
        model = TransformerPlanner(input_dim, output_dim, num_layers=2)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output, attention_weights = model.get_attention_weights(x)
        
        assert output.shape == (batch_size, seq_len, output_dim)
        assert len(attention_weights) == 2  # One for each layer
        assert attention_weights[0].shape == (batch_size, model.transformer_blocks[0].attention.num_heads, seq_len, seq_len)
    
    def test_transformer_planner_parameter_count(self):
        """Test that parameter count is reasonable."""
        input_dim = 64
        output_dim = 32
        d_model = 128
        num_heads = 8
        num_layers = 4
        
        model = TransformerPlanner(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # All parameters should be trainable
    
    def test_transformer_planner_gradient_flow(self):
        """Test that gradients flow through the model."""
        input_dim = 64
        output_dim = 32
        batch_size = 2
        seq_len = 5
        
        model = TransformerPlanner(input_dim, output_dim)
        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        target = torch.randn(batch_size, seq_len, output_dim)
        
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
    
    def test_transformer_planner_different_input_sizes(self):
        """Test transformer planner with different input sizes."""
        input_dim = 64
        output_dim = 32
        
        model = TransformerPlanner(input_dim, output_dim)
        
        # Test different batch sizes
        for batch_size in [1, 4, 8]:
            for seq_len in [5, 10, 15]:
                x = torch.randn(batch_size, seq_len, input_dim)
                output = model(x)
                assert output.shape == (batch_size, seq_len, output_dim)
    
    def test_transformer_planner_device_transfer(self):
        """Test that model can be moved to different devices."""
        if torch.cuda.is_available():
            input_dim = 64
            output_dim = 32
            batch_size = 2
            seq_len = 5
            
            model = TransformerPlanner(input_dim, output_dim)
            x = torch.randn(batch_size, seq_len, input_dim)
            
            # Move to GPU
            model = model.cuda()
            x = x.cuda()
            
            output = model(x)
            assert output.shape == (batch_size, seq_len, output_dim)
            assert output.device.type == 'cuda'
            
            # Move back to CPU
            model = model.cpu()
            x = x.cpu()
            
            output = model(x)
            assert output.shape == (batch_size, seq_len, output_dim)
            assert output.device.type == 'cpu'
    
    def test_transformer_planner_dropout(self):
        """Test that dropout is applied during training."""
        input_dim = 64
        output_dim = 32
        batch_size = 2
        seq_len = 5
        
        model = TransformerPlanner(input_dim, output_dim, dropout=0.5)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Training mode
        model.train()
        output_train = model(x)
        
        # Evaluation mode
        model.eval()
        output_eval = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train, output_eval, atol=1e-6)
    
    def test_transformer_planner_weight_initialization(self):
        """Test that weights are properly initialized."""
        input_dim = 64
        output_dim = 32
        
        model = TransformerPlanner(input_dim, output_dim)
        
        # Check that weights are not all zero
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert not torch.allclose(param.data, torch.zeros_like(param.data))
                assert not torch.isnan(param.data).any()
                assert not torch.isinf(param.data).any()


if __name__ == '__main__':
    pytest.main([__file__]) 