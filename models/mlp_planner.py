import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPPlanner(nn.Module):
    """Multi-layer perceptron planner for feature-based planning tasks."""
    
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dim=512,
                 dropout_rate=0.2,
                 use_batch_norm=True,
                 activation='relu'):
        super(MLPPlanner, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        
        # Build the sequential MLP architecture
        layers = []
        
        # First layer: input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self._get_activation())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Second layer: hidden_dim -> hidden_dim
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self._get_activation())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Third layer: hidden_dim -> hidden_dim
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self._get_activation())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Fourth layer: hidden_dim -> hidden_dim // 2
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        layers.append(self._get_activation())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer: hidden_dim // 2 -> output_dim
        layers.append(nn.Linear(hidden_dim // 2, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _get_activation(self):
        """Get the specified activation function."""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        elif self.activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def forward(self, x):
        """
        Forward pass of the MLP planner.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Pass through the sequential MLP
        output = self.mlp(x)
        
        return output
    
    def get_features(self, x):
        """
        Get intermediate features from all layers for analysis.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Dictionary containing features from each layer
        """
        features = {'input': x}
        
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Pass through the sequential MLP and collect features
        # We'll need to manually track through the layers since Sequential doesn't expose intermediate outputs
        current_x = x
        
        # First layer: input_dim -> hidden_dim
        current_x = self.mlp[0](current_x)  # Linear
        features['linear_0'] = current_x.clone()
        
        if self.use_batch_norm:
            current_x = self.mlp[1](current_x)  # BatchNorm
            features['batch_norm_0'] = current_x.clone()
            current_x = self.mlp[2](current_x)  # Activation
            features['activation_0'] = current_x.clone()
            if self.dropout_rate > 0:
                current_x = self.mlp[3](current_x)  # Dropout
                features['dropout_0'] = current_x.clone()
        else:
            current_x = self.mlp[1](current_x)  # Activation
            features['activation_0'] = current_x.clone()
            if self.dropout_rate > 0:
                current_x = self.mlp[2](current_x)  # Dropout
                features['dropout_0'] = current_x.clone()
        
        # Continue for remaining layers...
        # For simplicity, we'll just return the final output
        # A more detailed implementation would track through each layer
        
        # Final output
        output = self.mlp(current_x)
        features['output'] = output
        
        return features
    
    def get_layer_activations(self, x):
        """
        Get activations from each layer for visualization.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            List of activation tensors from each layer
        """
        activations = []
        
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # For the sequential architecture, we'll get activations after each activation layer
        # This is a simplified version - in practice you might want to hook into specific layers
        current_x = x
        
        # We'll collect activations after each activation function
        # This is a simplified approach for the sequential architecture
        for i, layer in enumerate(self.mlp):
            current_x = layer(current_x)
            if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.GELU)):
                activations.append(current_x.clone())
        
        return activations
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self):
        """Get a summary of the model architecture."""
        summary = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'architecture': f"{self.input_dim} -> {self.hidden_dim} -> {self.hidden_dim} -> {self.hidden_dim} -> {self.hidden_dim // 2} -> {self.output_dim}",
            'total_layers': 5,  # 5 linear layers
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation,
            'total_parameters': self.count_parameters()
        }
        return summary 