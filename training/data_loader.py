import torch
import torch.utils.data as data
import numpy as np
import random
from typing import Tuple, List, Optional, Union


class SyntheticDataset(data.Dataset):
    """Synthetic dataset for testing and training planners."""
    
    def __init__(self, 
                 num_samples=1000, 
                 input_dim=64, 
                 output_dim=32, 
                 seq_length=20,
                 noise_level=0.1,
                 task_type='regression'):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            input_dim: Input dimension
            output_dim: Output dimension
            seq_length: Sequence length for transformer data
            noise_level: Noise level for synthetic data
            task_type: Type of task ('regression' or 'classification')
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.noise_level = noise_level
        self.task_type = task_type
        
        # Generate synthetic data
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic data based on task type."""
        if self.task_type == 'regression':
            self._generate_regression_data()
        elif self.task_type == 'classification':
            self._generate_classification_data()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def _generate_regression_data(self):
        """Generate regression data with some underlying patterns."""
        # Generate random input data
        self.inputs = torch.randn(self.num_samples, self.input_dim)
        
        # Create some underlying patterns for the outputs
        # Use a combination of linear and non-linear transformations
        linear_component = torch.matmul(self.inputs, torch.randn(self.input_dim, self.output_dim))
        non_linear_component = torch.sin(self.inputs[:, :self.output_dim]) + torch.cos(self.inputs[:, :self.output_dim])
        
        # Combine components
        self.outputs = linear_component + non_linear_component
        
        # Add noise
        noise = torch.randn_like(self.outputs) * self.noise_level
        self.outputs += noise
    
    def _generate_classification_data(self):
        """Generate classification data."""
        # Generate random input data
        self.inputs = torch.randn(self.num_samples, self.input_dim)
        
        # Create classification targets based on input patterns
        # Use a simple rule: classify based on the sign of the first few dimensions
        decision_boundary = torch.sum(self.inputs[:, :5], dim=1)
        self.outputs = (decision_boundary > 0).long()
        
        # Convert to one-hot encoding
        self.outputs = torch.nn.functional.one_hot(self.outputs, num_classes=self.output_dim).float()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class SequenceDataset(data.Dataset):
    """Dataset for sequence-based planning tasks (for transformer)."""
    
    def __init__(self, 
                 num_samples=1000, 
                 input_dim=64, 
                 output_dim=32, 
                 seq_length=20,
                 noise_level=0.1):
        """
        Initialize sequence dataset.
        
        Args:
            num_samples: Number of samples to generate
            input_dim: Input dimension per timestep
            output_dim: Output dimension per timestep
            seq_length: Sequence length
            noise_level: Noise level for synthetic data
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.noise_level = noise_level
        
        # Generate synthetic sequence data
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic sequence data."""
        # Generate random input sequences
        self.inputs = torch.randn(self.num_samples, self.seq_length, self.input_dim)
        
        # Create output sequences with temporal dependencies
        self.outputs = torch.zeros(self.num_samples, self.seq_length, self.output_dim)
        
        for i in range(self.num_samples):
            # Create a simple temporal pattern
            for t in range(self.seq_length):
                # Use current input and previous outputs to create temporal dependency
                current_input = self.inputs[i, t]
                
                if t == 0:
                    # First timestep: use only current input
                    self.outputs[i, t] = torch.matmul(current_input, torch.randn(self.input_dim, self.output_dim))
                else:
                    # Subsequent timesteps: use current input and previous output
                    prev_output = self.outputs[i, t-1]
                    self.outputs[i, t] = (
                        torch.matmul(current_input, torch.randn(self.input_dim, self.output_dim)) +
                        torch.matmul(prev_output, torch.randn(self.output_dim, self.output_dim)) * 0.5
                    )
        
        # Add noise
        noise = torch.randn_like(self.outputs) * self.noise_level
        self.outputs += noise
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class DataLoader:
    """Data loader for both transformer and MLP planners."""
    
    def __init__(self, 
                 batch_size=32, 
                 shuffle=True, 
                 num_workers=0,
                 pin_memory=True):
        """
        Initialize data loader.
        
        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def create_mlp_dataloaders(self, 
                              train_size=800, 
                              val_size=100, 
                              test_size=100,
                              input_dim=64, 
                              output_dim=32,
                              task_type='regression'):
        """
        Create data loaders for MLP planner.
        
        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            test_size: Number of test samples
            input_dim: Input dimension
            output_dim: Output dimension
            task_type: Type of task ('regression' or 'classification')
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = SyntheticDataset(
            num_samples=train_size,
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type
        )
        
        val_dataset = SyntheticDataset(
            num_samples=val_size,
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type
        )
        
        test_dataset = SyntheticDataset(
            num_samples=test_size,
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def create_transformer_dataloaders(self, 
                                     train_size=800, 
                                     val_size=100, 
                                     test_size=100,
                                     input_dim=64, 
                                     output_dim=32,
                                     seq_length=20):
        """
        Create data loaders for transformer planner.
        
        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            test_size: Number of test samples
            input_dim: Input dimension per timestep
            output_dim: Output dimension per timestep
            seq_length: Sequence length
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = SequenceDataset(
            num_samples=train_size,
            input_dim=input_dim,
            output_dim=output_dim,
            seq_length=seq_length
        )
        
        val_dataset = SequenceDataset(
            num_samples=val_size,
            input_dim=input_dim,
            output_dim=output_dim,
            seq_length=seq_length
        )
        
        test_dataset = SequenceDataset(
            num_samples=test_size,
            input_dim=input_dim,
            output_dim=output_dim,
            seq_length=seq_length
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, val_loader, test_loader


def collate_fn_pad(batch):
    """
    Custom collate function for variable length sequences.
    
    Args:
        batch: List of (input, output) tuples
    
    Returns:
        Tuple of (padded_inputs, padded_outputs, lengths)
    """
    # Separate inputs and outputs
    inputs, outputs = zip(*batch)
    
    # Get lengths
    lengths = [len(input_seq) for input_seq in inputs]
    max_len = max(lengths)
    
    # Pad sequences
    padded_inputs = torch.zeros(len(batch), max_len, inputs[0].size(-1))
    padded_outputs = torch.zeros(len(batch), max_len, outputs[0].size(-1))
    
    for i, (input_seq, output_seq, length) in enumerate(zip(inputs, outputs, lengths)):
        padded_inputs[i, :length] = input_seq
        padded_outputs[i, :length] = output_seq
    
    return padded_inputs, padded_outputs, lengths 