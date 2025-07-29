import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TransformerPlanner, MLPPlanner
from training import Trainer, DataLoader
from training.data_loader import SyntheticDataset, SequenceDataset


class TestDataLoader:
    """Test cases for DataLoader."""
    
    def test_synthetic_dataset_regression(self):
        """Test synthetic dataset for regression."""
        dataset = SyntheticDataset(
            num_samples=100,
            input_dim=64,
            output_dim=32,
            task_type='regression'
        )
        
        assert len(dataset) == 100
        assert dataset.inputs.shape == (100, 64)
        assert dataset.outputs.shape == (100, 32)
        
        # Test getitem
        x, y = dataset[0]
        assert x.shape == (64,)
        assert y.shape == (32,)
    
    def test_synthetic_dataset_classification(self):
        """Test synthetic dataset for classification."""
        dataset = SyntheticDataset(
            num_samples=100,
            input_dim=64,
            output_dim=2,
            task_type='classification'
        )
        
        assert len(dataset) == 100
        assert dataset.inputs.shape == (100, 64)
        assert dataset.outputs.shape == (100, 2)
        
        # Test that outputs are one-hot encoded
        assert torch.all(torch.sum(dataset.outputs, dim=1) == 1.0)
    
    def test_sequence_dataset(self):
        """Test sequence dataset."""
        dataset = SequenceDataset(
            num_samples=100,
            input_dim=64,
            output_dim=32,
            seq_length=20
        )
        
        assert len(dataset) == 100
        assert dataset.inputs.shape == (100, 20, 64)
        assert dataset.outputs.shape == (100, 20, 32)
        
        # Test getitem
        x, y = dataset[0]
        assert x.shape == (20, 64)
        assert y.shape == (20, 32)
    
    def test_mlp_dataloaders(self):
        """Test MLP data loaders creation."""
        data_loader = DataLoader(batch_size=16)
        train_loader, val_loader, test_loader = data_loader.create_mlp_dataloaders(
            train_size=100,
            val_size=20,
            test_size=20,
            input_dim=64,
            output_dim=32
        )
        
        # Test that loaders have correct number of batches
        assert len(train_loader) == 7  # 100 samples / 16 batch size = 7 batches
        assert len(val_loader) == 2   # 20 samples / 16 batch size = 2 batches
        assert len(test_loader) == 2  # 20 samples / 16 batch size = 2 batches
        
        # Test batch shapes
        for batch in train_loader:
            inputs, targets = batch
            assert inputs.shape[0] <= 16  # Batch size
            assert inputs.shape[1] == 64  # Input dimension
            assert targets.shape[0] == inputs.shape[0]  # Same batch size
            assert targets.shape[1] == 32  # Output dimension
            break
    
    def test_transformer_dataloaders(self):
        """Test transformer data loaders creation."""
        data_loader = DataLoader(batch_size=16)
        train_loader, val_loader, test_loader = data_loader.create_transformer_dataloaders(
            train_size=100,
            val_size=20,
            test_size=20,
            input_dim=64,
            output_dim=32,
            seq_length=20
        )
        
        # Test that loaders have correct number of batches
        assert len(train_loader) == 7  # 100 samples / 16 batch size = 7 batches
        assert len(val_loader) == 2   # 20 samples / 16 batch size = 2 batches
        assert len(test_loader) == 2  # 20 samples / 16 batch size = 2 batches
        
        # Test batch shapes
        for batch in train_loader:
            inputs, targets = batch
            assert inputs.shape[0] <= 16  # Batch size
            assert inputs.shape[1] == 20  # Sequence length
            assert inputs.shape[2] == 64  # Input dimension
            assert targets.shape[0] == inputs.shape[0]  # Same batch size
            assert targets.shape[1] == 20  # Sequence length
            assert targets.shape[2] == 32  # Output dimension
            break


class TestTrainer:
    """Test cases for Trainer."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        log_dir = os.path.join(temp_dir, 'logs')
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        yield checkpoint_dir, log_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data_loaders(self):
        """Create sample data loaders for testing."""
        data_loader = DataLoader(batch_size=8)
        train_loader, val_loader, test_loader = data_loader.create_mlp_dataloaders(
            train_size=32,
            val_size=16,
            test_size=16,
            input_dim=32,
            output_dim=16
        )
        return train_loader, val_loader, test_loader
    
    def test_trainer_initialization(self, temp_dirs, sample_data_loaders):
        """Test trainer initialization."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=5,
            verbose=False
        )
        
        assert trainer.model == model
        assert trainer.device == 'cpu'
        assert trainer.num_epochs == 5
        assert trainer.checkpoint_dir == checkpoint_dir
        assert trainer.log_dir == log_dir
    
    def test_trainer_optimizer_setup(self, temp_dirs, sample_data_loaders):
        """Test optimizer setup."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        # Test different optimizers
        optimizers = ['adam', 'sgd', 'adamw']
        for optimizer_type in optimizers:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device='cpu',
                optimizer=optimizer_type,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                num_epochs=1,
                verbose=False
            )
            
            assert trainer.optimizer_type == optimizer_type
            assert trainer.optimizer is not None
    
    def test_trainer_scheduler_setup(self, temp_dirs, sample_data_loaders):
        """Test scheduler setup."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        # Test different schedulers
        schedulers = ['step', 'cosine', 'plateau']
        for scheduler_type in schedulers:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device='cpu',
                scheduler=scheduler_type,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                num_epochs=1,
                verbose=False
            )
            
            assert trainer.scheduler_type == scheduler_type
    
    def test_trainer_loss_function_setup(self, temp_dirs, sample_data_loaders):
        """Test loss function setup."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        # Test different loss functions
        loss_functions = ['mse', 'mae', 'cross_entropy']
        for loss_fn in loss_functions:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device='cpu',
                loss_fn=loss_fn,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                num_epochs=1,
                verbose=False
            )
            
            assert trainer.loss_fn_type == loss_fn
            assert trainer.loss_fn is not None
    
    def test_train_epoch(self, temp_dirs, sample_data_loaders):
        """Test training for one epoch."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=1,
            verbose=False
        )
        
        # Train for one epoch
        loss, metrics = trainer.train_epoch()
        
        assert isinstance(loss, float)
        assert loss > 0
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
    
    def test_validate_epoch(self, temp_dirs, sample_data_loaders):
        """Test validation for one epoch."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=1,
            verbose=False
        )
        
        # Validate for one epoch
        loss, metrics = trainer.validate_epoch()
        
        assert isinstance(loss, float)
        assert loss > 0
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
    
    def test_test_model(self, temp_dirs, sample_data_loaders):
        """Test model testing."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=1,
            verbose=False
        )
        
        # Test model
        loss, metrics = trainer.test_model()
        
        assert isinstance(loss, float)
        assert loss > 0
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
    
    def test_save_load_checkpoint(self, temp_dirs, sample_data_loaders):
        """Test checkpoint saving and loading."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=1,
            verbose=False
        )
        
        # Save checkpoint
        trainer.save_checkpoint(epoch=5, is_best=True)
        
        # Check that checkpoint files exist
        latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        best_checkpoint = os.path.join(checkpoint_dir, 'best_model.pth')
        
        assert os.path.exists(latest_checkpoint)
        assert os.path.exists(best_checkpoint)
        
        # Load checkpoint
        loaded_epoch = trainer.load_checkpoint(latest_checkpoint)
        assert loaded_epoch == 5
    
    def test_short_training_run(self, temp_dirs, sample_data_loaders):
        """Test a short training run."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=3,
            early_stopping_patience=10,
            verbose=False
        )
        
        # Train model
        history = trainer.train()
        
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'train_metrics' in history
        assert 'val_metrics' in history
        
        # Check that losses are recorded
        assert len(history['train_loss']) > 0
        assert len(history['val_loss']) > 0
        
        # Check that losses are decreasing (or at least not increasing dramatically)
        assert history['train_loss'][-1] < history['train_loss'][0] * 2
    
    def test_early_stopping(self, temp_dirs, sample_data_loaders):
        """Test early stopping functionality."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=10,
            early_stopping_patience=2,
            verbose=False
        )
        
        # Train model (should stop early)
        history = trainer.train()
        
        # Should have trained for at least 2 epochs but less than 10
        assert len(history['train_loss']) >= 2
        assert len(history['train_loss']) <= 10
    
    def test_transformer_training(self, temp_dirs):
        """Test training with transformer model."""
        checkpoint_dir, log_dir = temp_dirs
        
        # Create transformer data loaders
        data_loader = DataLoader(batch_size=8)
        train_loader, val_loader, test_loader = data_loader.create_transformer_dataloaders(
            train_size=32,
            val_size=16,
            test_size=16,
            input_dim=32,
            output_dim=16,
            seq_length=10
        )
        
        model = TransformerPlanner(
            input_dim=32,
            output_dim=16,
            d_model=64,
            num_heads=4,
            num_layers=2,
            max_seq_length=10
        )
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=2,
            verbose=False
        )
        
        # Train model
        history = trainer.train()
        
        assert isinstance(history, dict)
        assert len(history['train_loss']) > 0
        assert len(history['val_loss']) > 0
    
    def test_trainer_with_different_devices(self, temp_dirs, sample_data_loaders):
        """Test trainer with different devices."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        # Test CPU
        trainer_cpu = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=1,
            verbose=False
        )
        
        loss_cpu, _ = trainer_cpu.train_epoch()
        assert isinstance(loss_cpu, float)
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = MLPPlanner(input_dim=32, output_dim=16)
            trainer_cuda = Trainer(
                model=model_cuda,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device='cuda',
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
                num_epochs=1,
                verbose=False
            )
            
            loss_cuda, _ = trainer_cuda.train_epoch()
            assert isinstance(loss_cuda, float)
    
    def test_trainer_plotting_functions(self, temp_dirs, sample_data_loaders):
        """Test trainer plotting functions."""
        checkpoint_dir, log_dir = temp_dirs
        train_loader, val_loader, test_loader = sample_data_loaders
        
        model = MLPPlanner(input_dim=32, output_dim=16)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device='cpu',
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            num_epochs=2,
            verbose=False
        )
        
        # Train for a few epochs
        history = trainer.train()
        
        # Test plotting (should not raise errors)
        try:
            trainer.plot_training_history()
        except Exception as e:
            # If matplotlib is not available, this is expected
            assert "matplotlib" in str(e).lower() or "not available" in str(e).lower()
        
        # Test saving training history
        history_file = os.path.join(temp_dirs[0], 'history.json')
        trainer.save_training_history(history_file)
        assert os.path.exists(history_file)


if __name__ == '__main__':
    pytest.main([__file__]) 