import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import json


class Trainer:
    """Comprehensive trainer for both transformer and MLP planners."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 test_loader,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 optimizer: str = 'adam',
                 scheduler: str = 'step',
                 loss_fn: str = 'mse',
                 num_epochs: int = 100,
                 early_stopping_patience: int = 10,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs',
                 save_best: bool = True,
                 verbose: bool = True):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
            scheduler: Learning rate scheduler ('step', 'cosine', 'plateau')
            loss_fn: Loss function ('mse', 'mae', 'cross_entropy')
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            save_best: Whether to save best model
            verbose: Whether to print training progress
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.scheduler_type = scheduler
        self.loss_fn_type = loss_fn
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.save_best = save_best
        self.verbose = verbose
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_tensorboard()
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        elif self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs
            )
        elif self.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, factor=0.5
            )
        else:
            self.scheduler = None
    
    def _setup_loss_function(self):
        """Setup loss function."""
        if self.loss_fn_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss_fn_type == 'mae':
            self.loss_fn = nn.L1Loss()
        elif self.loss_fn_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_type}")
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer."""
        self.writer = SummaryWriter(self.log_dir)
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc='Training', disable=not self.verbose)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        metrics = {'loss': avg_loss}
        
        return avg_loss, metrics
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.loss_fn(outputs, targets)
                
                # Update statistics
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        metrics = {'loss': avg_loss}
        
        return avg_loss, metrics
    
    def test_model(self) -> Tuple[float, Dict[str, float]]:
        """Test the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.loss_fn(outputs, targets)
                
                # Update statistics
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        metrics = {'loss': avg_loss}
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'model_config': self._get_model_config()
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best and self.save_best:
            best_checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        return checkpoint['epoch']
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for saving."""
        if hasattr(self.model, 'get_model_summary'):
            return self.model.get_model_summary()
        else:
            return {
                'model_type': type(self.model).__name__,
                'num_parameters': sum(p.numel() for p in self.model.parameters())
            }
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        if self.verbose:
            print(f"Starting training for {self.num_epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_metrics = self.train_epoch()
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch()
            
            # Update learning rate scheduler
            if self.scheduler:
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Print progress
            if self.verbose:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Final test
        test_loss, test_metrics = self.test_model()
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\nTraining completed in {total_time:.2f}s")
            print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
            print(f"Test loss: {test_loss:.4f}")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.training_history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot losses
            ax1.plot(self.training_history['train_loss'], label='Train Loss')
            ax1.plot(self.training_history['val_loss'], label='Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot learning rate
            if self.scheduler:
                lr_history = [self.optimizer.param_groups[0]['lr']] * len(self.training_history['train_loss'])
                ax2.plot(lr_history, label='Learning Rate')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Learning Rate Schedule')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def save_training_history(self, file_path: str):
        """Save training history to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                history[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
            else:
                history[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(history, f, indent=2) 