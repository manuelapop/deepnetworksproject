#!/usr/bin/env python3
"""
Main training script for PyTorch Transformer and MLP Planners.

Usage:
    python main.py --model transformer --epochs 100 --batch_size 32
    python main.py --model mlp --epochs 100 --batch_size 32
"""

import argparse
import torch
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models import TransformerPlanner, MLPPlanner
from training import Trainer, DataLoader
from utils.metrics import calculate_metrics, plot_training_curves, plot_model_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Transformer or MLP Planner')
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['transformer', 'mlp'], 
                       default='transformer', help='Model type to train')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd', 'adamw'], 
                       default='adam', help='Optimizer type')
    parser.add_argument('--scheduler', type=str, choices=['step', 'cosine', 'plateau'], 
                       default='step', help='Learning rate scheduler')
    parser.add_argument('--loss_fn', type=str, choices=['mse', 'mae', 'cross_entropy'], 
                       default='mse', help='Loss function')
    
    # Model architecture parameters
    parser.add_argument('--input_dim', type=int, default=64, help='Input dimension')
    parser.add_argument('--output_dim', type=int, default=32, help='Output dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, 
                       help='Hidden dimension for MLP')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension for transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_length', type=int, default=20, help='Sequence length for transformer')
    
    # Data parameters
    parser.add_argument('--train_size', type=int, default=800, help='Number of training samples')
    parser.add_argument('--val_size', type=int, default=100, help='Number of validation samples')
    parser.add_argument('--test_size', type=int, default=100, help='Number of test samples')
    parser.add_argument('--task_type', type=str, choices=['regression', 'classification'], 
                       default='regression', help='Task type')
    
    # Training configuration
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--early_stopping_patience', type=int, default=10, 
                       help='Early stopping patience')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logs')
    parser.add_argument('--save_best', action='store_true', default=True, 
                       help='Save best model')
    parser.add_argument('--verbose', action='store_true', default=True, 
                       help='Print training progress')
    parser.add_argument('--step_training', action='store_true', 
                       help='Train in steps instead of all epochs at once')
    parser.add_argument('--step_size', type=int, default=10, 
                       help='Number of epochs per step when using step training')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model after training')
    parser.add_argument('--plot_results', action='store_true', default=True, 
                       help='Plot training results')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup device for training."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            print("CUDA not available, using CPU")
    else:
        device = device_arg
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    
    return device


def create_model(args):
    """Create model based on arguments."""
    if args.model == 'transformer':
        model = TransformerPlanner(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            max_seq_length=args.seq_length
        )
        print(f"Created Transformer Planner with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    elif args.model == 'mlp':
        model = MLPPlanner(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout,
            use_batch_norm=True,
            activation='relu'
        )
        print(f"Created MLP Planner with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def create_data_loaders(args):
    """Create data loaders based on model type."""
    data_loader = DataLoader(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    if args.model == 'transformer':
        train_loader, val_loader, test_loader = data_loader.create_transformer_dataloaders(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            seq_length=args.seq_length
        )
    else:  # MLP
        train_loader, val_loader, test_loader = data_loader.create_mlp_dataloaders(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            task_type=args.task_type
        )
    
    return train_loader, val_loader, test_loader


def evaluate_model(model, test_loader, device, task_type):
    """Evaluate model on test set."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            all_predictions.append(outputs)
            all_targets.append(targets)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_predictions, task_type)
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.4f}")
    print("="*50)
    
    return all_predictions, all_targets, metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    
    # Create model
    model = create_model(args)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(args)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        loss_fn=args.loss_fn,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_best=args.save_best,
        verbose=args.verbose
    )
    
    # Train model
    print(f"\nStarting training for {args.model.upper()} planner...")
    
    if args.step_training:
        print(f"Training in steps of {args.step_size} epochs...")
        step_size = args.step_size
        total_steps = args.epochs // step_size
        
        for step in range(total_steps):
            print(f"\n{'='*40}")
            print(f"STEP {step + 1}/{total_steps}")
            print(f"{'='*40}")
            
            result = trainer.train_step(step_size)
            
            print(f"\nStep {step + 1} completed:")
            print(f"  Completed epochs: {result['completed_epochs']}/{result['total_epochs']}")
            print(f"  Best validation loss so far: {result['best_val_loss']:.4f}")
            
            if result['early_stopped']:
                print("  Training stopped early due to no improvement")
                break
            
            if result['training_complete']:
                print("  Training completed!")
                break
        
        training_history = trainer.training_history
    else:
        training_history = trainer.train()
    
    # Evaluate model
    if args.evaluate:
        print("\nEvaluating model...")
        predictions, targets, metrics = evaluate_model(
            model, test_loader, device, args.task_type
        )
        
        # Plot results
        if args.plot_results:
            print("\nGenerating plots...")
            
            # Plot training curves
            plot_training_curves(
                training_history['train_loss'],
                training_history['val_loss'],
                save_path=f'training_curves_{args.model}.png'
            )
            
            # Plot model predictions
            plot_model_predictions(
                targets, predictions, args.task_type,
                save_path=f'predictions_{args.model}.png'
            )
            
            # Save training history
            trainer.save_training_history(f'training_history_{args.model}.json')
            
            print(f"Plots saved as training_curves_{args.model}.png and predictions_{args.model}.png")
    
    print(f"\nTraining completed! Checkpoints saved in {args.checkpoint_dir}/")
    print(f"TensorBoard logs saved in {args.log_dir}/")
    
    # Print model summary
    if hasattr(model, 'get_model_summary'):
        summary = model.get_model_summary()
        print("\nModel Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main() 