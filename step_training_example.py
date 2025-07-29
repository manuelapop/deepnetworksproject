#!/usr/bin/env python3
"""
Example script demonstrating step-by-step training with 10-epoch steps.

This script shows how to:
1. Train a model in steps of 10 epochs
2. Resume training from checkpoints
3. Monitor progress between steps
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from models import TransformerPlanner, MLPPlanner
from training import Trainer, DataLoader
from utils.metrics import calculate_metrics


def step_training_example():
    """Example of step-by-step training."""
    print("="*60)
    print("STEP-BY-STEP TRAINING EXAMPLE")
    print("="*60)
    
    # Create data loaders
    data_loader = DataLoader(batch_size=16)
    train_loader, val_loader, test_loader = data_loader.create_transformer_dataloaders(
        train_size=200,
        val_size=50,
        test_size=50,
        input_dim=32,
        output_dim=16,
        seq_length=15
    )
    
    # Create transformer model
    model = TransformerPlanner(
        input_dim=32,
        output_dim=16,
        d_model=128,
        num_heads=8,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        max_seq_length=15
    )
    
    print(f"Transformer model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer with 50 total epochs
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cpu',
        learning_rate=1e-3,
        num_epochs=50,  # Total epochs to train
        early_stopping_patience=15,  # Increased patience for step training
        checkpoint_dir='checkpoints_step',
        log_dir='logs_step',
        verbose=True
    )
    
    # Train in steps of 10 epochs
    step_size = 10
    total_steps = trainer.num_epochs // step_size
    
    print(f"\nTraining for {trainer.num_epochs} epochs in {total_steps} steps of {step_size} epochs each")
    
    for step in range(total_steps):
        print(f"\n{'='*40}")
        print(f"STEP {step + 1}/{total_steps}")
        print(f"{'='*40}")
        
        # Train for this step
        result = trainer.train_step(step_size)
        
        # Print step results
        print(f"\nStep {step + 1} completed:")
        print(f"  Completed epochs: {result['completed_epochs']}/{result['total_epochs']}")
        print(f"  Best validation loss so far: {result['best_val_loss']:.4f}")
        print(f"  Best epoch so far: {result['best_epoch'] + 1}")
        
        if result['early_stopped']:
            print("  Training stopped early due to no improvement")
            break
        
        if result['training_complete']:
            print("  Training completed!")
            break
        
        # Optional: Add a pause between steps
        if step < total_steps - 1:  # Don't pause after the last step
            print(f"\nPausing for 2 seconds before next step...")
            time.sleep(2)
    
    # Final evaluation
    print(f"\n{'='*40}")
    print("FINAL EVALUATION")
    print(f"{'='*40}")
    
    test_loss, test_metrics = trainer.test_model()
    print(f"Final test loss: {test_loss:.4f}")
    
    # Plot training history
    trainer.plot_training_history(save_path='step_training_curves.png')
    
    # Save training history
    trainer.save_training_history('step_training_history.json')
    
    print(f"\nTraining completed!")
    print(f"Checkpoints saved in: checkpoints_step/")
    print(f"Logs saved in: logs_step/")
    print(f"Training curves saved as: step_training_curves.png")
    
    return trainer, result


def resume_training_example():
    """Example of resuming training from a checkpoint."""
    print("\n" + "="*60)
    print("RESUME TRAINING EXAMPLE")
    print("="*60)
    
    # Create data loaders
    data_loader = DataLoader(batch_size=16)
    train_loader, val_loader, test_loader = data_loader.create_mlp_dataloaders(
        train_size=200,
        val_size=50,
        test_size=50,
        input_dim=64,
        output_dim=32,
        task_type='regression'
    )
    
    # Create MLP model
    model = MLPPlanner(
        input_dim=64,
        output_dim=32,
        hidden_dim=256,
        dropout_rate=0.2,
        use_batch_norm=True,
        activation='relu'
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cpu',
        learning_rate=1e-3,
        num_epochs=30,
        early_stopping_patience=10,
        checkpoint_dir='checkpoints_resume',
        log_dir='logs_resume',
        verbose=True
    )
    
    # Train for first 10 epochs
    print("Training for first 10 epochs...")
    result1 = trainer.train_step(10)
    print(f"First step completed: {result1['completed_epochs']} epochs")
    
    # Simulate stopping and resuming
    print("\nSimulating training interruption...")
    print("(In real scenario, you would stop the script here)")
    
    # Create a new trainer instance (simulating restart)
    trainer2 = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cpu',
        learning_rate=1e-3,
        num_epochs=30,
        early_stopping_patience=10,
        checkpoint_dir='checkpoints_resume',
        log_dir='logs_resume',
        verbose=True
    )
    
    # Load checkpoint to resume training
    print("\nLoading checkpoint to resume training...")
    loaded_epoch = trainer2.load_checkpoint('checkpoints_resume/latest_checkpoint.pth')
    print(f"Resumed from epoch {loaded_epoch + 1}")
    
    # Continue training for remaining epochs
    remaining_epochs = trainer2.num_epochs - trainer2.current_epoch
    print(f"Continuing training for {remaining_epochs} more epochs...")
    result2 = trainer2.train_step(remaining_epochs)
    
    print(f"Resume training completed: {result2['completed_epochs']} total epochs")
    
    return trainer2, result2


def main():
    """Main function to run step training examples."""
    print("Step-by-Step Training Examples")
    print("This script demonstrates training in 10-epoch steps.")
    
    try:
        # Run step training example
        trainer1, result1 = step_training_example()
        
        # Run resume training example
        trainer2, result2 = resume_training_example()
        
        print("\n" + "="*60)
        print("EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("  - step_training_curves.png")
        print("  - step_training_history.json")
        print("  - checkpoints_step/ (model checkpoints)")
        print("  - checkpoints_resume/ (model checkpoints)")
        print("  - logs_step/ (TensorBoard logs)")
        print("  - logs_resume/ (TensorBoard logs)")
        
        print("\nTo view TensorBoard logs:")
        print("  tensorboard --logdir logs_step")
        print("  tensorboard --logdir logs_resume")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 