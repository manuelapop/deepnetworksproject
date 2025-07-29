#!/usr/bin/env python3
"""
Example script demonstrating the usage of Transformer and MLP Planners.

This script shows how to:
1. Create and train a Transformer Planner
2. Create and train an MLP Planner
3. Evaluate and visualize results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from models import TransformerPlanner, MLPPlanner
from training import Trainer, DataLoader
from utils.metrics import calculate_metrics, plot_attention_weights, plot_model_predictions


def example_transformer_planner():
    """Example of using the Transformer Planner."""
    print("="*60)
    print("TRANSFORMER PLANNER EXAMPLE")
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
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cpu',
        learning_rate=1e-3,
        num_epochs=10,
        early_stopping_patience=5,
        checkpoint_dir='checkpoints_transformer',
        log_dir='logs_transformer',
        verbose=True
    )
    
    # Train the model
    print("\nTraining Transformer Planner...")
    history = trainer.train()
    
    # Evaluate the model
    print("\nEvaluating Transformer Planner...")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_predictions.append(outputs)
            all_targets.append(targets)
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(targets, predictions, 'regression')
    print("\nTransformer Planner Results:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Visualize attention weights
    print("\nGenerating attention weight visualization...")
    sample_input = next(iter(test_loader))[0][:1]  # Take first sample
    _, attention_weights = model.get_attention_weights(sample_input)
    
    plot_attention_weights(
        attention_weights,
        save_path='transformer_attention_weights.png'
    )
    
    return model, history, metrics


def example_mlp_planner():
    """Example of using the MLP Planner."""
    print("\n" + "="*60)
    print("MLP PLANNER EXAMPLE")
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
    
    print(f"MLP model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Print model summary
    summary = model.get_model_summary()
    print("\nMLP Model Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device='cpu',
        learning_rate=1e-3,
        num_epochs=10,
        early_stopping_patience=5,
        checkpoint_dir='checkpoints_mlp',
        log_dir='logs_mlp',
        verbose=True
    )
    
    # Train the model
    print("\nTraining MLP Planner...")
    history = trainer.train()
    
    # Evaluate the model
    print("\nEvaluating MLP Planner...")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            all_predictions.append(outputs)
            all_targets.append(targets)
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(targets, predictions, 'regression')
    print("\nMLP Planner Results:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Visualize predictions
    print("\nGenerating prediction visualization...")
    plot_model_predictions(
        targets, predictions, 'regression',
        save_path='mlp_predictions.png'
    )
    
    # Demonstrate feature extraction
    print("\nDemonstrating feature extraction...")
    sample_input = torch.randn(1, 64)
    features = model.get_features(sample_input)
    
    print("Feature shapes:")
    for key, feature in features.items():
        print(f"  {key}: {feature.shape}")
    
    return model, history, metrics


def compare_models(transformer_results, mlp_results):
    """Compare the performance of both models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    transformer_metrics, mlp_metrics = transformer_results[2], mlp_results[2]
    
    print("\nPerformance Comparison:")
    print(f"{'Metric':<15} {'Transformer':<15} {'MLP':<15} {'Winner':<10}")
    print("-" * 60)
    
    for metric in ['mse', 'mae', 'r2']:
        if metric in transformer_metrics and metric in mlp_metrics:
            t_val = transformer_metrics[metric]
            m_val = mlp_metrics[metric]
            
            if metric in ['mse', 'mae']:  # Lower is better
                winner = 'Transformer' if t_val < m_val else 'MLP'
            else:  # Higher is better (R²)
                winner = 'Transformer' if t_val > m_val else 'MLP'
            
            print(f"{metric.upper():<15} {t_val:<15.4f} {m_val:<15.4f} {winner:<10}")
    
    # Compare training time and model size
    transformer_params = sum(p.numel() for p in transformer_results[0].parameters())
    mlp_params = sum(p.numel() for p in mlp_results[0].parameters())
    
    print(f"\nModel Complexity:")
    print(f"Transformer parameters: {transformer_params:,}")
    print(f"MLP parameters: {mlp_params:,}")
    print(f"Parameter ratio (Transformer/MLP): {transformer_params/mlp_params:.2f}")


def main():
    """Main function to run both examples."""
    print("PyTorch Transformer and MLP Planner Examples")
    print("This script demonstrates both models with synthetic data.")
    
    try:
        # Run transformer example
        transformer_results = example_transformer_planner()
        
        # Run MLP example
        mlp_results = example_mlp_planner()
        
        # Compare results
        compare_models(transformer_results, mlp_results)
        
        print("\n" + "="*60)
        print("EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("  - transformer_attention_weights.png")
        print("  - mlp_predictions.png")
        print("  - checkpoints_transformer/ (model checkpoints)")
        print("  - checkpoints_mlp/ (model checkpoints)")
        print("  - logs_transformer/ (TensorBoard logs)")
        print("  - logs_mlp/ (TensorBoard logs)")
        
        print("\nTo view TensorBoard logs:")
        print("  tensorboard --logdir logs_transformer")
        print("  tensorboard --logdir logs_mlp")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 