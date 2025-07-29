import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, task_type: str = 'regression') -> Dict[str, float]:
    """
    Calculate various metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_type: Type of task ('regression' or 'classification')
    
    Returns:
        Dictionary containing calculated metrics
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    metrics = {}
    
    if task_type == 'regression':
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional regression metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        metrics['smape'] = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
    elif task_type == 'classification':
        # For classification, convert probabilities to predictions
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Multi-class classification
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        else:
            # Binary classification
            y_pred_classes = (y_pred > 0.5).astype(int)
            y_true_classes = y_true
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true_classes, y_pred_classes)
        
        # For multi-class, calculate macro averages
        if len(np.unique(y_true_classes)) > 2:
            metrics['precision'] = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
            metrics['recall'] = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
            metrics['f1'] = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
        else:
            metrics['precision'] = precision_score(y_true_classes, y_pred_classes, zero_division=0)
            metrics['recall'] = recall_score(y_true_classes, y_pred_classes, zero_division=0)
            metrics['f1'] = f1_score(y_true_classes, y_pred_classes, zero_division=0)
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return metrics


def plot_attention_weights(attention_weights: List[torch.Tensor], 
                          layer_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot attention weights from transformer layers.
    
    Args:
        attention_weights: List of attention weight tensors from different layers
        layer_names: Names for each layer (optional)
        save_path: Path to save the plot (optional)
        figsize: Figure size for the plot
    """
    if not attention_weights:
        print("No attention weights provided")
        return
    
    # Convert to numpy arrays
    attention_weights_np = []
    for weights in attention_weights:
        if isinstance(weights, torch.Tensor):
            # Average over heads and batch
            weights_np = weights.detach().cpu().numpy()
            if weights_np.ndim == 4:  # (batch, heads, seq_len, seq_len)
                weights_np = np.mean(weights_np, axis=(0, 1))  # Average over batch and heads
            attention_weights_np.append(weights_np)
        else:
            attention_weights_np.append(weights)
    
    # Create subplots
    num_layers = len(attention_weights_np)
    num_cols = min(3, num_layers)
    num_rows = (num_layers + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_layers == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each layer's attention weights
    for i, (weights, ax) in enumerate(zip(attention_weights_np, axes)):
        # Create heatmap
        im = ax.imshow(weights, cmap='viridis', aspect='auto')
        
        # Set title
        if layer_names and i < len(layer_names):
            ax.set_title(f'Layer {i+1}: {layer_names[i]}')
        else:
            ax.set_title(f'Layer {i+1}')
        
        # Set labels
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    # Hide empty subplots
    for i in range(num_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        train_metrics: Optional[List[Dict[str, float]]] = None,
                        val_metrics: Optional[List[Dict[str, float]]] = None,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot training curves including loss and metrics.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: List of training metrics dictionaries
        val_metrics: List of validation metrics dictionaries
        save_path: Path to save the plot (optional)
        figsize: Figure size for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics if provided
    if train_metrics and val_metrics and len(train_metrics) > 0:
        # Get the first metric key (assuming all dictionaries have the same keys)
        metric_key = list(train_metrics[0].keys())[0]
        
        train_metric_values = [metrics[metric_key] for metrics in train_metrics]
        val_metric_values = [metrics[metric_key] for metrics in val_metrics]
        
        axes[1].plot(epochs, train_metric_values, 'b-', label=f'Training {metric_key.title()}')
        axes[1].plot(epochs, val_metric_values, 'r-', label=f'Validation {metric_key.title()}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_key.title())
        axes[1].set_title(f'Training and Validation {metric_key.title()}')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_model_predictions(y_true: torch.Tensor, 
                          y_pred: torch.Tensor,
                          task_type: str = 'regression',
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot model predictions vs true values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: Type of task ('regression' or 'classification')
        save_path: Path to save the plot (optional)
        figsize: Figure size for the plot
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    if task_type == 'regression':
        # Flatten arrays for plotting
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Scatter plot
        axes[0, 0].scatter(y_true_flat, y_pred_flat, alpha=0.6)
        axes[0, 0].plot([y_true_flat.min(), y_true_flat.max()], 
                       [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predicted vs True Values')
        axes[0, 0].grid(True)
        
        # Residual plot
        residuals = y_pred_flat - y_true_flat
        axes[0, 1].scatter(y_pred_flat, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True)
        
        # Time series plot (if applicable)
        if y_true.ndim == 1:
            axes[1, 1].plot(y_true_flat[:100], label='True', alpha=0.7)
            axes[1, 1].plot(y_pred_flat[:100], label='Predicted', alpha=0.7)
            axes[1, 1].set_xlabel('Sample Index')
            axes[1, 1].set_ylabel('Values')
            axes[1, 1].set_title('Time Series Comparison (First 100 samples)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Time series plot not available\nfor multi-dimensional data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Time Series Comparison')
    
    elif task_type == 'classification':
        # For classification, show confusion matrix and ROC curve
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Multi-class classification
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        else:
            # Binary classification
            y_pred_classes = (y_pred > 0.5).astype(int)
            y_true_classes = y_true
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # Classification report
        from sklearn.metrics import classification_report
        report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
        axes[0, 1].text(0.1, 0.9, str(classification_report(y_true_classes, y_pred_classes)), 
                       transform=axes[0, 1].transAxes, fontsize=10, verticalalignment='top')
        axes[0, 1].set_title('Classification Report')
        axes[0, 1].axis('off')
        
        # Hide unused subplots
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def calculate_gradient_norm(model: torch.nn.Module) -> float:
    """
    Calculate the L2 norm of gradients for all parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        L2 norm of gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    } 