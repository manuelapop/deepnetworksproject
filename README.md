# PyTorch Transformer and MLP Planner

This project implements transformer and MLP-based planners using PyTorch, with comprehensive training loops and tests.

## Project Structure

```
deepnetworksproject/
├── models/
│   ├── __init__.py
│   ├── transformer_planner.py
│   └── mlp_planner.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── data_loader.py
├── tests/
│   ├── __init__.py
│   ├── test_transformer_planner.py
│   ├── test_mlp_planner.py
│   └── test_trainer.py
├── utils/
│   ├── __init__.py
│   └── metrics.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the transformer planner:
```bash
python main.py --model transformer --epochs 100 --batch_size 32
```

Train the MLP planner:
```bash
python main.py --model mlp --epochs 100 --batch_size 32
```

### Testing

Run all tests:
```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_transformer_planner.py
pytest tests/test_mlp_planner.py
pytest tests/test_trainer.py
```

## Models

### Transformer Planner
- Multi-head self-attention mechanism
- Positional encoding
- Configurable number of layers and heads
- Suitable for sequence-based planning tasks

### MLP Planner
- Multi-layer perceptron with configurable architecture
- ReLU activation functions
- Dropout for regularization
- Suitable for feature-based planning tasks

## Features

- Comprehensive training loops with validation
- TensorBoard logging
- Model checkpointing
- Early stopping
- Learning rate scheduling
- Comprehensive test suite
- Configurable hyperparameters 