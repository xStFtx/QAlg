# Scalar Field Evolution Predictor with Neural Networks

This project uses convolutional neural networks to predict the evolution of a scalar field. The model can be further trained on-the-fly, leveraging discrepancies between predictions and actual evolutions to continually enhance its accuracy.

## Features

- Simulates the dynamics of a scalar field on a lattice.
- Evolves the field using a discrete Laplacian and a potential term.
- Generates training datasets by evolving random initial configurations.
- Constructs a convolutional neural network model for predicting the field evolution.
- Continuously retrains the model on new data, enabling it to refine its predictions over time.

## Quick Start

1. Ensure you have the required libraries installed:
    - numpy
    - matplotlib
    - tensorflow

2. Clone the repository:
```
git clone https://github.com/xStFtx/QAlg.git
```

3. Change dir:
```
cd QAlg
```
4. Run the main script:
```
python main.py
```