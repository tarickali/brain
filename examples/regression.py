"""
title : regression.py
create : @tarickali 23/12/18
update : @tarickali 23/12/18
"""

import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from brain.core import Node
from brain.modules import Linear
from brain.models import Sequential
from brain.losses import MeanSquaredError
from brain.optimizers import Adam

__all__ = ["regression"]


def regression():
    # Set random seed
    np.random.seed(42)

    # Get data
    X, y = make_regression(256, 100, random_state=42)
    y = y.reshape(-1, 1)

    # Track data as Node objects
    X = Node(X)
    y = Node(y)

    # Create model
    model = Sequential(
        [
            Linear(100, 64, "relu"),
            Linear(64, 32, "relu"),
            Linear(32, 1),
        ]
    )

    # Setup loss and optimizer
    loss = MeanSquaredError()
    optim = Adam(model.parameters, learning_rate=0.01)

    # Training loop
    history = []
    for e in range(100):
        o = model(X)
        l = loss(y, o)
        history.append(l.data.array[0])

        model.zero_grad()
        l.backward()
        optim.update()

        print(f"epoch {e} -- loss {l.data.array}")

    # Get predictions of final model
    pred = model(X)

    # Setup subplots
    fig, axes = plt.subplots(nrows=2)

    # Plot loss curve
    axes[0].plot(history)
    axes[0].set(title="Loss Curve", xlabel="Epochs", ylabel="Loss")

    # Plot true and predicted values to view overlap
    axes[1].scatter(
        range(y.shape[0]), y.data.array[:, 0], marker="o", color="r", label="True"
    )
    axes[1].scatter(
        range(y.shape[0]), pred.data.array[:, 0], marker="x", color="b", label="Pred"
    )
    plt.legend()
    plt.show()
