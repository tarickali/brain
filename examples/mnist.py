"""
title : mnist.py
create : @tarickali 23/12/19
update : @tarickali 23/12/19
"""

from keras.datasets import mnist

import numpy as np

from brain.core import Node, Model
import brain.functional as F
from brain.modules import Linear, Conv2d, Flatten
from brain.models import Sequential
from brain.losses import CategoricalCrossentropy
from brain.optimizers import Adam
from brain.data_utils import one_hot, get_batches

__all__ = ["mnist_driver"]


def preprocess_data(x, y):
    x = x.reshape(-1, 1, 28, 28)
    x = x / 255.0
    y = one_hot(y, 10)
    return x, y


def train(model: Model, X, y, epochs=5, batch_size=32, learning_rate=0.001):
    loss_fn = CategoricalCrossentropy(logits=True)
    optimizer = Adam(model.parameters, learning_rate=learning_rate)

    n = X.shape[0]
    m = batch_size
    k = 10
    history = []
    for e in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for Xb, yb in get_batches(X, y, m):
            Xb = Node(Xb)
            yb = Node(yb)
            # print("yb", yb.shape)
            pred = model(Xb)
            loss = loss_fn(yb, pred)

            model.zero_grad()
            loss.backward()
            optimizer.update()

            probs = F.softmax(pred).data.array
            ob = one_hot(np.argmax(probs, axis=1))
            acc = np.sum(yb.data.array == ob) / k

            epoch_loss += loss.data.array * m
            epoch_acc += acc * m
        history.append({"epoch": e, "loss": epoch_loss / n, "acc": epoch_acc / n})
        print(history[-1])


def mnist_driver():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    ALPHA = 0.001
    EPOCHS = 5

    model = Sequential(
        [
            Conv2d(1, 3, 5, activation="sigmoid"),
            Flatten(),
            Linear(100, activation="sigmoid"),
            Linear(10),
        ]
    )

    train(model, X_train, y_train, epochs=EPOCHS, learning_rate=ALPHA)
