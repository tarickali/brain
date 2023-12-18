import numpy as np
from brain.core import Node
from brain.modules import Linear, Flatten
from brain.models import Sequential


def main():
    model = Sequential(
        [
            Linear(16, 32, "relu"),
            Linear(32, 32, "relu"),
            Linear(32, 10, "sigmoid"),
            Flatten(),
        ]
    )
    X = Node(np.random.randn(32, 16))

    y = model(X)
    print(y)

    y.backward()

    print(model.parameters[0]["W"])


if __name__ == "__main__":
    main()
