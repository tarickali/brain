"""
title : test_losses.py
create : @tarickali 23/12/18
update : @tarickali 23/12/18
"""

import numpy as np
import torch
import torch.nn.functional as G
from brain.core import Node
import brain.functional as F
import brain.losses as L
import brain.math as M

from tensorflow import keras


def get_data(input_dim: int = 32, output_dim: int = 1):
    np.random.seed()
    x = np.random.randn(32, input_dim)
    W = np.random.randn(input_dim, output_dim)
    b = np.zeros(output_dim)
    y = np.random.randn(32, output_dim)

    node_W = Node(W)
    node_x = Node(x)
    node_b = Node(b)
    node_z = node_x @ node_W + node_b

    torch_W = torch.Tensor(W)
    torch_W.requires_grad = True
    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    torch_b = torch.Tensor(b)
    torch_b.requires_grad = True
    torch_z = torch_x @ torch_W + torch_b

    return (node_x, node_W, node_b, node_z), (torch_x, torch_W, torch_b, torch_z), y


def test_regression_losses():
    (
        (node_x, node_W, node_b, node_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    ) = get_data(32, 16)

    node_o = F.relu(node_z)
    node_loss = L.MeanAbsoluteError()(Node(y), node_o)
    node_loss.backward()

    torch_o = G.relu(torch_z)
    torch_loss = torch.nn.L1Loss()(torch.Tensor(y), torch_o)
    torch_loss.backward()

    # print("node loss", node_loss)
    # print("torch loss", torch_loss)

    for i, (node, tensor) in enumerate(
        [(node_W, torch_W), (node_x, torch_x), (node_b, torch_b)]
    ):
        # print(i)
        # print("node data", node.data.array, node.shape)
        # print()
        # print("tensor data", tensor.data.numpy(), tensor.shape)
        # print()
        # print("node grad", node.grad.array, node.grad.shape)
        # print()
        # print("tensor grad", tensor.grad.numpy(), tensor.grad.shape)
        # print()
        print("max", np.max(np.abs(node.grad.array - tensor.grad.numpy())))
        assert np.allclose(node.data.array, tensor.data.numpy(), atol=1e-6)
        assert np.allclose(node.grad.array, tensor.grad.numpy(), atol=1e-6)


def test_binary_loss():
    (
        (node_x, node_W, node_b, node_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    ) = get_data(32, 1)

    y = np.random.randint(0, 2, (32, 1))
    # print(y)

    node_o = F.sigmoid(node_z)
    node_loss = L.BinaryCrossentropy(logits=False)(Node(y), node_o)
    node_loss.backward()

    torch_o = G.sigmoid(torch_z)
    torch_o.retain_grad()
    torch_loss = torch.nn.BCELoss()(torch_o, torch.Tensor(y))
    torch_loss.retain_grad()
    torch_loss.backward()

    keras_loss = keras.losses.BinaryCrossentropy(False)(y, node_o.data.array)

    # print("node o", node_o)
    # print("torch o", torch_o)

    print("node loss", node_loss)
    print("torch loss", torch_loss)
    print("keras loss", keras_loss)

    # print(node_loss.children)
    # print(node_o.grad)
    # print(torch_o.grad)

    for i, (node, tensor) in enumerate(
        [(node_W, torch_W), (node_x, torch_x), (node_b, torch_b)]
    ):
        print(i)
        # print()
        # print("node grad", node.grad.array, node.grad.shape)
        # print()
        # print("tensor grad", tensor.grad.numpy(), tensor.grad.shape)
        # print()
        print("max", np.max(np.abs(node.grad.array - tensor.grad.numpy())))
        assert np.allclose(node.data.array, tensor.data.numpy(), atol=1e-6)
        assert np.allclose(node.grad.array, tensor.grad.numpy(), atol=1e-6)


def test_multi_loss():
    (
        (node_x, node_W, node_b, node_z),
        (torch_x, torch_W, torch_b, torch_z),
        y,
    ) = get_data(32, 10)

    y = np.eye(10)[np.random.choice(10, 32)]
    # print(y)

    node_o = node_z
    node_loss = L.CategoricalCrossentropy(logits=True)(Node(y), node_o)
    node_loss.backward()

    torch_o = torch_z
    torch_o.retain_grad()
    torch_loss = torch.nn.CrossEntropyLoss()(torch_o, torch.Tensor(y))
    torch_loss.retain_grad()
    torch_loss.backward()

    # keras_loss = keras.losses.CategoricalCrossentropy(True)(y, node_o.data.array)

    # print("node o", node_o)
    # print("torch o", torch_o)

    # print("node loss", node_loss)
    # print("torch loss", torch_loss)
    # print("keras loss", keras_loss)

    # print(node_loss.children)
    # print(node_o.grad)
    # print(torch_o.grad)

    for i, (node, tensor) in enumerate(
        [(node_W, torch_W), (node_x, torch_x), (node_b, torch_b)]
    ):
        # print(i)
        # print()
        # print("node grad", node.grad.array, node.grad.shape)
        # print()
        # print("tensor grad", tensor.grad.numpy(), tensor.grad.shape)
        # print()
        print("max", np.max(np.abs(node.grad.array - tensor.grad.numpy())))
        assert np.allclose(node.data.array, tensor.data.numpy(), atol=1e-6)
        assert np.allclose(node.grad.array, tensor.grad.numpy(), atol=1e-6)


if __name__ == "__main__":
    for i in range(100):
        print(f"--------------------- {i} --------------------")
        test_regression_losses()
        # test_multi_loss()
        # test_binary_loss()
