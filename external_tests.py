"""
title : external_tests.py
create : @tarickali 23/12/17
update : @tarickali 23/12/17
"""

import numpy as np
import torch
import tensorflow as tf

import torch.nn.functional as F
from brain.core import Node
import brain.math as bmath
import brain.functional as G


def test_gradients():
    W = np.random.randn(10, 16)
    x = np.random.randn(32, 10)
    b = np.zeros(16)
    t = np.random.randn(32, 16)

    node_W = Node(W)
    node_x = Node(x)
    node_b = Node(b)
    node_t = Node(t)
    node_z = node_x @ node_W + node_b
    node_y = G.tanh(node_z)
    node_loss = bmath.mean(node_t - node_y)
    # print(node_loss.data)
    # node_loss = fun.mean(node_loss, axis=0)
    # node_loss = fun.sum(node_t - node_y)
    node_loss.backward()

    torch_W = torch.Tensor(W)
    torch_W.requires_grad = True
    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    torch_b = torch.Tensor(b)
    torch_b.requires_grad = True
    torch_t = torch.Tensor(t)
    torch_t.requires_grad = True
    torch_y = F.tanh(torch_x @ torch_W + torch_b)
    torch_y.retain_grad()

    torch_loss = torch.mean(torch_t - torch_y)
    # torch_loss = torch.mean(torch_loss, axis=0)
    # torch_loss = torch.sum(torch_t - torch_y)
    torch_loss.backward()

    print("node loss", node_loss)
    print("torch loss", torch_loss)

    # print("node y", node_y.grad)
    # print("torch y", torch_y.grad)

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
        # print("max", np.max(np.abs(node.grad.array - tensor.grad.numpy())))
        # assert np.allclose(node.data.array, tensor.data.numpy(), atol=1e-4)
        assert np.allclose(node.grad.array, tensor.grad.numpy(), atol=1e-4)


def test_log():
    W = np.random.uniform(0.1, 10.0, (10, 16))
    x = np.random.uniform(0.1, 10.0, (32, 10))
    b = np.zeros(16)
    t = np.random.uniform(10.0, 20.0, (32, 16))

    node_W = Node(W)
    node_x = Node(x)
    node_b = Node(b)
    node_t = Node(t)
    node_y = bmath.log(node_x @ node_W + node_b)
    node_loss = bmath.sum(node_t - node_y) / 32
    node_loss.backward()

    torch_W = torch.Tensor(W)
    torch_W.requires_grad = True
    torch_x = torch.Tensor(x)
    torch_x.requires_grad = True
    torch_b = torch.Tensor(b)
    torch_b.requires_grad = True
    torch_t = torch.Tensor(t)
    torch_t.requires_grad = True
    torch_y = torch.log(torch_x @ torch_W + torch_b)
    torch_loss = torch.sum(torch_t - torch_y) / 32
    torch_loss.backward()

    for node, tensor in [(node_W, torch_W), (node_x, torch_x), (node_b, torch_b)]:
        assert np.allclose(node.grad.array, tensor.grad.numpy())


def test_functional():
    x = np.random.randn(32, 10)

    # identity #
    y = G.identity(x)
    z = tf.keras.layers.Identity()(x)
    assert np.allclose(y.data.array, z.numpy())

    # affine #
    y = G.affine(x, 2, 3)
    z = tf.constant(x) * 2 + 3
    assert np.allclose(y.data.array, z.numpy())

    # relu #
    y = G.relu(x)
    z = tf.keras.activations.relu(x)
    assert np.allclose(y.data.array, z.numpy())

    y = G.relu(x, 0.2)
    z = tf.keras.activations.relu(x, 0.2)
    assert np.allclose(y.data.array, z.numpy())

    # sigmoid #
    y = G.sigmoid(x)
    z = tf.keras.activations.sigmoid(x)
    assert np.allclose(y.data.array, z.numpy())

    # tanh #
    y = G.tanh(x)
    z = tf.keras.activations.tanh(x)
    assert np.allclose(y.data.array, z.numpy())

    # elu #
    y = G.elu(x, 0.2)
    z = tf.keras.activations.elu(x, 0.2)
    assert np.allclose(y.data.array, z.numpy())

    # selu #
    y = G.selu(x)
    z = tf.keras.activations.selu(x)
    assert np.allclose(y.data.array, z.numpy())

    # softplus #
    y = G.softplus(x)
    z = tf.keras.activations.softplus(x)
    assert np.allclose(y.data.array, z.numpy())

    # softmax #
    y = G.softmax(x)
    z = tf.keras.activations.softmax(tf.constant(x))
    assert np.allclose(y.data.array, z.numpy())


if __name__ == "__main__":
    # test_functional()
    test_gradients()
