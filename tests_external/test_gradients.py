"""
title : test_gradients.py
create : @tarickali 23/12/17
update : @tarickali 23/12/17
"""

import numpy as np
import torch

import torch.nn.functional as G
from brain.core import Node
import brain.math as bmath
import brain.functional as F


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
    node_y = F.tanh(node_z)
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
    torch_y = G.tanh(torch_x @ torch_W + torch_b)
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


if __name__ == "__main__":
    test_gradients()
