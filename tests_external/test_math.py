"""
title : test_math.py
create : @tarickali 23/12/17
update : @tarickali 23/12/17
"""

import numpy as np
import torch
from brain.core import Node
import brain.math as bmath


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


if __name__ == "__main__":
    test_log()
