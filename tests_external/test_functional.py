"""
title : test_functional.py
create : @tarickali 23/12/17
update : @tarickali 23/12/17
"""

import numpy as np
import tensorflow as tf
import brain.functional as F


def test_functional():
    for _ in range(50):
        x = np.random.randn(32, 10)

        # identity #
        y = F.identity(x)
        z = tf.keras.layers.Identity()(x)
        assert np.allclose(y.data.array, z.numpy())

        # affine #
        y = F.affine(x, 2, 3)
        z = tf.constant(x) * 2 + 3
        assert np.allclose(y.data.array, z.numpy())

        # relu #
        y = F.relu(x)
        z = tf.keras.activations.relu(x)
        assert np.allclose(y.data.array, z.numpy())

        y = F.relu(x, 0.2)
        z = tf.keras.activations.relu(x, 0.2)
        assert np.allclose(y.data.array, z.numpy())

        # sigmoid #
        y = F.sigmoid(x)
        z = tf.keras.activations.sigmoid(x)
        assert np.allclose(y.data.array, z.numpy())

        # tanh #
        y = F.tanh(x)
        z = tf.keras.activations.tanh(x)
        assert np.allclose(y.data.array, z.numpy())

        # elu #
        y = F.elu(x, 0.2)
        z = tf.keras.activations.elu(x, 0.2)
        assert np.allclose(y.data.array, z.numpy())

        # selu #
        y = F.selu(x)
        z = tf.keras.activations.selu(x)
        assert np.allclose(y.data.array, z.numpy())

        # softplus #
        y = F.softplus(x)
        z = tf.keras.activations.softplus(x)
        assert np.allclose(y.data.array, z.numpy())

        # softmax #
        y = F.softmax(x)
        z = tf.keras.activations.softmax(tf.constant(x))
        assert np.allclose(y.data.array, z.numpy())


if __name__ == "__main__":
    test_functional()
