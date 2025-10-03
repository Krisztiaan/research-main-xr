from __future__ import annotations
import jax, jax.numpy as jnp
from flax import linen as nn
from typing import Tuple

class ImpalaConv(nn.Module):
    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        # Expect x in [B,H,W,C] or [H,W,C]
        if x.ndim == 3:
            x = x[None, ...]
        x = nn.Conv(16, (3,3), strides=(1,1), padding='SAME', kernel_init=kernel_init)(x); x = nn.relu(x)
        x = nn.Conv(16, (3,3), strides=(1,1), padding='SAME', kernel_init=kernel_init)(x); x = nn.relu(x)
        x = nn.max_pool(x, (2,2), (2,2))

        x = nn.Conv(32, (3,3), strides=(1,1), padding='SAME', kernel_init=kernel_init)(x); x = nn.relu(x)
        x = nn.Conv(32, (3,3), strides=(1,1), padding='SAME', kernel_init=kernel_init)(x); x = nn.relu(x)
        x = nn.max_pool(x, (2,2), (2,2))

        x = nn.Conv(32, (3,3), strides=(1,1), padding='SAME', kernel_init=kernel_init)(x); x = nn.relu(x)
        x = nn.Conv(32, (3,3), strides=(1,1), padding='SAME', kernel_init=kernel_init)(x); x = nn.relu(x)
        x = nn.max_pool(x, (2,2), (2,2))

        x = x.reshape((x.shape[0], -1))
        dense_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        x = nn.Dense(256, kernel_init=dense_init)(x); x = nn.relu(x)
        return x

class GRUAC(nn.Module):
    num_actions: int
    @nn.compact
    def __call__(self, x, h):
        feat = ImpalaConv()(x)
        gru_kernel = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        h1, _ = nn.GRUCell(256, kernel_init=gru_kernel,
                           recurrent_kernel_init=gru_kernel)(h, feat)
        dense_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        logits = nn.Dense(self.num_actions, kernel_init=dense_init)(h1)
        value = nn.Dense(1, kernel_init=dense_init)(h1)
        return logits, value.squeeze(-1), h1
