from __future__ import annotations
import jax, jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from typing import Any, NamedTuple

class ACState(NamedTuple):
    params: Any
    h: jnp.ndarray       # [B, H] hidden state
    opt_state: Any
    rho: jnp.ndarray     # scalar
    rng: jax.random.KeyArray

def build_optim(lr: float):
    return optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr))

def forward_view_weights(delta: jnp.ndarray, lam: float) -> jnp.ndarray:
    # delta: [T, B]
    def scan_fn(carry, x):
        acc = x + lam * carry
        return acc, acc
    # reverse scan over time
    _, w_rev = jax.lax.scan(scan_fn, jnp.zeros(delta.shape[1]), delta[::-1, :])
    return w_rev[::-1, :]

def compute_loss(params, apply_fn, batch, h0, rho, lam, entropy_coef):
    # batch: (obs[T,B,H,W,C], actions[T,B], rewards[T,B], dones[T,B])
    obs, actions, rewards, dones = batch
    T, B = actions.shape

    def step_fn(carry, t):
        h = carry
        logits, values, h1 = apply_fn({'params': params}, obs[t], h)
        # sample actions already taken in batch; compute log-prob
        logp = jax.nn.log_softmax(logits, axis=-1)
        logp_a = jnp.take_along_axis(logp, actions[t][...,None], axis=-1).squeeze(-1)
        probs = jax.nn.softmax(logits, axis=-1)
        ent = -(probs * logp).sum(-1)
        return h1, (logp_a, ent, values, h1)

    hT, (logp_a, ent, values, h_seq) = jax.lax.scan(step_fn, h0, jnp.arange(T))
    # bootstrap value_next with shifted values (continuing task)
    values_next = jnp.concatenate([values[1:], values[-1:]], axis=0)
    delta = rewards - rho + (values_next - values)
    # standardize/clip delta across time-batch for actor only
    mean = jnp.mean(delta); std = jnp.std(delta) + 1e-6
    delta_n = jnp.clip((delta - mean)/std, -5.0, 5.0)
    w = forward_view_weights(delta_n, lam)

    actor_loss = -(w * logp_a).mean() - entropy_coef * ent.mean()
    target = rewards - rho + values_next
    critic_loss = 0.5 * ((values - jax.lax.stop_gradient(target))**2).mean()
    loss = actor_loss + critic_loss
    metrics = {
        'loss': loss, 'actor_loss': actor_loss, 'critic_loss': critic_loss,
        'entropy': ent.mean(), 'delta_mean': mean, 'delta_std': std
    }
    return loss, (metrics, hT, delta)

def update(state: ACState, apply_fn, batch, opt_tx, lam=0.9, entropy_coef=0.01, lr_rho=1e-3):
    (loss, (metrics, hT, delta)), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        state.params, apply_fn, batch, state.h, state.rho, lam, entropy_coef
    )
    updates, opt_state = opt_tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    # rho update with raw delta (not standardized)
    rho = state.rho + lr_rho * jnp.mean(delta)
    return ACState(params, hT, opt_state, rho, state.rng), metrics
