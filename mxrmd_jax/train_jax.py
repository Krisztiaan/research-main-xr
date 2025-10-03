from __future__ import annotations
import argparse, os, time
import numpy as np
import jax, jax.numpy as jnp
from flax.core.frozen_dict import freeze
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm

from mxrmd_jax.driver.model_flax import GRUAC
from mxrmd_jax.driver.avg_reward_ac import ACState, build_optim, update
from mxrmd_jax.logging.rlds_jsonl import RLDSWriter, Step

# Gymnax/Craftax imports (optional)
try:
    from craftax import make_craftax_env_from_name
except Exception:
    make_craftax_env_from_name = None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='craftax', choices=['craftax'])
    p.add_argument('--env-id', type=str, default='craftax-classic-v1')
    p.add_argument('--num-envs', type=int, default=2048)
    p.add_argument('--unroll', type=int, default=64)
    p.add_argument('--total-frames', type=int, default=10_000_000)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--entropy-coef', type=float, default=0.01)
    p.add_argument('--lam', type=float, default=0.9)
    p.add_argument('--r-reset', type=int, default=50)
    p.add_argument('--run-dir', type=str, default='runs/latest')
    return p.parse_args()

def make_survival_env(env_id: str, r_reset: int):
    env = make_craftax_env_from_name(env_id, auto_reset=True)
    params = env.default_params

    def reset_fn(key):
        obs, state = env.reset(key, params)
        reset_phase = jnp.zeros((), dtype=jnp.int32)
        return (state, reset_phase, key, obs), obs

    def step_fn(carry, action):
        state, reset_phase, key, obs = carry

        def in_reset(_):
            rp = jnp.maximum(reset_phase - 1, 0)
            obs0 = jnp.zeros_like(obs)
            return (state, rp, key, obs0), (obs0, 0.0, False)

        def alive(_):
            key1, key2 = jax.random.split(key)
            obs1, st, rew_env, done, info = env.step(key1, state, action, params)
            rp = jnp.where(done, jnp.array(r_reset, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
            return (st, rp, key2, obs1), (obs, 1.0, done)

        carry_next, out = jax.lax.cond(reset_phase > 0, in_reset, alive, operand=None)
        return carry_next, out

    return reset_fn, step_fn, params, env

def main():
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    assert make_craftax_env_from_name is not None, "Install craftax>=1.5 and import make_craftax_env_from_name."
    key = jax.random.PRNGKey(args.seed)
    reset_fn, step_fn, params, env = make_survival_env(args.env_id, args.r_reset)

    # Vectorize over num_envs
    v_reset = jax.vmap(reset_fn)
    v_step = jax.vmap(step_fn, in_axes=((0, 0, 0, 0), 0))

    # Reset batch
    keys = jax.random.split(key, args.num_envs)
    carry, obs0 = v_reset(keys)

    # Model init
    num_actions = env.action_space(params).n
    model = GRUAC(num_actions=num_actions)
    params_init = model.init(jax.random.PRNGKey(args.seed+123), obs0, jnp.zeros((args.num_envs, 256)))['params']
    opt = build_optim(args.lr)
    opt_state = opt.init(params_init)
    state = ACState(params_init, jnp.zeros((args.num_envs, 256)), opt_state, jnp.array(0.0), key)

    @jax.jit(donate_argnums=(0, 1))
    def rollout_update(state: ACState, carry):
        # carry = (state_env, reset_phase, key, obs_t)
        def one_step(carry, _):
            st, rp, k, obs_t, h_t = carry
            logits, v, h1 = model.apply({'params': state.params}, obs_t, h_t)
            k, k2 = jax.random.split(k)
            a = jax.random.categorical(k, logits, axis=-1)
            (st1, rp1, k3, obs_next), (obs_out, rew, done) = v_step((st, rp, k2, obs_t), a)
            return (st1, rp1, k3, obs_next, h1), (obs_out, a, rew, done, logits, v)

        st, rp, k, obs0 = carry
        carry0 = (st, rp, k, obs0, state.h)

        (stT, rpT, kT, obsT, hT), (obs_seq, act_seq, rew_seq, done_seq, logits_seq, val_seq) = \
            jax.lax.scan(one_step, carry0, None, length=args.unroll)

        batch = (obs_seq, act_seq, rew_seq, done_seq)
        new_state, metrics = update(state, model.apply, batch, opt,
                                    lam=args.lam, entropy_coef=args.entropy_coef, lr_rho=1e-3)
        new_state = new_state._replace(h=hT)
        new_carry = (stT, rpT, kT, obsT)
        return new_state, new_carry, metrics

    frames = 0
    total_frames = args.total_frames
    carry_state = carry

    pbar = tqdm(total=total_frames, desc="Training (JAX)")
    while frames < total_frames:
        state, carry_state, metrics = rollout_update(state, carry_state)
        frames += args.unroll * args.num_envs
        pbar.update(args.unroll * args.num_envs)
    pbar.close()

    # Save checkpoint (msgpack via flax.serialization)
    import flax
    ckpt = {
        'params': jax.device_get(state.params),
        'rho': float(state.rho),
        'config': vars(args)
    }
    import msgpack, msgpack_numpy
    msgpack_numpy.patch()
    with open(os.path.join(args.run_dir, 'checkpoint.msgpack'), 'wb') as f:
        f.write(msgpack.packb(ckpt))

    print("Saved:", os.path.join(args.run_dir, 'checkpoint.msgpack'))

if __name__ == '__main__':
    main()
