# MAIN-XR-MD Phase-0 (JAX): Average-Reward Actor–Critic + Gymnasium outward, Gymnax/Craftax inward

This repo implements the first logical experiments with a **JAX/Flax** driver, **Gymnasium** outward API,
**Gymnax/Craftax** inner stepping, and **RLDS-lite JSONL** logging. It is estimator-correct for average reward and
designed for high throughput (JIT + vmap + scan).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train (Craftax via Gymnax, JAX driver)
python -m mxrmd_jax.train_jax --env craftax --env-id craftax-classic-v1   --num-envs 2048 --unroll 64 --total-frames 10_000_000 --r-reset 50

# Evaluate (Gymnasium outward) and log RLDS-lite
python -m mxrmd_jax.eval_gym --env craftax --env-id craftax-classic-v1 --episodes 200   --r-reset 50 --log runs/latest/eval_rlds.jsonl
```

## Optional extras (strict policy)
Any deviation from the mainstream direction is an **optional extra**:

- Procgen / FruitBot parity: `pip install -e .[procgen]`
- Multi-agent (PettingZoo): `pip install -e .[marl]`
- EnvPool vectorization: `pip install -e .[envpool]`
- Dev tooling: `pip install -e .[dev]`

Core experiments **do not** require these extras.
