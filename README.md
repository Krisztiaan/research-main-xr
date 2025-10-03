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

Lower-case `craftax-*` environment identifiers are normalised automatically to Craftax 1.5's canonical `Craftax-*` names, so older scripts continue to run.

## Python Environment (pyenv)
- Install a CPython 3.11 build: `pyenv install 3.11.9`
- Pin the project: `pyenv local 3.11.9` (a `.python-version` file is already checked in)
- Create/refresh the virtualenv: `python -m venv .venv && source .venv/bin/activate`
- Upgrade pip tooling before installing deps: `python -m pip install --upgrade pip wheel setuptools`

## Remote Accelerators (CUDA / TPU)
- Provision a Linux machine with the accelerator you need (e.g. AWS g5, GCP A2, TPU VM).
- Install JAX with the matching backend wheels, for example CUDA 12: `pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- TPU VMs ship with a preconfigured jaxlib; upgrade with `pip install --upgrade jax[tpu]>=0.4.30 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`.
- Reuse the same training command; attach your experiment storage via NFS/GCS/S3 as needed.
- Destroy the VM or pod when the run finishes to keep the “rented as you go” cost model.

## Local CPU Smoke Tests
- For quick checks without an accelerator, force the CPU backend: `JAX_PLATFORMS=cpu python -m mxrmd_jax.train_jax ...`
- Keep `--num-envs` and `--unroll` small (e.g. `--num-envs 4 --unroll 2`) to get feedback in under a minute.

## Optional extras (strict policy)
Any deviation from the mainstream direction is an **optional extra**:

- Procgen / FruitBot parity: `pip install -e .[procgen]`
- Multi-agent (PettingZoo): `pip install -e .[marl]`
- EnvPool vectorization: `pip install -e .[envpool]`
- Dev tooling: `pip install -e .[dev]`

Core experiments **do not** require these extras.
