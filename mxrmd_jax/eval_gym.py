from __future__ import annotations
import argparse, os, numpy as np
import gymnasium as gym
from mxrmd_jax.envs.craftax_survival_gym import CraftaxSurvivalGym
from mxrmd_jax.envs.fruitbot_survival_gym import FruitBotSurvivalGym
from mxrmd_jax.logging.rlds_jsonl import RLDSWriter, Step

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='craftax', choices=['craftax','fruitbot'])
    p.add_argument('--env-id', type=str, default='craftax-classic-v1')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--r-reset', type=int, default=50)
    p.add_argument('--log', type=str, default='runs/latest/eval_rlds.jsonl')
    return p.parse_args()

def make_env(args):
    if args.env == 'craftax':
        return CraftaxSurvivalGym(env_id=args.env_id, R_reset=args.r_reset, seed=0)
    else:
        return FruitBotSurvivalGym(R_reset=args.r_reset, distribution='hard', seed=0)

def main():
    args = parse_args()
    env = make_env(args)
    writer = RLDSWriter(args.log)

    for ep in range(args.episodes):
        obs, info = env.reset()
        steps = []
        done = False
        t = 0
        while True:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info2 = env.step(action)
            is_first = (t == 0)
            is_last = bool(info2.get('life_ended', False)) and (env._resets_left == env.R_reset)  # life ended this step
            is_terminal = is_last  # survival is terminal at death for logging
            steps.append(Step(observation=obs.tolist(), action=int(action), reward=float(reward),
                              discount=1.0, is_first=is_first, is_last=is_last, is_terminal=is_terminal,
                              info={'reset_phase': bool(info2.get('reset_phase', False))}))
            obs = next_obs
            t += 1
            if is_last:
                break
        writer.write_episode(seed=0, env_id=args.env, config={'env_id': args.env_id, 'R_reset': args.r_reset}, steps=steps)

    writer.close()
    env.close()
    print(f"Wrote RLDS-lite logs to {args.log}")

if __name__ == '__main__':
    main()
