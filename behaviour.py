import os
import sys
import json

import numpy as np
import torch
import gym

from dqn import QNetwork

torch.set_num_threads(1)

pole_lengths = np.linspace(0.4, 1.8, 30)
runs_per_length = 2


# Same force rule as test_pole_length in test_script.py, but we keep the states and stop at step 1000
def run_episode(env, q_network):
    state = env.reset()[0]
    states = []
    steps = 0
    terminated = False

    while not terminated and steps < 1000:
        with torch.no_grad():
            action = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        states.append(state)
        steps += 1

        if steps >= 500 and steps % 25 == 0: # force 75 from step 500, like in the test
            env.unwrapped.force_mag = 75

    return np.array(states), steps


# How far the agent lets the cart and the pole move, before (steps 1-500) and during (501-1000) the force of 75
def behaviour(strategy, total_steps, seed):

    name = strategy + "_" + str(total_steps // 1000) + "k_seed" + str(seed)
    model = QNetwork(4, 2)
    model.load_state_dict(torch.load("weights/" + name + ".pth", weights_only=True))
    model.eval()

    normal_states = []
    strong_states = []
    falls = 0

    for i in range(len(pole_lengths)):
        for x in range(runs_per_length):
            env = gym.make("CartPole-v1")
            env.unwrapped.length = pole_lengths[i]
            env.reset(seed=200000 + 100 * i + x) # other start states than evaluate.py
            states, steps = run_episode(env, model)
            normal_states.append(np.abs(states[:500]))
            strong_states.append(np.abs(states[500:]))
            if steps < 1000:
                falls += 1
            env.close()

    normal_states = np.concatenate(normal_states)
    strong_states = np.concatenate(strong_states)

    result = {
        "strategy": strategy,
        "seed": seed,
        "total_steps": total_steps,
        "normal_mean_abs": np.mean(normal_states, axis=0).tolist(), # mean |x|, |x_dot|, |theta|, |theta_dot|
        "strong_mean_abs": np.mean(strong_states, axis=0).tolist(),
        "falls_before_1000": falls,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/behaviour_" + name + ".json", "w") as f:
        json.dump(result, f)

    print("{} |x| {:.3f} -> {:.3f}, |theta| {:.4f} -> {:.4f}".format(name, result["normal_mean_abs"][0], result["strong_mean_abs"][0], result["normal_mean_abs"][2], result["strong_mean_abs"][2]))


# usage: python behaviour.py <strategy> <total_steps> <seed> [<seed> ...]
if __name__ == "__main__":
    strategy = sys.argv[1]
    total_steps = int(sys.argv[2])
    for seed in sys.argv[3:]:
        behaviour(strategy, total_steps, int(seed))
