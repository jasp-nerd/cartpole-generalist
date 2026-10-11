import os
import sys
import time
import json

import numpy as np
import torch
import gym

# the official test, unchanged
from test_script import QNetwork, test_pole_length

torch.set_num_threads(1)

pole_lengths = np.linspace(0.4, 1.8, 30)
runs_per_length = 10


def evaluate(strategy, total_steps, seed):

    name = strategy + "_" + str(total_steps // 1000) + "k_seed" + str(seed)
    model = QNetwork(4, 2)
    model.load_state_dict(torch.load("weights/" + name + ".pth", weights_only=True))
    model.eval()

    avg_scores = []
    std_scores = []
    start_time = time.time()

    for i in range(len(pole_lengths)):
        pole_score = []
        for x in range(runs_per_length):
            env = gym.make("CartPole-v1")
            env.unwrapped.length = pole_lengths[i]
            env.reset(seed=100000 + 100 * i + x) # seeds the start states so the test can be repeated
            with torch.no_grad():
                score = test_pole_length(env, model)
            pole_score.append(score)
            env.close()
        avg_scores.append(np.mean(pole_score))
        std_scores.append(np.std(pole_score))

    result = {
        "strategy": strategy,
        "seed": seed,
        "total_steps": total_steps,
        "pole_lengths": list(pole_lengths),
        "avg": avg_scores,
        "std": std_scores,
        "overall_avg": np.mean(avg_scores),
        "test_time": time.time() - start_time,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/test_" + name + ".json", "w") as f:
        json.dump(result, f)

    print("{} average score over all pole lengths = {}".format(name, round(np.mean(avg_scores))))
    return avg_scores, std_scores


# usage: python evaluate.py <strategy> <total_steps> <seed> [<seed> ...]
if __name__ == "__main__":
    strategy = sys.argv[1]
    total_steps = int(sys.argv[2])
    for seed in sys.argv[3:]:
        evaluate(strategy, total_steps, int(seed))
