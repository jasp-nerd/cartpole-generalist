import os
import sys
import time
import json
import random

import numpy as np
import torch
import torch.optim as optim
import gym

from dqn import QNetwork, ReplayBuffer, select_action, add_noise, update, validate

torch.set_num_threads(1)


# Hyperparameters, the same for every strategy
learning_rate = 0.0005
gamma = 0.99
batch_size = 64
buffer_size = 50000
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 20000  # linear decay over the first 20000 steps
warmup_steps = 1000          # fill the buffer before the first update
target_update = 500          # copy the policy net into the target net every 500 steps
total_steps = 1000000
check_every = 10000          # validate and keep the best weights every 10000 steps
min_length = 0.4
max_length = 1.8

# strategy: [noise_level, alpha]
strategies = {
    "baseline": [0.0, 0.0],
    "noise": [1.0, 0.0],
    "pessimistic": [0.0, 0.05],
    "noise_pessimistic": [1.0, 0.05],
}


def train(strategy, seed):

    noise_level = strategies[strategy][0]
    alpha = strategies[strategy][1]
    name = strategy + "_seed" + str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize the networks, optimizer and replay buffer
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    policy_net = QNetwork(4, 2)
    target_net = QNetwork(4, 2)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer(buffer_size)

    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    step = 0
    episode = 0
    best_score = -1
    best_step = 0
    episode_rewards = []
    pole_lengths = []
    validation_log = []
    start_time = time.time()

    # Training loop
    while step < total_steps:

        # new random pole length every episode, the only change we are allowed to make
        length = random.uniform(min_length, max_length)
        env.unwrapped.length = length
        if episode == 0:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()
        state = add_noise(state, noise_level)
        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated and step < total_steps:

            epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * step / epsilon_decay_steps)
            action = select_action(policy_net, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = add_noise(next_state, noise_level)

            # only store terminated, at the 500 step time limit the pole did not fall so we keep bootstrapping
            memory.push(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            step += 1

            if step >= warmup_steps:
                update(policy_net, target_net, optimizer, memory, batch_size, gamma, alpha)

            if step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # keep the weights with the best validation score, the test script is never used here
            if step % check_every == 0:
                score = validate(policy_net, False) + validate(policy_net, True)
                validation_log.append([step, score])
                if score >= best_score: # on a tie the later checkpoint wins
                    best_score = score
                    best_step = step
                    torch.save(policy_net.state_dict(), "weights/" + name + ".pth")

        episode_rewards.append(total_reward)
        pole_lengths.append(length)
        episode += 1

    env.close()

    log = {
        "strategy": strategy,
        "seed": seed,
        "noise_level": noise_level,
        "alpha": alpha,
        "total_steps": total_steps,
        "best_step": best_step,
        "best_score": best_score,
        "train_time": time.time() - start_time,
        "validation": validation_log,
        "episode_rewards": episode_rewards,
        "pole_lengths": pole_lengths,
    }
    with open("results/train_" + name + ".json", "w") as f:
        json.dump(log, f)

    print("{} done, best validation {} at step {}, {} episodes".format(name, best_score, best_step, episode))
    return episode_rewards, validation_log


# usage: python train.py <strategy> <seed> [<seed> ...]
if __name__ == "__main__":
    strategy = sys.argv[1]
    for seed in sys.argv[2:]:
        train(strategy, int(seed))
