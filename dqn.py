import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import gym


# This is the 'standard' neural network, copied from test_script.py (do not change)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay buffer, FIFO with a fixed capacity
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for transition in batch:
            states.append(transition[0])
            actions.append(transition[1])
            rewards.append(transition[2])
            next_states.append(transition[3])
            terminals.append(transition[4])

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        terminals = torch.tensor(terminals, dtype=torch.float32)
        return states, actions, rewards, next_states, terminals

    def __len__(self):
        return len(self.memory)


# Epsilon-greedy policy for action selection
def select_action(q_network, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    else:
        with torch.no_grad():
            q_values = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return q_values.argmax().item()


# std of the observation noise for cart position, cart velocity, pole angle, pole angular velocity
NOISE_STD = np.array([0.05, 0.05, 0.005, 0.05], dtype=np.float32)


def add_noise(state, noise_level):
    # the agent sees a noisy state, the environment itself does not change
    if noise_level == 0:
        return np.array(state, dtype=np.float32)
    noise = np.random.normal(0, 1, 4) * NOISE_STD * noise_level
    return np.array(state + noise, dtype=np.float32)


# One gradient step on a random minibatch from the replay buffer
def update(policy_net, target_net, optimizer, memory, batch_size, gamma, alpha):
    states, actions, rewards, next_states, terminals = memory.sample(batch_size)

    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1) # Q(s,a)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        # pessimistic target: alpha = 0 is normal DQN, alpha > 0 also counts the worst next action
        next_value = (1 - alpha) * next_q_values.max(1)[0] + alpha * next_q_values.min(1)[0]
        td_target = rewards + gamma * (1 - terminals) * next_value

    loss = nn.functional.smooth_l1_loss(current_q, td_target) # huber loss
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()


# Validation, never uses the test protocol: normal force and pole lengths that are not in the test grid
VAL_LENGTHS = np.linspace(0.45, 1.75, 8)


def validate(q_network, pushes):
    total_steps = 0
    for i in range(len(VAL_LENGTHS)):
        env = gym.make("CartPole-v1")
        env.unwrapped.length = VAL_LENGTHS[i]
        state, _ = env.reset(seed=10000 + i)
        rng = random.Random(10000 + i)
        steps = 0
        push_left = 0
        push_action = 0
        terminated = False

        while not terminated and steps < 2000: # the 500 step time limit is ignored, like in the test

            if pushes and steps >= 100 and steps % 100 == 0:
                push_left = 3
                push_action = rng.randint(0, 1)

            if push_left > 0: # forced push, the agent does not choose
                action = push_action
                push_left -= 1
            else:
                action = select_action(q_network, state, 0)

            state, reward, terminated, truncated, _ = env.step(action)
            steps += 1

        total_steps += steps
        env.close()

    return total_steps / len(VAL_LENGTHS)
