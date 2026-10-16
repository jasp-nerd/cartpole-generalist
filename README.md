# Observation Noise and Pessimistic Targets

Code for Assignment 3 of Introduction to Reinforcement Learning, VU Amsterdam, Team 11.

We train a deep Q-network to balance CartPole-v1 for pole lengths from 0.4 to 1.8. The test script makes the push force much stronger after step 500, and the agent never sees that force during training. We ask whether robustness to this unseen force comes from what the agent observes, from what it optimizes, or from both.

## Files

- dqn.py: the standard network (copied from the test script), replay buffer, epsilon-greedy action selection, observation noise, the DQN update and validation
- train.py: trains one strategy for a given number of steps and seeds, and keeps the checkpoint with the best validation score
- evaluate.py: runs the official test (test_pole_length from test_script.py, unchanged) on trained weights, with fixed start states
- behaviour.py: records cart position, pole angle and early falls during the first 1000 test steps
- analysis.py: statistics for the paper (geometric mean ratios, bootstrap intervals, permutation tests with Holm correction, 2x2 main effects)
- test_script.py: the test script provided by the course
- weights/: the submitted network for each strategy
- results/: training logs, test scores and behaviour results for every run (4 strategies x 2 budgets x 40 seeds)
- paper/: LaTeX source and figures of the report

## Installation

    pip install -r requirements.txt

We used Python 3.12, PyTorch 2.14.1 and Gym 0.26.2.

## Reproducing the results

Run all commands from the repository root. Strategy names are baseline, noise, pessimistic and noise_pessimistic. The paper uses seeds 0 to 39 and budgets of 300000 and 1000000 steps.

1. Train (saves weights/<strategy>_<steps>k_seed<seed>.pth and results/train_*.json):

    python train.py noise 1000000 0 1 2 3

2. Test (saves results/test_*.json):

    python evaluate.py noise 1000000 0 1 2 3

3. Behaviour (saves results/behaviour_*.json):

    python behaviour.py noise 1000000 0 1 2 3

4. Statistics (prints every number in the paper and saves both figures):

    python analysis.py


