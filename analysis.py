import json

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

strategies = ["baseline", "noise", "pessimistic", "noise_pessimistic"]
labels = ["Baseline", "Noise", "Pessimistic", "Both"]
budgets = [300000, 1000000]
seeds = range(40)
pole_lengths = np.linspace(0.4, 1.8, 30)


def load(kind, strategy, total_steps, seed):
    name = strategy + "_" + str(total_steps // 1000) + "k_seed" + str(seed)
    with open("results/" + kind + "_" + name + ".json") as f:
        return json.load(f)


# Two-sided permutation test on the difference between the mean log10 scores
def permutation_test(a, b, n=20000):
    observed = abs(np.mean(a) - np.mean(b))
    both = np.concatenate([a, b])
    count = 0
    for i in range(n):
        np.random.shuffle(both)
        if abs(np.mean(both[:len(a)]) - np.mean(both[len(a):])) >= observed:
            count += 1
    return (count + 1) / (n + 1)


# 95% bootstrap interval of the ratio between the geometric mean scores
def bootstrap_ratio(a, b, n=10000):
    ratios = []
    for i in range(n):
        ratios.append(10 ** (np.mean(np.random.choice(a, len(a))) - np.mean(np.random.choice(b, len(b)))))
    return np.percentile(ratios, [2.5, 97.5])


# Holm correction: the smallest p-value is multiplied by m, the next one by m - 1, and so on
def holm(p_values):
    order = np.argsort(p_values)
    adjusted = [0] * len(p_values)
    previous = 0
    for k in range(len(order)):
        previous = max(previous, min(1, (len(order) - k) * p_values[order[k]]))
        adjusted[order[k]] = previous
    return adjusted


# test score (mean over the 30 pole lengths) of every run
scores = {}
for total_steps in budgets:
    for strategy in strategies:
        values = []
        for seed in seeds:
            values.append(load("test", strategy, total_steps, seed)["overall_avg"])
        scores[strategy, total_steps] = np.array(values)


# Table 1: every strategy against the baseline with the same number of training steps
for total_steps in budgets:
    base = np.log10(scores["baseline", total_steps])
    p_values = []
    for strategy in strategies[1:]:
        p_values.append(permutation_test(np.log10(scores[strategy, total_steps]), base))
    p_holm = holm(p_values)

    for i in range(len(strategies)):
        s = scores[strategies[i], total_steps]
        # checkpoints with the highest possible validation score (2 x 10000), so validation can not rank them
        saturated = 0
        for seed in seeds:
            for step, score in load("train", strategies[i], total_steps, seed)["validation"]:
                if score == 20000:
                    saturated += 1
        line = "{}k {}: mean {:.0f}, median {:.0f}, worst {:.0f}, sd log10 {:.3f}, saturated checkpoints per run {:.1f}".format(total_steps // 1000, strategies[i], np.mean(s), np.median(s), np.min(s), np.std(np.log10(s)), saturated / len(seeds))
        if i > 0:
            interval = bootstrap_ratio(np.log10(s), base)
            line += ", ratio {:.2f} [{:.2f}, {:.2f}], p {:.4f}, Holm p {:.4f}".format(10 ** (np.mean(np.log10(s)) - np.mean(base)), interval[0], interval[1], p_values[i - 1], p_holm[i - 1])
        print(line)

    # 2x2 design: runs with the change against runs without it
    for change in ["noise", "pessimistic"]:
        with_change = []
        without_change = []
        for strategy in strategies:
            if change in strategy:
                with_change.extend(np.log10(scores[strategy, total_steps]))
            else:
                without_change.extend(np.log10(scores[strategy, total_steps]))
        print("{}k main effect of {}: ratio {:.2f}, p {:.4f}".format(total_steps // 1000, change, 10 ** (np.mean(with_change) - np.mean(without_change)), permutation_test(np.array(with_change), np.array(without_change))))


# longer training, for every strategy
for strategy in strategies:
    short = np.log10(scores[strategy, 300000])
    long = np.log10(scores[strategy, 1000000])
    print("{}: 1000k against 300k ratio {:.2f}, p {:.4f}".format(strategy, 10 ** (np.mean(long) - np.mean(short)), permutation_test(long, short)))


# behaviour in the first 1000 test steps (behaviour.py)
for total_steps in budgets:
    all_x = []
    all_falls = []
    all_scores = []
    for strategy in strategies:
        x = []
        theta = []
        falls = []
        for seed in seeds:
            result = load("behaviour", strategy, total_steps, seed)
            x.append(result["normal_mean_abs"][0])
            theta.append(result["normal_mean_abs"][2])
            falls.append(result["falls_before_1000"])
        print("{}k {}: mean |x| {:.2f}, mean |theta| {:.4f}, falls before step 1000 {:.1f}%".format(total_steps // 1000, strategy, np.mean(x), np.mean(theta), 100 * np.sum(falls) / (len(seeds) * 30 * 2)))
        all_x.extend(x)
        all_falls.extend(falls)
        all_scores.extend(np.log10(scores[strategy, total_steps]))
    print("{}k correlation with the log10 score: mean |x| {:.2f}, falls {:.2f}".format(total_steps // 1000, np.corrcoef(all_x, all_scores)[0, 1], np.corrcoef(all_falls, all_scores)[0, 1]))


# Figure 1: test score of every run, the line is the mean
plt.figure(figsize=(3.4, 2.4))
for i in range(len(strategies)):
    for j in range(len(budgets)):
        s = scores[strategies[i], budgets[j]]
        x = i - 0.2 + 0.4 * j
        plt.scatter(x + np.random.uniform(-0.12, 0.12, len(s)), s, s=4, alpha=0.5, color=['blue', 'orange'][j])
        plt.plot([x - 0.15, x + 0.15], [np.mean(s), np.mean(s)], color='black')
plt.yscale('log')
plt.minorticks_off()
plt.xticks(range(len(strategies)), labels, fontsize=8)
plt.yticks([5000, 10000, 20000, 40000], ["5000", "10000", "20000", "40000"], fontsize=7)
plt.ylabel('Test score', fontsize=8)
plt.scatter([], [], s=4, color='blue', label='300k steps')
plt.scatter([], [], s=4, color='orange', label='1M steps')
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("paper/scores.pdf")


# Appendix: bar plot of the network we submit for each strategy, the same plot as bar_plot in test_script.py
plt.figure(figsize=(12, 9))
for i in range(len(strategies)):
    s = scores[strategies[i], 1000000]
    seed = np.argsort(s)[len(s) // 2] # the run with the median test score
    result = load("test", strategies[i], 1000000, seed)
    plt.subplot(2, 2, i + 1)
    plt.bar(range(len(pole_lengths)), result["avg"], yerr=result["std"], capsize=2, alpha=0.7)
    plt.xticks(range(len(pole_lengths)), [round(length, 2) for length in pole_lengths], rotation=90)
    plt.xlabel('Pole length')
    plt.ylabel('Episode length')
    plt.title('{} (weights/{}_1000k_seed{}.pth)\nAverage score over all pole lengths = {}'.format(labels[i], strategies[i], seed, round(result["overall_avg"])))
plt.tight_layout()
plt.savefig("paper/bar_plots.pdf")
