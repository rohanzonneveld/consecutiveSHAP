import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics():
    # load the data
    aopc = pd.read_csv('experiments/HEDGE/bert/aopc.csv', header=None)
    log_odds = pd.read_csv('experiments/HEDGE/bert/log_odds.csv', header=None)

    # plot the data
    x = np.arange(5,55,5)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(x, aopc, marker='o')
    # ax[0].set_xticklabels(x)
    ax[0].set_xlim(5, 50)
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('AOPC')
    ax[0].set_ylim(0, 0.9)

    ax[1].plot(x, log_odds, marker='o')
    # ax[1].set_xticklabels(x)
    ax[1].set_xlim(5, 50)
    ax[1].set_xlabel('r')
    ax[1].set_ylabel('Log Odds')
    ax[1].set_ylim(-2.5, 0)

    # save the plot
    plt.savefig('experiments/HEDGE/bert/metrics.png')
    plt.show()

if __name__ == '__main__':
    plot_metrics()