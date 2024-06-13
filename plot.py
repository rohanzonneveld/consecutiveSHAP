import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 18

def plot_metrics(dataset='SST-2'):
    # load the data
    aopc_mobius = pd.read_csv(f'experiments/MöbiusHEDGE/{dataset}/aopc.csv', header=None)
    aopc_gemfix = pd.read_csv(f'experiments/CompleteMöbius/{dataset}/aopc.csv', header=None)
    aopc_hedge = pd.read_csv(f'experiments/HEDGE/{dataset}/aopc.csv', header=None)
    aopc_timeshap = pd.read_csv(f'experiments/TimeSHAP/{dataset}/aopc.csv', header=None)
    log_odds_mobius = pd.read_csv(f'experiments/MöbiusHEDGE/{dataset}/log_odds.csv', header=None)
    log_odds_gemfix = pd.read_csv(f'experiments/CompleteMöbius/{dataset}/log_odds.csv', header=None)
    log_odds_hedge = pd.read_csv(f'experiments/HEDGE/{dataset}/log_odds.csv', header=None)
    log_odds_timeshap = pd.read_csv(f'experiments/TimeSHAP/{dataset}/log_odds.csv', header=None)

    methods = ['HIMEX', 'GEM-FIX', 'HEDGE', 'TimeSHAP']
    # plot the data
    x = np.arange(5,55,5)
    fig = plt.figure(figsize=(8, 8))
    plt.plot(x, aopc_mobius, marker='o')
    plt.plot(x, aopc_gemfix, marker='o')
    plt.plot(x, aopc_hedge, marker='o')
    plt.plot(x, aopc_timeshap, marker='o')
    # plt.set_xticklabels(x)
    plt.xlim(4.9, 50.1)
    plt.xticks(x)
    plt.xlabel('k')
    plt.ylabel('AOPC')
    plt.ylim(bottom=0)
    plt.legend(methods)

    plt.savefig(f'aopc_{dataset}.png')

    fig = plt.figure(figsize=(8, 8))
    plt.plot(x, log_odds_mobius, marker='o')
    plt.plot(x, log_odds_gemfix, marker='o')
    plt.plot(x, log_odds_hedge, marker='o')
    plt.plot(x, log_odds_timeshap, marker='o')
    # plt.set_xticklabels(x)
    plt.xlim(4.9, 50.1)
    plt.xticks(x)
    plt.xlabel('k')
    plt.ylabel('LOR')
    plt.ylim(top=0)
    plt.legend(methods)

    # save the plot
    plt.savefig(f'log_odds_{dataset}.png')
    plt.show()

if __name__ == '__main__':
    plot_metrics()