import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt


def CalculateRanking(acc):
    acc_sorted = sorted(acc)
    acc_sorted = acc_sorted[::-1]
    acc_unique = np.unique(acc)
    acc_unique = sorted(acc_unique)
    acc_unique = acc_unique[::-1]
    ranking = [0, 0, 0, 0, 0, 0, 0, 0]
    for value in acc_unique:
        loc = [i for i,v in enumerate(acc) if v == value]
        rank = [j for j,v in enumerate(acc_sorted) if v == value]
        aver = np.average(rank) + 1
        for l in loc:
            ranking[l] = aver
    return ranking


def AveRanking(results):
    N = results.shape[0]
    k = results.shape[1] - 1
    for i in range(N):
        if i == 0:
            acc = results[i, ]
            temp_rank = CalculateRanking(acc)
            rank = np.array(temp_rank)
        else:
            acc = results[i, :]
            temp_rank = CalculateRanking(acc)
            rank = np.vstack([rank, temp_rank])
    ave_rank = np.average(rank, axis=0)
    return ave_rank


def PlotFriedmanFig(name, ave_rank, cd):
    fig, ax = plt.subplots()
    ax.set_xlim((0, 8))
    ax.set_ylim(0, 5)
    for algorithm in range(len(name)):
        y = 0.5+4/len(name)*algorithm
        ax.annotate(name[algorithm], xy=(0, y))
        ax.plot([ave_rank[algorithm], ave_rank[algorithm]], [y, y], 'k*')
        ax.plot([ave_rank[algorithm]-cd, ave_rank[algorithm]+cd], [y, y], color="k")
    plt.show()


if __name__ == "__main__":

    with open("results/immunotherapy-result.csv", mode='r') as data_file:
        dataSet = pd.read_csv(data_file, header=0, dtype='float')
    data = pd.DataFrame.as_matrix(dataSet[dataSet.columns[1:]])
    fried_chi, p_value = friedmanchisquare(data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7])
    print("fried_chi:")
    print(fried_chi)
    print("p_value:")
    print(p_value)
    ave_rank = AveRanking(data)

    algorithm = ["id3", "id3-pre", "id3-post", "id3-sklearn", "cart", "cart-pre", "cart-post", "cart-sklearn"]
    if p_value > 0.1:
        k = 8
        N = 5
        qtukey = 2.780  # alpha = 0.1
        cd = qtukey * np.sqrt(k * (k + 1) / N / 6)
        print("Algorithm-pairs which are different (p = 0.1):")
        for i in range(k):
            for j in range(i+1, k):
                if abs(ave_rank[i] - ave_rank[j]) > cd:
                    print(algorithm[i]+" and "+algorithm[j])
        PlotFriedmanFig(algorithm, ave_rank, cd)
    elif p_value > 0.05:
        k = 8
        N = 5
        qtukey = 3.031  # alpha = 0.05
        cd = qtukey * np.sqrt(k * (k + 1) / N / 6)
        print("Algorithm-pairs which are different (p = 0.1):")
        for i in range(k):
            for j in range(i+1, k):
                if abs(ave_rank[i] - ave_rank[j]) > cd:
                    print(algorithm[i]+" and "+algorithm[j])
        PlotFriedmanFig(algorithm, ave_rank, cd)
    else:
        print("Algorithms are similar (p = 0.05)")
        k = 8
        N = 5
        qtukey = 3.031  # alpha = 0.05
        cd = qtukey * np.sqrt(k * (k + 1) / N / 6)
        PlotFriedmanFig(algorithm, ave_rank, cd)
