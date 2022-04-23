import matplotlib.pyplot as plt
import numpy as np

def plotTrend(best_fitness, fileName, fitnessDomain = None):
    y = best_fitness
    x = [i for i in range(len(y))]

    plt.xlabel("generation", fontsize = 16)
    plt.ylabel("fitness", fontsize = 16)

    if len(x) < 10:
        xticks = x
    else:
        nb_xticks = 12
        stepx = np.round(x[-1] / nb_xticks, 0)
        xticks = np.arange(0, stepx * nb_xticks + 2*stepx, stepx)

    if fitnessDomain is None:
        yticks = np.arange(fitnessDomain[0], fitnessDomain[1]).round(2)
    else:
        stepy = ( y[-1] - y[0] ) / 7
        yticks = np.arange(y[0], y[-1] + stepy, stepy).round(2)
    
    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.grid(which='major', color=(0.8,0.8,0.8,0.5), linestyle='-', linewidth=0.1)

    plt.plot(x, y)
    plt.savefig(fileName)
