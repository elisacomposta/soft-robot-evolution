import matplotlib.pyplot as plt
import numpy as np

def plotTrend(best_fitness, fileName, nb_xticks = 14, nb_yticks = 7, tot_random = 0, showRandLimit = False, showRandCol = False):
    x = list(best_fitness.keys())
    y = list(best_fitness.values())
    x.insert(0, 0)
    y.insert(0, 0)

    tot_evaluations = x[-1]

    stepx = np.round(tot_evaluations / nb_xticks, 0)
    xticks = np.arange(x[0], tot_evaluations + stepx, stepx)
    
    stepy = ( y[-1] - y[0] ) / nb_yticks
    if stepy != 0:
        yticks = np.arange(y[0], y[-1] + stepy, stepy).round(2)
    else:
        yticks = y
    
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlabel("total evaluations", fontsize = 12)
    plt.ylabel("fitness", fontsize = 12)
    plt.grid(which='major', color=(0.8,0.8,0.8,0.5), linestyle='-', linewidth=0.1)
    
    plt.plot(x, y, color = 'blue')

    if tot_random is not None and tot_random > 0:
        x0 = []
        y0 = []
        for i in range(len(x)):
            if x[i] <= tot_random:
                x0.append(x[i])
                y0.append(y[i])
        if showRandCol:
            plt.plot(x0, y0, color = 'green')
        if showRandLimit:
            plt.vlines(x=[tot_random], ymin=0, ymax=y[-1], colors='orange', ls='--', lw=1)

    plt.savefig(fileName)
