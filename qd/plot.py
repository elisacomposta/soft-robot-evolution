import matplotlib.pyplot as plt
import numpy as np

def plotTrend(data, fileName, xlabel = "", ylabel = "", nb_xticks = 14, nb_yticks = 7, y_whole = False, tot_random = 0, showRandLimit = False, showRandCol = False):
    """
        data:           dictionary of data to plot. keys = x, values = y
        fileName:       file name to save plot ( e.g: file.pdf )
        nb_ticks:       number of ticks on x axis
        ny_ticks:       number of ticks on y axis
        tot_random:     total number of first generation random-generated individuals
        showRandLimit:  if True show a vertical dashed line to graphically separate random generation
        showRandCol:    if True use a different color for the random generation  
    """

    # set data
    x = list(data.keys())
    y = list(data.values())
    x.insert(0, 0)
    y.insert(0, 0)
    tot_evaluations = x[-1]

    # generate ticks
    stepx = np.round(tot_evaluations / nb_xticks, 0)
    xticks = np.arange(x[0], tot_evaluations + stepx, stepx)
    
    stepy = ( y[-1] - y[0] ) / nb_yticks
    if y_whole:
        stepy = np.round(stepy, 0)
    if stepy != 0:
        yticks = np.arange(y[0], y[-1] + stepy, stepy).round(2)
    else:
        yticks = y
    
    # set ticks, labels, grid
    fig, ax = plt.subplots()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)
    ax.grid(which='major', color=(0.8,0.8,0.8,0.5), linestyle='-', linewidth=0.1)
    
    # plot
    ax.plot(x, y)

    # plot random section if required
    if (showRandCol or showRandLimit) and tot_random is not None and tot_random > 0:
        x0 = [x[i] for i in range(len(x)) if x[i] <= tot_random]
        y0 = [y[i] for i in range(len(y)) if x[i] <= tot_random]
        if showRandCol:
            plt.plot(x0, y0, color = 'green')
        if showRandLimit:
            plt.vlines(x=[tot_random], ymin=0, ymax=y[-1], colors='orange', ls='--', lw=1)
        
    # save plot
    plt.savefig(fileName)
