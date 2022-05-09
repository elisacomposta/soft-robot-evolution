import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plotTrend(x, y, path, fileName, error=[], xlabel = "", ylabel = "", nb_xticks = 10, nb_yticks = 7, y_whole = False, color = 'blue', tot_random = 0, showRandLimit = False, showRandCol = False):
    """
    Plot trend.

    Args:
        data:           dictionary of data to plot. keys = x, values = y NO
        x:
        y:
        fileName:       file name to save plot ( e.g: "path/file" )
        xlabel:         label on x axis
        ylabel:         label on y axis
        nb_xticks:      number of ticks on x axis
        nb_yticks:      number of ticks on y axis
        y_whole:        if y_whole, y ticks will be whole numbers
        color:          color of the main line
        tot_random:     total number of first generation random-generated individuals
        showRandLimit:  if True show a vertical dashed line to graphically separate random generation
        showRandCol:    if True use a different color for the random generation  
    """

    # set data
    x = np.array(list(x))
    y = np.array(list(y))
    
    if x[0] != 0:   # (0, 0) missing
        x =np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)
    tot_evaluations = x[-1]

    # generate ticks
    stepx = np.round(tot_evaluations / nb_xticks, 0)
    xticks = np.arange(x[0], tot_evaluations + stepx, stepx)
    

    if len(error) > 0:
        max_y = y[-1] + error[-1]
    else:
        max_y = y[-1]

    stepy = ( max_y - y[0] ) / nb_yticks
    if y_whole:
        stepy = np.round(stepy, 0)
    if stepy != 0:
        yticks = np.arange(y[0], max_y + stepy, stepy).round(2)
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
    ax.plot(x, y, color=color)

    # plot random section if required
    if (showRandCol or showRandLimit) and tot_random is not None and tot_random > 0:
        x0 = [x[i] for i in range(len(x)) if x[i] <= tot_random]
        y0 = [y[i] for i in range(len(y)) if x[i] <= tot_random]
        if showRandCol:
            ax.plot(x0, y0, color = 'tomato')
        if showRandLimit:
            ax.vlines(x=[tot_random], ymin=0, ymax=y[-1], colors='orange', ls='--', lw=1)
        
    # std dev
    if len(error)>0:
        if len(error) > len(y):     # (0, 0) missing
            error = list(error)
            error.insert(0, 0)
            error = np.array(error)
        ax.fill_between(x, y-error, y+error,  color=color, alpha=0.15)

    # save plot
    plt.savefig(os.path.join(path, fileName+'.pdf'))

    # store coordinates for future plots
    store_coordinates((x, y), path, fileName)


def store_coordinates(data, path, fileName):
    try:
        os.makedirs(os.path.join(path, 'coordinates'))
    except:
        pass
    np.save(os.path.join(path, 'coordinates', fileName.split('-')[0]), data)



if __name__ == '__main__':

    experiments = ['test_qd', 'test_qd2']
    dir_name = 'multi_exp_test'
    results_dir = os.path.join(os.path.dirname(os.getcwd()), 'results')

    plot_names = ['activityTrend.npy', 'fitnessTrend.npy']
    colors = ['purple', 'green']

    # set saving directory
    saving_path = os.path.join(os.path.dirname(os.getcwd()), 'results', dir_name)
    try:
        os.makedirs(saving_path)
    except:
        pass

    print()
    # precompute plot infos
    for i in range(len(plot_names)):

        # get experiment name to plot
        name = plot_names[i]

        # generate matrix with coordinate values for each experiment
        x_dir = np.zeros((len(experiments), 1))
        y_dir = np.zeros((len(experiments), 1))

        for j in range(len(experiments)):   

            # find file with stored coordinates
            for (root,dirs,files) in  os.walk(os.path.join(results_dir, experiments[i]), topdown=True):
                if name in files:
                    coord_path = str(os.path.join(root, name))

            # load coordinates
            x, y = np.load(coord_path, 'r')

            # resize coordinate matrix if necessary (only the first time)
            if j==0:        
                x_dir.resize((len(experiments), len(x)))
                y_dir.resize((len(experiments), len(x)))
                y = [t+2 if t != 0 else 0 for t in x ]
        
            # populate coordinate matrix
            x_dir[:][j] = x
            y_dir[:][j] = y

        # compute mean of coordinates
        x_mean = x_dir.mean(axis=0)
        y_mean =  y_dir.mean(axis=0)

        # compute standard deviation on y axis
        error = y_dir.std(axis=0) 
        
        # plot
        plotTrend(x_mean, y_mean, saving_path, name, error=error, color=colors[i])
        print(name, "plot at", saving_path)
    