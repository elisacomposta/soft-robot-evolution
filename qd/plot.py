import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

def plotTrend(x, y, path, fileName, error=[], xlabel = "", ylabel = "", nb_xticks = 10, nb_yticks = 7, y_whole = False, color = 'blue', tot_random = 0, showRandCol = False, ax=None, line_label=''):
    """
    Plot trend.

    Args:
        x, y:           coordinates of points to plot
        path:           path where the plot will be saved 
        fileName:       file name to save plot ( no extension )
        error:          structure to plot std dev
        xlabel:         label on x axis
        ylabel:         label on y axis
        nb_xticks:      number of ticks on x axis
        nb_yticks:      number of ticks on y axis
        y_whole:        if y_whole, y ticks will be whole numbers
        color:          color of the main line
        tot_random:     total number of first generation random-generated individuals
        showRandCol:    if True use a different color for the random generation  
        ax:             axes to plot on [optional]
        line_label:     label for the legend
    """

    
    # set data
    x = np.array(list(x))
    y = np.array(list(y))
    
    # define 0-error array if not provided
    if len(error)==0:
        error = np.zeros(len(y))

    if x[0] != 0:       # push front (0, 0) if missing
        x =np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)
        error = np.insert(error, 0, 0)
    tot_evaluations = x[-1]

    # generate x ticks
    stepx = np.round(tot_evaluations / nb_xticks, 0)
    xticks = np.arange(x[0], tot_evaluations + stepx, stepx)
    
    # generate y ticks
    max_y = y[-1] + error[-1]
    stepy = ( max_y - y[0] ) / nb_yticks

    if y_whole:
        stepy = np.round(stepy, 0)

    if stepy != 0:
        yticks = np.arange(y[0], max_y + stepy, stepy).round(2)
    else:
        yticks = y
    
    if ax == None:
        fig, ax = plt.subplots()

    # set ticks, labels, grid
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(xlabel, fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)
    ax.grid(which='major', color=(0.8,0.8,0.8,0.5), linestyle='-', linewidth=0.1)
    
    # plot
    ax.plot(x, y, color=color, label=line_label)
    if line_label != '':
        ax.legend(loc="lower right")

    # plot standard deviation
    ax.fill_between(x, y-error, y+error,  color=color, alpha=0.15)

    # plot random section if required
    if tot_random > 0 or showRandCol:
        x0 = [x[i] for i in range(len(x)) if x[i] <= tot_random]
        y0 = [y[i] for i in range(len(y)) if x[i] <= tot_random]
        if showRandCol:
            ax.plot(x0, y0, color = 'tomato')
        ax.vlines(x=[tot_random], ymin=0, ymax=y[-1], colors='orange', ls='--', lw=1)

    # save plot
    plt.savefig(os.path.join(path, fileName+'.pdf'))

    try:
        # store coordinates for future plots
        store_plot_data((x, y, error), path, fileName)
    except:
        print("Skipped plot data storing")


def plot_mean_trend(experiments, name, saving_path, results_dir, color, x_label='', y_label='', y_whole=False, tot_random=0):
    """
    Plot mean trend.
    
    Args:
        experiments:    list of experiments with data to compute and plot
        name:           name of the file with data to look for ( also resulting plot file name )
        saving_path:    where to save the new plot
        results_dir:    main dir where data to compute are stored
        color:          color of the trend line
        x_label:        label on x axis
        y_label:        label on y axis
        y_whole:        if y_whole, y ticks will be whole numbers
        tot_random:     total number of first generation random-generated individuals
    """

    for j in range(len(experiments)):   

        # find file with stored coordinates
        for (root,dirs,files) in  os.walk(os.path.join(results_dir, experiments[j]), topdown=True):
            if name+'.npy' in files:
                coord_path = str(os.path.join(root, name+'.npy'))

        # load coordinates
        x, y = np.load(coord_path, 'r')

        # define 2d array to store the coordinates for each experiment (first iteration only)
        if j==0:        
            x_dir = np.zeros((len(experiments), len(x)))
            y_dir = np.zeros((len(experiments), len(x)))
    
        # populate coordinates structures
        x_dir[:][j] = x
        y_dir[:][j] = y

    # compute mean of coordinates
    x_mean = x_dir.mean(axis=0)
    y_mean =  y_dir.mean(axis=0)

    # compute standard deviation on y axis
    error = y_dir.std(axis=0) 
    
    # plot trend
    plotTrend(x_mean, y_mean, saving_path, name, error=error, color=color, xlabel=x_label, ylabel=y_label, y_whole=y_whole, tot_random=tot_random)



def plot_mean_grid(experiments, name, color, saving_path, results_dir, x_label='', y_label='', fitnessDomain=None):
    """
    Plot mean grid.
    
    Args:
        experiments:    list of experiments with data to compute and plot
        name:           name of the file with data to look for ( also resulting plot file name )
        color:          color of the trend line
        saving_path:    where to save the new plot
        results_dir:    main dir where data to compute are stored
        x_label:        label on x axis
        y_label:        label on y axis
        fitnessDomain:  range for the color bar
    """

    # generate structure to store grid with averaged values
    mean_grid = []

    for j in range(len(experiments)):  
        
        # find file with stored coordinates
        for (root,dirs,files) in  os.walk(os.path.join(results_dir, experiments[j]), topdown=True):
            if name+'.npy' in files:
                grid_path = str(os.path.join(root, name+'.npy'))

        # load grid
        data = np.load(grid_path, 'r')

        # define structure to store coordinates of each experiment
        if j==0:
            grid_aggregation = [[] for _ in range(len(data))]

        # populate grid aggregation structure with new experiment data
        for k in range(len(data)):
            grid_aggregation[k].append((data[k]))

    # compute mean per raw and populate mean grid
    for raw_aggregation in grid_aggregation:
        raw_aggregation = np.array(raw_aggregation)
        mean_grid.append(np.nanmean(raw_aggregation, axis=0))
    
    mean_grid = np.array(mean_grid)
    path = os.path.join(saving_path, name+".pdf")
    
    # plot grid
    from qdpy.plots import plotGridSubplots
    if fitnessDomain == None:
        fitnessDomain = [0, np.max(mean_grid)]
    plotGridSubplots(mean_grid, path, plt.get_cmap(color),
                    featuresBounds=[(0.0, 1.0), (0.0, 1.0)],
                    fitnessBounds=fitnessDomain, 
                    xlabel=x_label, ylabel=y_label)


def compare_trends(experiments, label, name, saving_path, results_dir, x_label='', y_label='', y_whole=False, tot_random=0):
    """
    Plot mean trend.
    
    Args:
        experiments:    list of experiments to plot
        name:           name of the file with data to look for ( also resulting plot file name )
        saving_path:    where to save the new plot
        results_dir:    array of dir where data to compute are stored
        x_label:        label on x axis
        y_label:        label on y axis
        y_whole:        if y_whole, y ticks will be whole numbers
        tot_random:     total number of first generation random-generated individuals
    """

    color = ['royalblue', 'crimson', 'green']
    fig, ax = plt.subplots()

    for i in range(len(experiments)):

        # find file with stored coordinates
        for (root,dirs,files) in  os.walk(os.path.join(results_dir[i], experiments[i]), topdown=True):
            if name+'.npy' in files:
                coord_path = str(os.path.join(root, name+'.npy'))

        # load coordinates
        x, y, error = np.load(coord_path, 'r')
        
        # plot trend
        plotTrend(x, y, saving_path, name, error=error, color=color[i], xlabel=x_label, ylabel=y_label, y_whole=y_whole, tot_random=tot_random, ax=ax, line_label=label[i])


def store_plot_data(data, path, file_name):
    """
    Store data to .npy file

    Args:
        data:       data to store
        path:       the file will be stored at <path>/plot_data
        file_name:  name of the file to store ( no extension )
    """
    
    try:
        os.makedirs(os.path.join(path, 'plot_data'))
    except:
        pass

    np.save(os.path.join(path, 'plot_data', file_name.split('-')[0]), data)


def whiten_cmap(cmap_name):
    """
    Returns a cmap with 0 value corresponding to white
    """
    newcolors = cm.get_cmap(cmap_name, 256)(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([0/256, 0/256, 0/256, 0])
    return ListedColormap(newcolors)
