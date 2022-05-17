import os
from qd.plot import plot_mean_trend, plot_mean_grid, whiten_cmap
from utils.algo_utils import string_to_list, find_in_metadata
import numpy as np

if __name__ == '__main__':
    
    # QD PLOT
    experiments = ['test_1', 'test_2']
    saving_dir = 'test_qd_avg'
    results_dir = 'results'
    tot_random = 0
    features = string_to_list(find_in_metadata(os.path.join('results', experiments[0]), 'FEATURES')) # find features to set grid labels
    
    
    # EVOGYM PLOT
    """
    experiments = ['test_ga']
    saving_dir = os.path.join(experiments[0], 'plots')
    results_dir = os.path.join('evogym', 'examples', 'saved_data')
    features = ['actuation', 'emptiness']
    """
    
    fitnessDomain = [0, 10]

    # set saving directory
    saving_path = os.path.join(results_dir, saving_dir)
    try:
        os.makedirs(saving_path)
    except:
        pass

    save_path_metadata = os.path.join(saving_path, 'plot_metadata.txt')
    f = open(save_path_metadata, "w")
    for exp in experiments:
        f.write(exp + "\n")
    f.close()


    # plot trends and grids with multiple experiments data
    plot_mean_trend(experiments, 'activityTrend',  saving_path=saving_path, results_dir=results_dir, color='purple', x_label='Evaluations', y_label='Explored bins', y_whole=True, tot_random=tot_random)
    print("\nA plot of the activityTrend was saved at", saving_path)

    plot_mean_trend(experiments, 'fitnessTrend',  saving_path=saving_path, results_dir=results_dir, color='green', x_label='Evaluations', y_label='Fitness', tot_random=tot_random)
    print("A plot of the fitnessTrend was saved at", saving_path)

    plot_mean_grid(experiments,'activityGrid',  saving_path=saving_path, results_dir=results_dir, color=whiten_cmap('Blues'), x_label=features[0], y_label=features[1])
    print("A plot of the activityGrid was saved at", saving_path)

    plot_mean_grid(experiments, 'performancesGrid', saving_path=saving_path, results_dir=results_dir, color='YlGn', x_label=features[0], y_label=features[1], fitnessDomain=fitnessDomain)
    print("A plot of the performancesGrid was saved at", saving_path)
