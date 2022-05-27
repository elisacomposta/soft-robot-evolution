import os
from qd.plot import plot_mean_trend, plot_mean_grid, compare_trends, whiten_cmap
from utils.algo_utils import string_to_list, find_in_metadata
import numpy as np


if __name__ == '__main__':

    # select plot type
    qd_plot = False
    evogym_plot = False
    compare_exp = True
    
    if qd_plot:
        # QD PLOT
        experiments = ['test_qd_1', 'test_qd_2']
        saving_dir = 'test_qd'
        results_dir = 'results'
        tot_random = 500
        features = string_to_list(find_in_metadata(os.path.join('results', experiments[0]), 'FEATURES')) # find features to set grid labels
        fitnessDomain = [0, 10]
    
    if evogym_plot:
        # EVOGYM PLOT
        experiments = ['test_ga_1', 'test_ga_2']
        #saving_dir = os.path.join(experiments[0], 'plots')      # plot one exp only
        saving_dir = 'test_ga'
        results_dir = os.path.join('evogym', 'examples', 'saved_data')
        features = ['actuation', 'emptiness']
        fitnessDomain = [0, 10]
        tot_random = 0

    if compare_exp:
        # EXP COMPARISON TREND
        experiments = ['test_qd', 'test_ga']               
        line_label = ['MAP-elites', 'GA']     # line labels in legend
        saving_dir = '_qd_vs_ga_'
        qd_dir = 'results'
        gen_algo_dir = os.path.join('evogym', 'examples', 'saved_data')  # with evogym experiments

        results_dir = qd_dir
        results_from = [qd_dir, gen_algo_dir]
        tot_random = 0
    

    # set saving directory
    saving_path = os.path.join(results_dir, saving_dir)
    try:
        os.makedirs(saving_path)
    except:
        pass

    # store matadata (name of experiments used)
    save_path_metadata = os.path.join(saving_path, 'plot_metadata.txt')
    f = open(save_path_metadata, "w")
    for exp in experiments:
        f.write(exp + "\n")
    f.close()


    if compare_exp:
        # plot trend comparison
        compare_trends(experiments, line_label, 'activityTrend',  saving_path=saving_path, results_dir=results_from, x_label='Evaluations', y_label='Explored bins', y_whole=True, tot_random=tot_random)    
        print("\nA plot of activityTrend comparing", experiments, "was saved at", saving_path)

        compare_trends(experiments, line_label, 'fitnessTrend',  saving_path=saving_path, results_dir=results_from, x_label='Evaluations', y_label='Fitness', tot_random=tot_random)    
        print("A plot of fitnessTrend comparing", experiments, "was saved at", saving_path)

    else:
        # plot average trends and grids with multiple experiments data
        plot_mean_trend(experiments, 'activityTrend',  saving_path=saving_path, results_dir=results_dir, color='purple', x_label='Evaluations', y_label='Explored bins', y_whole=True, tot_random=tot_random)
        print("\nA plot of the activityTrend was saved at", saving_path)

        plot_mean_trend(experiments, 'fitnessTrend',  saving_path=saving_path, results_dir=results_dir, color='green', x_label='Evaluations', y_label='Fitness', tot_random=tot_random)
        print("A plot of the fitnessTrend was saved at", saving_path)
    
        plot_mean_grid(experiments,'activityGrid',  saving_path=saving_path, results_dir=results_dir, color=whiten_cmap('Blues'), x_label=features[0].capitalize(), y_label=features[1].capitalize())
        print("A plot of the activityGrid was saved at", saving_path)

        plot_mean_grid(experiments, 'performancesGrid', saving_path=saving_path, results_dir=results_dir, color=whiten_cmap('YlGn'), x_label=features[0].capitalize(), y_label=features[1].capitalize(), fitnessDomain=fitnessDomain)
        print("A plot of the performancesGrid was saved at", saving_path)
        