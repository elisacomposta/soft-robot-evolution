import os
from qd.plot import plot_mean_trend, plot_mean_grid
from utils.algo_utils import string_to_list, find_in_metadata

if __name__ == '__main__':
    
    experiments = ['test_qd1', 'test_qd2', 'test_qd3']
    saving_dir = 'multi_exp_test'

    # set saving directory
    saving_path = os.path.join('results', saving_dir)
    try:
        os.makedirs(saving_path)
    except:
        pass

    # find features to set grid labels
    features = string_to_list(find_in_metadata(os.path.join('results', experiments[0]), 'FEATURES'))
    print()

    # plot trends and grids with multiple experiments data
    plot_mean_trend(experiments, 'activityTrend',  saving_path=saving_path, results_dir='results', color='purple', x_label='Evaluations', y_label='Explored bins', y_whole=True)
    print("A plot of the activityTrend was saved at", saving_path)

    plot_mean_trend(experiments, 'fitnessTrend',  saving_path=saving_path, results_dir='results', color='green', x_label='Evaluations', y_label='Fitness')
    print("A plot of the fitnessTrend was saved at", saving_path)

    plot_mean_grid(experiments,'activityGrid',  saving_path=saving_path, results_dir='results', color='Purples', x_label=features[0], y_label=features[1])
    print("A plot of the activityGrid was saved at", saving_path)

    plot_mean_grid(experiments, 'performancesGrid', saving_path=saving_path, results_dir='results', color='YlGn', x_label=features[0], y_label=features[1])
    print("A plot of the performancesGrid was saved at", saving_path)
