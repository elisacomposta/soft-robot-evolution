#    This file is part of qdpy.
#
#    qdpy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    qdpy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with qdpy. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`qdpy.experiment` module contains classes providing a standard way of performing a QD Experiment """

__all__ = ["QDExperiment"]

from qdpy.algorithms import *
from qdpy.containers import *
from qdpy.plots import *
from qdpy.base import *
from qdpy import tools
from qd.plot import *

import yaml
import random
import datetime
import pathlib
import shutil

class QDExperiment(object):
    def __init__(self, config_filename, parallelism_type = "sequential", seed = None, base_config = None):
        self._loadConfig(config_filename)
        if base_config is not None:
            self.config = {**self.config, **base_config}
        self.parallelism_type = parallelism_type
        self.config['parallelism_type'] = parallelism_type
        self._init_seed(seed)
        self.reinit()

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['algo']
        del odict['container']
        return odict


    def _loadConfig(self, config_filename):
        self.config_filename = config_filename
        self.config_name = os.path.splitext(os.path.basename(config_filename))[0]
        self.config = yaml.safe_load(open(config_filename))

    def _get_features_list(self):
        features_list = self.config['features_list']
        fitness_type = self.config['fitness_type']
        return features_list, fitness_type

    def _define_domains(self):
        self.features_list, self.fitness_type = self._get_features_list()
        self.config['features_domain'] = []
        for feature_name in self.features_list:
            val = self.config['%s%s' % (feature_name, "Domain")]
            self.config['features_domain'] += [tuple(val)]
        self.config['fitness_domain'] = tuple(self.config['%s%s' % (self.fitness_type, "Domain")]),

    def _init_seed(self, rnd_seed = None):
        # Find random seed
        if rnd_seed is not None:
            seed = rnd_seed
        elif "seed" in self.config:
            seed = self.config["seed"]
        else:
            seed = np.random.randint(1000000)

        # Update and print seed
        np.random.seed(seed)
        random.seed(seed)
        print("Seed: %i" % seed)


    def reinit(self):
        # Name of the expe instance based on the current timestamp
        self.instance_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        # Identify and create result data dir
        if not self.config.get('dataDir'):
            resultsBaseDir = self.config.get('resultsBaseDir') or "./results/"
            dataDir = os.path.join(os.path.expanduser(resultsBaseDir), os.path.splitext(os.path.basename(self.config_filename))[0])
            self.config['dataDir'] = dataDir
        pathlib.Path(self.config['dataDir']).mkdir(parents=True, exist_ok=True)

        # Find the domains of the fitness and features
        self._define_domains()
        default_config = {}
        default_config["fitness_domain"] = self.config['fitness_domain']
        default_config["features_domain"] = self.config['features_domain']

        # Create containers and algorithms from configuration
        factory = Factory()         #qdpy/base.py
        assert "containers" in self.config, f"Please specify configuration entry 'containers' containing the description of all containers."
        factory.build(self.config["containers"], default_config)
        assert "algorithms" in self.config, f"Please specify configuration entry 'algorithms' containing the description of all algorithms."
        factory.build(self.config["algorithms"])
        assert "main_algorithm_name" in self.config, f"Please specify configuration entry 'main_algorithm' containing the name of the main algorithm."
        self.algo = factory[self.config["main_algorithm_name"]]
        self.container = self.algo.container

        self.batch_mode = self.config.get('batch_mode', False)
        self.log_base_path = self.config['dataDir']

    def run(self):
        # Run illumination process !
        with ParallelismManager(self.parallelism_type) as pMgr:
            try:
                history = self.population_structure_hashes
            except:
                history = None
            best_after_eval, activity_after_eval = self.algo.optimise(self.eval_fn, executor = pMgr.executor, batch_mode=self.batch_mode, pop_structure_hashes=history) # Disable batch_mode (steady-state mode) to ask/tell new individuals without waiting the completion of each batch

        # Save results
        if isinstance(self.container, Grid):
            grid = self.container
        else:
            # Transform the container into a grid
            print("\n{:70s}".format("Transforming the container into a grid, for visualisation..."), end="", flush=True)
            grid = Grid(self.container, shape=(10,10), max_items_per_bin=1, fitness_domain=self.container.fitness_domain, features_domain=self.container.features_domain, storage_type=list)
            print("\tDone !")

        algo_budget = {}
        if isinstance(self.algo, Sq):
            for i in range(len(self.algo.algorithms)):
                algo_budget[str(self.algo.algorithms[i])] = self.algo.algorithms[i].budget

        # Create plot of the fitness trend
        plot_path = os.path.join(self.log_base_path)
        file_name =  f"fitnessTrend-{self.instance_name}"
        plotTrend(  x=best_after_eval.keys(), y=best_after_eval.values(), 
                    path=plot_path, fileName=file_name, 
                    xlabel="Evaluations", ylabel="Fitness",
                    color='green', 
                    tot_random = algo_budget['Random'],
                    showRandLimit=True, showRandCol=True)
        print("\nA plot of the fitness trend was saved in '%s'." % os.path.abspath(plot_path))

        # Create plot of the activity trend
        plot_path = os.path.join(self.log_base_path)
        file_name = f"activityTrend-{self.instance_name}"
        plotTrend(  x=activity_after_eval.keys(), y=activity_after_eval.values(), 
                    path=plot_path, fileName=file_name, 
                    color='royalblue', y_whole = True, 
                    xlabel="Evaluations", ylabel="Explored bins",
                    tot_random = algo_budget['Random'],
                    showRandLimit=True, showRandCol=True)
        print("A plot of the activity trend was saved in '%s'." % os.path.abspath(plot_path))

        # Create plot of the performance grid
        plot_path = os.path.join(self.log_base_path, f"performancesGrid-{self.instance_name}.pdf")
        quality = grid.quality_array[(slice(None),) * (len(grid.quality_array.shape) - 1) + (0,)]        
        plotGridSubplots(quality, plot_path, plt.get_cmap("YlGn"),
                        grid.features_domain, grid.fitness_domain[0], 
                        xlabel=self.features_list[0], ylabel=self.features_list[1])
        print("A plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))

        # Create plot of the activity grid
        plot_path = os.path.join(self.log_base_path, f"activityGrid-{self.instance_name}.pdf")
        plotGridSubplots(grid.activity_per_bin, plot_path, whiten_cmap('Blues'), 
                        grid.features_domain, [0, np.max(grid.activity_per_bin)], 
                        xlabel=self.features_list[0], ylabel=self.features_list[1])
        print("A plot of the activity grid was saved in '%s'." % os.path.abspath(plot_path))


    def _removeTmpFiles(self, fileList):
        keepTemporaryFiles = self.config.get('keepTemporaryFiles')
        if not keepTemporaryFiles:
            for f in fileList:
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                    else:
                        shutil.rmtree(f)
                except:
                    pass

    def eval_fn(self, ind):
        """ Default evaluate function: random values. Can override when inheriting """
        print("Using default evaluate function... Random values generated")
        fitness = [np.random.uniform(x[0], x[1]) for x in self.config['fitness_domain']]
        features = [np.random.uniform(x[0], x[1]) for x in self.config['features_domain']]
        ind.fitness.values = fitness
        ind.features = features
        return ind
