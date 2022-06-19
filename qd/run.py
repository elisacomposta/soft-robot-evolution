import os
import shutil
from unittest import result
import warnings
import traceback

from torch import from_dlpack

from qdpy.experiment import QDExperiment
from qd.sim import compute_features, make_env, simulate, evaluate_ind

import time

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')

class EvoGymExperiment(QDExperiment):
    def reinit(self):
        super().reinit()
        self.env_name = self.config['env_name']
        self.shape = self.config['algorithms']['shape']
        self.population_structure_hashes = {}

    def eval_fn(self, individuals):
        
        for ind in individuals:
            compute_features(ind, self.features_list)
            make_env(env_name = self.env_name, ind=ind)

        if self.reoptimize != '' and not self.reoptimize:
            evaluate_ind(self.env_name, individuals, self.structure_from, from_labels=self.from_labels, num_cores=self.num_cores)
        else:
            simulate(self.env_name, individuals, self.experiment_name, self.config[('indv_eps')], num_cores=self.num_cores)  #compute fitness
        
        ## STORE RESULTS
        store_results(path=self.save_path, individuals=individuals)
        return individuals


##### BASE FUNCTIONS ######
def create_base_config(resultDir):
    base_config = {}
    if len(resultDir) > 0:
        base_config['resultsBaseDir'] = resultDir
    return base_config

def create_experiment(experiment_name, configFileName, parallelismType, base_config, num_cores, save_path='results'):
    exp = EvoGymExperiment(configFileName, parallelismType, seed=None, base_config=base_config) #seed defined in conf file
    exp.experiment_name = experiment_name
    exp.num_cores = num_cores
    exp.save_path = save_path
    print("Using configuration file '%s'. Instance name: '%s'" % (configFileName, exp.instance_name))
    return exp

def launch_experiment(exp):
    print()
    exp.run()      # run illumination process

##### STORE DATA ####
def store_metadata(exp, save_path):
    save_path_metadata = os.path.join(save_path, 'metadata.txt')
    f = open(save_path_metadata, "w")
    f.write(f'ENVIRONMENT: {exp.env_name}\n')
    f.write(f'CONFIGURATION FILE: {exp.config_filename}\n')
    f.write(f'INSTANCE: {exp.instance_name}\n')
    if exp.structure_from == '':
        f.write(f'SEED: {exp.config["seed"]}\n')
        f.write(f'SHAPE: {exp.shape}\n')
    else:
        f.write(f'STRUCTURES FROM: {exp.structure_from}\n')
    if exp.reoptimize != False:
        f.write(f'INDV EPS: {exp.config["indv_eps"]}\n')
    else:
        f.write('REOPTIMIZE CONTROLLER: False\n')
    f.write(f'FITNESS: {exp.fitness_type}\n')
    f.write(f'FEATURES: {exp.features_list}\n')
    f.write("\n")
    
    algos = exp.config['algorithms']['algoTotal']['algorithms']
    for x in range(len(algos)):
        tot = exp.config['algorithms'][algos[x]]['budget']
        f.write(f'generation_{x}\ttot_ind:{tot}\t{algos[x]}\n')

    f.close()

def store_results(path, individuals):
    test_path = os.path.join(path, "results.csv")
    f_csv = open(test_path, "a")
    for ind in individuals:
        f_csv.write(f"ind{ind.structure.label};generation{ind.structure.generation};{ind.features};{ind.fitness[0]}\n")
    f_csv.close()


###### RUN EXPERIMENT ######
def run_qd(experiment_name, configFileName, parallelismType, num_cores=4):
    print()

    ## MANAGE DIRECTORIES
    save_path = os.path.join(root_dir, 'results', experiment_name)
    try:
        os.makedirs(save_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print('Override? (y/n): ', end='')
        ans = input()
        if ans.lower() == 'y':
            shutil.rmtree(save_path)
            os.makedirs(save_path)
        else:
            return None, None
        print()


    ## LAUNCH EXPERIMENT
    base_config = create_base_config(save_path)
    try:
        exp = create_experiment(experiment_name, configFileName, parallelismType, base_config, num_cores, save_path)
        store_metadata(exp, save_path)

        start_time = time.time()    # tmp: start timer

        launch_experiment(exp)

        print("\n-------", time.time()-start_time, "-----\n")   # tmp: keep track of total execution time

    except Exception as e:
        warnings.warn(f"Run failed: {str(e)}")
        traceback.print_exc()
