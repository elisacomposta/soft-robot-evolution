import os
from qd.run import run_qd

if __name__ == '__main__':

    run_qd(
        experiment_name = 'test_qd',
        configFileName = 'conf/test_short.yaml',
        resultsBaseDir = os.path.join('results'),
        parallelismType = 'multithreading',
        num_cores = 12
    )
