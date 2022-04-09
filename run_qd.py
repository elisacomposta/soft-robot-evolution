import os
from qd.run import run_qd

    
if __name__ == '__main__':

    ## SET EXPERIMENT NAME
    experiment_name = 'test_qd'

    ## SET EXPERIMENT ARGS
    args = {}
    args['configFileName'] = 'conf/test.yaml'
    args['resultsBaseDir'] = os.path.join('results', experiment_name)
    args['parallelismType'] = 'multithreading'      

    ## RUN                                       
    run_qd(
        experiment_name = experiment_name,
        args = args,
    )