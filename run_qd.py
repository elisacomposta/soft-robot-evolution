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


"""
python run_qd.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 20
"""