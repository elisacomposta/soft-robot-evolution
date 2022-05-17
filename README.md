# Soft Robot evolution: Evolution Gym and Map Elites
Soft robot evolution using:
- [EvolutionGym](https://github.com/EvolutionGym/evogym) benchmark, to co-optimize control and design and run simulations
- Map Elites, provided by [QDPY](https://gitlab.com/leo.cazenille/qdpy), to promote diversity

# Installation

Clone the repository:
```shell
git clone https://github.com/elisacomposta/soft-robot-evolution.git
```
<br>Install qdpy:
```shell
pip3 install qdpy[all]
```

<br>Inside the main folder, clone EvolutionGym repository:
```shell
git clone --recurse-submodules https://github.com/elisacomposta/evogym.git
```
and follow the instructions to install it at [evogym repository](https://github.com/elisacomposta/evogym.git)

# Run example
Parameters can be defined in `run_qd.py`:
- `experiment_name` = name of the experiment; all results are saved in results/experiment_name
- `configFileName` = name of the YAML configuration file
- `num_cores` = number of robots to train in parallel
- `parallelismType` = type of parallelism used for the illumination process. <br>
   _Note_: QDPY proposes many types of executors ('sequential', 'multithreading', 'multiprocessing', 'concurrent', 'scoop', 'ray'). For more details see `qdpy/base.py`.


The environment name and other parameters can be defined in the configuration file. See the examples in `conf/`.

Run the example using the following command.
```shell
python run_qd.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --no-cuda --eval-interval 20
```
All PPO hyperparameters are specified through command line arguments. For more details please see [this repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).
