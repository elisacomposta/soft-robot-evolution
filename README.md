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

<br>Inside the main folder, clone Evolution Gym repository:
```shell
git clone --recurse-submodules https://github.com/EvolutionGym/evogym.git
```
and follow the instructions to install it at [evogym repository](https://github.com/EvolutionGym/evogym)

# Run example
```shell
python run_qd.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 4 --num-steps 128 --num-mini-batch 4 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --eval-interval 20
```
