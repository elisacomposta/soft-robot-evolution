import math
from os import name
from evogym import is_connected, has_actuator, get_full_connectivity, draw, get_uniform
import random

class Structure():

    def __init__(self, body, connections, label=-1, generation = -1, shape = [-1, -1]):
        self.body = body
        self.connections = connections
        self.shape = shape

        self.reward = 0
        self.fitness = self.compute_fitness()
        
        self.is_survivor = False
        self.prev_gen_label = 0

        self.label = label
        self.generation = generation

    def compute_fitness(self):

        self.fitness = self.reward
        return self.fitness

    def set_reward(self, reward):

        self.reward = reward
        self.compute_fitness()

    def __str__(self):
        return f'\n\nStructure:\n{self.body}\nF: {self.fitness}\tR: {self.reward}\tID: {self.label}'

    def __repr__(self):
        return self.__str__()


class TerminationCondition():

    def __init__(self, max_iters):
        self.max_iters = max_iters

    def __call__(self, iters):
        return iters >= self.max_iters

    def change_target(self, max_iters):
        self.max_iters = max_iters


def mutate(child, shape, mutation_rate=0.1, num_attempts=10):
    pd = get_uniform(5)  # probability of sampling each element
    pd[0] = 0.6 #it is 3X more likely for a cell to become empty
    # iterate until valid robot found
    for n in range(num_attempts):
        mutated = False
        # for every cell there is mutation_rate% chance of mutation
        for i in range(shape[0]):
            for j in range(shape[1]):
                mutation = [mutation_rate, 1-mutation_rate]
                if draw(mutation) == 0: # mutation
                    mutated = True
                    child[i][j] = draw(pd) 
        if is_connected(child) and has_actuator(child) and mutated:
            #print("Successful mutation")
            return (child, get_full_connectivity(child))

    # Force mutation of one voxel only (mutation failed after num_attempts)
    for n in range(num_attempts):
        i = random.randrange(shape[0])
        j = random.randrange(shape[1])
        child[i][j] = draw(pd)
        if is_connected(child) and has_actuator(child):
            #print("Mutation forced")
            return (child, get_full_connectivity(child))

    #print("Mutation failed")
    return None, None # no valid robot found

class EvaluationMap():
    history = {}

    def init(self):
        self.history = {}

    def get_evaluation(self, ind):
        hash = ind.structure.body.data.tobytes()
        fitness = self.history.get(hash)
        if fitness is None:
            return None
        else:
            ind.set_fitness(fitness)
            return fitness

    def add(self, individuals):
        for ind in individuals:
            hash = ind.structure.body.data.tobytes()
            fitness = ind.structure.fitness
            if self.get_evaluation(ind) is None:
                self.history[hash] = fitness
            else:
                assert(self.history[hash] == fitness)

    def pretty_print(self):
        print("\n", len(self.history), " evaluations stored")
        count = 1
        for i in self.history:
            print("eval ", count, ": ", self.history[i])
            count += 1
        print()