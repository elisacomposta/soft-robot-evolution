from cProfile import label
import math
import os
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

def pretty_print(list_org, max_name_length=30):

    list_formatted = []
    for i in range(len(list_org)//4 +1):
        list_formatted.append([])

    for i in range(len(list_org)):
        row = i%(len(list_org)//4 +1)
        list_formatted[row].append(list_org[i])

    print()
    for row in list_formatted:
        out = ""
        for el in row:
            out += str(el) + " "*(max_name_length - len(str(el)))
        print(out)

class EvaluationMap():
    history = {}

    def init(self):
        self.history = {}

    def get_evaluation(self, ind):
        hash = ind.structure.body.data.tobytes()
        values = self.history.get(hash) # (label, fitness)
        if values is None:
            return None
        else:
            ind.set_fitness(values[1])
            return values

    def add(self, individuals):
        for ind in individuals:
            hash = ind.structure.body.data.tobytes()
            fitness = ind.structure.fitness
            label = ind.structure.label     # save first ind label to copy structure and controller in dir
            if self.get_evaluation(ind) is None:
                self.history[hash] = (label, fitness)   
            else:
                assert(self.history[hash][2] == fitness)

    def pretty_print(self):
        print("\n", len(self.history), " evaluations stored")
        count = 1
        for i in self.history:
            print("eval ", count, " fitness:" , self.history[i][1])
            count += 1
        print()


def get_ind_path(label, base_path='results'):
    name = "ind" + str(label)
    for (root,dirs,files) in os.walk(base_path, topdown=True):
        if name in dirs:
            return os.path.join(root, name)

def generate_ind_path(base_path, generation, label):
    return os.path.join(base_path, "generation_" + str(generation), "ind" + str(label))

