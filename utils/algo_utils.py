import os
import numpy as np
from evogym import is_connected, has_actuator, get_full_connectivity, draw, get_uniform

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


def mutate(child, shape, mutation_rate=0.1, num_attempts=50):
    pd = get_uniform(5)  # probability of sampling each element
    pd[0] = 0.6 #it is 3X more likely for a cell to become empty
    
    for n in range(num_attempts): # iterate until valid robot found
        # for every cell there is mutation_rate% chance of mutation
        for i in range(shape[0]):
            for j in range(shape[1]):
                mutation = [mutation_rate, 1-mutation_rate]
                if draw(mutation) == 0: # mutation
                    child[i][j] = draw(pd) 
        if is_connected(child) and has_actuator(child):
            return (child, get_full_connectivity(child))

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


def get_ind_path(label, base_path='results'):
    """
    Returns the path of individual "ind<label>".
    """
    name = "ind" + str(label)
    for (root,dirs,files) in os.walk(base_path, topdown=True):
        if name in dirs:
            return os.path.join(root, name)


def find_in_metadata(path, field):
    """
    Gets the value stored in the path/metadata.txt file of an experiment, for a corrisponding field
    """

    path = os.path.join(path, 'metadata.txt')
    f = open(path, "r")
    line = f.readline().rstrip().split(": ")
    while line[0]!= field:
        line = f.readline().rstrip().split(": ")
    f.close()
    return line[1]


def string_to_list(word):
    """
    Converts a string of csv into a list.
    e.g.: ['val1', 'val2'], type: string -> type: list, items: val1, val2
    """

    splitted = word.split(",")
    clean_list = []
    for item in splitted:
        clean_list.append(item.translate({ ord(c): None for c in "[]' " }))
    return clean_list

def get_stored_structure(structure_path):
    """
    Returns a structure previously stored in
    """
    structure_data = np.load(structure_path)
    structure = []
    for key, value in structure_data.items():
        structure.append(value)
    structure = tuple(structure)
    return structure
