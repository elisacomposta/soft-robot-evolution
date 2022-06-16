import os
import numpy as np
import pandas as pd
import math
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

def best_in_exp(exp, n, bounds=[-50, 50]):
    """
    Args:
        exp:    path of the experiment to get the best individuals from
        n:      number of top individuals to find (Note: max is the number of cells in the map)
        bounds: fitness domain mask [optional]
    Returns:
        the labels of the best n individuals in exp
    """
    
    df = pd.read_csv(os.path.join(exp, 'results.csv'), delimiter=';', header=None)  # read results csv
    df2 = df.sort_values(by=df.columns[3], ascending = False)   # sort by fitness value

    top = {}    # keys: unique floored features, values: ind number

    row_index = 0
    while len(top) < n and row_index < len(df2.index):
        row = df2.iloc[row_index]
        features = ( float(row[2].split(',')[0][1:4]), float(row[2].split(',')[1][1:4]) )

        if features[0] == 1.0:
            features = (0.9, features[1])
        if features[1] == 1.0:
            features = (features[0], 0.9)
            
        if features not in top.keys() and row[3]>bounds[0]:
            top[features] = row[0][3:]
        row_index += 1

    top = list(top.values())

    print("Found top", len(top), "individuals at", exp)
    return top