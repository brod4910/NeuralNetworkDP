# python imports
from queue import Queue
import json

# library imports
import torch
import torch.nn as nn
import numpy as np

# local imports
from .nndp import NNDP
from utils.prepare_data import prepare_data

def train(args):
    D = prepare_data(args.file)
    n_verticies = len(D)

    epsilon_t = 1.0

    nndp_model = NNDP(n_verticies)

    for t in range(args.iters):
        # intialize data structures
        state_list = {}
        Q = Queue()
        visited_states = {}
        # initial state (start at vertex 1)
        s0 = {'P': [1 for i in range(n_verticies)], 'c':[1 if j == 0 else 0 for j in range(n_verticies)]}
        Q.put(s0)
        # put s0 into visited states
        vs = stringify_dict(s0)
        visited_states[vs] = True
        while not Q.empty():
            s = Q.get()
            # stringify state
            vs = stringify_dict(s)
            # add state to state list
            state_list[vs] = True
            if np.random.rand() >= epsilon_t:
                a = decision(nndp_model, s, D)
            else:
                a = random_decision(s, n_verticies)

def decision(nndp_model, state, D):
    pass    


def stringify_dict(l):     
    return json.dumps(l)

def random_decision(s, n_verticies):
    pass


