# python imports
from queue import Queue
import json
from copy import deepcopy

# library imports
import torch
import torch.nn as nn
import numpy as np

def train(args, nndp_model, graph, n_verticies, device, optimizer, criterion):
    epsilon_t = .995

    for t in range(1):
        for start_vertex in range(n_verticies):
            # intialize data structures
            visited_states = {}
            # initial state (start at vertex 1)
            s0 = {'P': [1 for i in range(n_verticies)], 'c':[1 if j == start_vertex else 0 for j in range(n_verticies)]}
            s_final = {'P': [0 if i != start_vertex else 1 for i in range(n_verticies)], 'c':[1 if j == start_vertex else 0 for j in range(n_verticies)]}

            # put s0 into visited states
            state = s0
            for i in range(n_verticies):
                if s_final != state:
                    string_state = stringify_dict(state)
                    visited_states[string_state] = True
                    if np.random.rand() >= epsilon_t:
                        torch_state = [*state["P"], *state["c"]]
                        torch_state = torch.FloatTensor(torch_state).to(device)
                        out = nndp_model(torch_state).argmax()
                        print(out)
                    else:
                        out = random_decision(state, visited_states, device)

                    state['P'][out] = 0
                else:
                    cost = rebuild_route(visited_states, graph)


def rebuild_route(visited_states, graph):
    print(len(visited_states))

def loss()

'''
    Given a map, returns a string form of the map

    m: instance of a map
'''
def stringify_dict(m):
    return json.dumps(m)


def get_feasible_decisions(curr_state, visited_states):
    c = curr_state["c"]
    P = curr_state["P"]
    # get idx of starting vertex
    idx = [i for i, sv in enumerate(c) if sv == 1][0]
    # get decision indicies
    decisions = [i for i in range(len(c)) if P[i] == 1 and i != idx]
    feas_decisions = []

    for f_idx in decisions:
        p = deepcopy(P)
        cs = deepcopy(c)

        p[f_idx] = 0
        pc = {"P" : p, "c" : cs}
        feas_decisions.append(pc)

    feas_decisions = [fd for fd in feas_decisions if stringify_dict(fd) not in visited_states]

    return feas_decisions

'''
    Makes a random feasible decision from the state
    params:
        curr_state: map with keys "P" and "c", where "P" is a 
        one-hot vector of visited and not visited verticies from the state and "c"
        is the starting vertex of the state

        visited_states: map of all the visited states, where the keys of the map are
        the states that have been visited

        device: device to send the input data to. ("cpu" or "cuda")
'''
def random_decision(curr_state, visited_states, device):
    feas_decisions = get_feasible_decisions(curr_state, visited_states)
    rand_decision = np.random.randint(len(feas_decisions))

    final_decision = feas_decisions[rand_decision]

    final_decision = [i for i in range(len(final_decision['P'])) if final_decision['P'][i] != curr_state['P'][i]][0]

    return final_decision