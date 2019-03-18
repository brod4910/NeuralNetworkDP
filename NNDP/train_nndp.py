# python imports
from queue import Queue
import json
from copy import deepcopy

# library imports
import torch
import torch.nn as nn
import numpy as np

def train(args, nndp_model, graph, n_verticies, device):

    epsilon_t = 1.0
    data_pool = Queue(maxsize=1000)

    for t in range(args.iters):
        # intialize data structures
        state_list = {}
        Q = Queue()
        visited_states = {}
        # initial state (start at vertex 1)
        s0 = {'P': [1 for i in range(n_verticies)], 'c':[1 if j == 0 else 0 for j in range(n_verticies)]}
        Q.put(s0)
        # put s0 into visited states
        string_state = stringify_dict(s0)
        visited_states[string_state] = True
        while not Q.empty():
            curr_state = Q.get()
            # stringify state
            string_state = stringify_dict(curr_state)
            # add state to state list
            state_list[string_state] = True
            if np.random.rand() >= epsilon_t:
                decision_a, a_state, substates = decision(nndp_model, curr_state, visited_states, graph, device)
            else:
                a_state, substates = random_decision(curr_state, visited_states, device)
            
            for ss in substates:
                string_state = stringify_dict(ss)
                if string_state not in visited_states:
                    Q.put(ss)
                    visited_states[string_state] = True



def stringify_dict(l):
    return json.dumps(l)

def loss_function(nndp_model, graph):
    pass

def get_substates(curr_state):
    c = curr_state["c"]
    P = curr_state["P"]
    # get idx of starting vertex
    idx = [i for i, sv in enumerate(c) if sv == 1][0]
    substates = []
    for i in range(len(P)):
        if i != idx and P[i] == 1:
            p = deepcopy(P)
            p[i] = 0
            substates.append({"P" : p, "c" : c})

    return substates

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

        cs[idx] = 0
        cs[f_idx] = 1
        p[f_idx] = 0
        pc = {"P" : p, "c" : cs}
        feas_decisions.append(pc)

    feas_decisions = [fd for fd in feas_decisions if stringify_dict(fd) not in visited_states]

    return feas_decisions

def decision(nndp_model, curr_state, visited_states, graph, device):
    c = curr_state["c"]
    P = curr_state["P"]

    feas_decisions = get_feasible_decisions(curr_state, visited_states)

    arg_min = float("inf")
    arg_min_state = arg_min_substates = None
    # iterate over all feasible decisions
    for f_idx in feas_decisions:
        p = deepcopy(P)
        cs = deepcopy(c)

        cs[idx] = 0
        cs[f_idx] = 1
        p[f_idx] = 0

        pc = {"P" : p, "c" : cs}
        substates = get_substates(pc)
        outputs = 0
        for state in substates:
            state = [*state["P"], *state["c"]]
            state = torch.FloatTensor(state).to(device)
            outputs += nndp_model(state)
        
        outputs += graph[idx][str(f_idx)]
        if outputs < arg_min:
            arg_min = outputs
            arg_min_state = pc
            arg_min_substates = substates

    return arg_min, arg_min_state, arg_min_substates

def random_decision(curr_state, visited_states, device):
    feas_decisions = get_feasible_decisions(curr_state, visited_states)
    rand_decision = np.random.randint(len(feas_decisions))

    final_decision = feas_decisions[rand_decision]
    substates = get_substates(final_decision)

    return final_decision, substates


