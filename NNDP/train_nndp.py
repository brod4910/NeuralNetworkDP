# python imports
from queue import Queue
import json
from copy import deepcopy

# library imports
import torch
import torch.nn as nn
import numpy as np

'''
    Trains the neural network dynamic programming model

    params:
        args: CLI arguments which contain hyper-parameters configurations

        nndp_model: network that will be tested on the graph

        graph: contains all verticies and their respective edge costs to all other verticies

        n_verticies: number of verticies in the graph

        device: device to send the input data to for network training. ("cpu" or "cuda")
'''
def train(args, nndp_model, graph, n_verticies, optimizer, device):
    epsilon_t = init_epsilon_t = .995
    epsilon_t_decay = .95
    decay_frequency = 10
    data_pool = Queue(maxsize=1000)

    for t in range(args.iters):
        for start_vertex in range(n_verticies):
            Q = Queue()
            visited_states = {}
            # initial state (start at vertex 1)
            s0 = {'P': [1 for i in range(n_verticies)], 'c':[1 if j == start_vertex else 0 for j in range(n_verticies)]}
            s_final = {'P': [0 if i != start_vertex else 1 for i in range(n_verticies)], 'c':[1 if j == start_vertex else 0 for j in range(n_verticies)]}
            # print(s_final)

            Q.put(s0)
            # put s0 into visited states
            string_state = stringify_dict(s0)
            visited_states[string_state] = True
            while not Q.empty():
                curr_state = Q.get()
                if curr_state != s_final:
                    # stringify states
                    string_state = stringify_dict(curr_state)
                    if np.random.rand() >= epsilon_t:
                        loss, substates, min_state = loss_function(nndp_model, graph, curr_state, visited_states, device)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        a_state, substates = random_decision(curr_state, visited_states, device)
                    for ss in substates:
                        string_state = stringify_dict(ss)
                        if string_state not in visited_states:
                            Q.put(ss)
                            visited_states[string_state] = True

        epsilon_t = init_epsilon_t * (epsilon_t_decay ** (t // decay_frequency))
        print(epsilon_t)
        test(nndp_model, n_verticies, graph, device)

def test(nndp_model, n_verticies, graph, device):
    for start_vertex in range(n_verticies):
        s0 = {'P': [1 for i in range(n_verticies)], 'c':[1 if j == start_vertex else 0 for j in range(n_verticies)]}
        s_final = {'P': [0 if i != start_vertex else 1 for i in range(n_verticies)], 'c':[1 if j == start_vertex else 0 for j in range(n_verticies)]}
        Q = Queue()
        Q.put(s0)
        visited_states = {}
        for i in range(n_verticies):
            curr_state = Q.get()
            if curr_state != s_final:
                loss, substates, min_state = loss_function(nndp_model, graph, curr_state, visited_states, device)
                print(min_state)

            for ss in substates:
                string_state = stringify_dict(ss)
                if string_state not in visited_states:
                    Q.put(ss)
                    visited_states[string_state] = True
'''
    Given a map, returns a string form of the map

    m: instance of a map
'''
def stringify_dict(m):
    return json.dumps(m)

def loss_function(nndp_model, graph, curr_state, visited_states, device):
    decision_cost, a_state, substates = decision(nndp_model, curr_state, visited_states, graph, device)
    output = forward_pass(curr_state, nndp_model, device)
    loss = torch.pow(output - decision_cost, 2)

    return loss, substates, a_state
    
'''
    Gets all possible substates given a state.
    params:
        curr_state: map with keys "P" and "c", where "P" is a 
        one-hot vector of visited and not visited verticies from the state and "c"
        is the starting vertex of the state
'''
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

'''
    Get all possible feasible dedcisions from the state
    params:
        curr_state: map with keys "P" and "c", where "P" is a 
        one-hot vector of visited and not visited verticies from the state and "c"
        is the starting vertex of the state

        visited_states: map of all the visited states, where the keys of the map are
        the states that have been visited
'''
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

        # cs[idx] = 0
        # cs[f_idx] = 1
        p[f_idx] = 0
        pc = {"P" : p, "c" : cs}
        feas_decisions.append(pc)

    feas_decisions = [fd for fd in feas_decisions if stringify_dict(fd) not in visited_states]

    return feas_decisions, idx

'''
    Makes a feasible decision according to equation (2) in the paper
    params:
        nndp_model: network model that will be making decisions

        curr_state: map with keys "P" and "c", where "P" is a 
        one-hot vector of visited and not visited verticies from the state and "c"
        is the starting vertex of the state

        visited_states: map of all the visited states, where the keys of the map are
        the states that have been visited

        graph: contains all verticies and their respective edge costs to all other verticies

        device: device to send the input data to. ("cpu" or "cuda")
'''
def decision(nndp_model, curr_state, visited_states, graph, device):
    c = curr_state["c"]
    P = curr_state["P"]

    feas_decisions, idx = get_feasible_decisions(curr_state, visited_states)

    arg_min = float("inf")
    arg_min_state = arg_min_substates = None
    # iterate over all feasible decisions
    for feas_decision in feas_decisions:
        p = deepcopy(P)
        cs = deepcopy(c)
        feas_idx = [i for i in range(len(feas_decision['P'])) if feas_decision['P'][i] != curr_state['P'][i]][0]

        p[feas_idx] = 0

        pc = {"P" : p, "c" : cs}
        substates = get_substates(pc)
        outputs = 0
        for state in substates:
            outputs += forward_pass(state, nndp_model, device)

        outputs += graph[idx][str(feas_idx)]
        if outputs < arg_min:
            arg_min = outputs
            arg_min_state = pc
            arg_min_substates = substates

    return arg_min, arg_min_state, arg_min_substates

def forward_pass(state, nndp_model, device):
    state = [*state["P"], *state["c"]]
    state = torch.FloatTensor(state).to(device)
    output = nndp_model(state)
    return output

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
    # print('curr_state state: ', curr_state)
    feas_decisions, idx = get_feasible_decisions(curr_state, visited_states)
    # print('decisions: ', feas_decisions)
    rand_decision = np.random.randint(len(feas_decisions))

    final_decision = feas_decisions[rand_decision]

    substates = get_substates(final_decision)

    return final_decision, substates


