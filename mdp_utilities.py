import numpy as np
import pomdp_definitions as pomdps

def generate_chain_mdp(n_states):
    # Initialize the table with zeros
    mdp = pomdps.Mdp(n_states=n_states, n_actions=2)
    mdp.d0[0] = 1
    
    # Fill in the transitions for each state
    for i in range(n_states):
        if i == 0:
            # State s0: a0 -> s0, a1 -> s1
            mdp.P[i, 0, i] = 1.0  # s0, a0 -> s0
            mdp.P[i, 1, i+1] = 1.0  # s0, a1 -> s1
        elif i == n_states - 1:
            # State s(n-1): a0 -> s(n-2), a1 -> s(n-1)
            mdp.P[i, 0, i-1] = 1.0  # s(n-1), a0 -> s(n-2)
            mdp.P[i, 1, i] = 1.0  # s(n-1), a1 -> s(n-1)
        else:
            # State s1 to s(n-2): a0 -> s(i-1), a1 -> s(i+1)
            mdp.P[i, 0, i-1] = 1.0  # s(i), a0 -> s(i-1)
            mdp.P[i, 1, i+1] = 1.0  # s(i), a1 -> s(i+1)
    
    return mdp

def generate_3state_mdp():
    mdp = generate_chain_mdp(n_states=3)
    mdp.d0[0] = 0
    mdp.d0[1] = 1
    return mdp

def generate_river_swim_mdp(n_states=3):
    mdp = generate_chain_mdp(n_states)
    for i in range(mdp.S - 1):
        mdp.P[i, 1, i+1] = 0.3
        mdp.P[i, 1, i] = 0.7
    return mdp

def generate_chain_mdp_unobservable(n_states):
    mdp = generate_chain_mdp(n_states)
    pomdp = pomdps.Pomdp(n_states, mdp.A, 1)
    pomdp.d0 = mdp.d0
    pomdp.P = mdp.P
    pomdp.O[:, :, 0] = 1
    return pomdp

def generate_river_swim_unobservable(n_states=3):
    pomdp = generate_chain_mdp_unobservable(n_states)
    for i in range(pomdp.S - 1):
        pomdp.P[i, 1, i+1] = 0.3
        pomdp.P[i, 1, i] = 0.7
    return pomdp

def generate_river_swim_partially_observable(n_states=3):
    mdp = generate_river_swim_mdp(n_states)
    pomdp = pomdps.Pomdp(n_states, 2, 2)
    pomdp.d0 = mdp.d0
    pomdp.P = mdp.P
    pomdp.O[:, :, 0] = 1
    pomdp.O[:, n_states-1, 0] = 1/3
    pomdp.O[:, n_states-1, 1] = 2/3
    return pomdp

def generate_river_swim_partially_observable_multiple_end_component():
    n_states = 4
    pomdp = pomdps.Pomdp(n_states, 3, 2)
    i=0
    s0 = i
    s1 = i+1
    s2 = i+2
    s3 = i+3
    pomdp.d0[s0] = 1
    pomdp.P[s0, 0, s0] = 1
    pomdp.P[s0, 1, s1] = .7
    pomdp.P[s0, 1, s0] = .3
    pomdp.P[s0, 2, s0] = 1 # Rest action, nothing happens
    pomdp.P[s1, 0, s0] = 1
    pomdp.P[s1, 1, s2] = .7
    pomdp.P[s1, 1, s1] = .3
    pomdp.P[s1, 2, s3] = 1 # Rest action, we go to sink state
    pomdp.P[s2, 0, s1] = 1
    pomdp.P[s2, 1, s2] = 1
    pomdp.P[s2, 2, s2] = 1
    pomdp.P[s3, :, s3] = 1
    pomdp.O[:, s0, 0] = 1.0
    pomdp.O[:, s1, 0] = 1.0
    pomdp.O[:, s2, 0] = .7
    pomdp.O[:, s2, 1] = .3
    pomdp.O[:, s3, 0] = 1.0
    return pomdp
    
if __name__ == '__main__':
    pomdp = generate_river_swim_partially_observable_multiple_end_component()
    print(pomdp.is_well_formed())

# def generate_chain_pomdp(n_states):
#     pomdp = pomdps.Pomdp(n_states=n_states, n_actions=2, n_observations=n_states)
#     for i in range(n_states):
#         pomdp.O[i, i] = 1.0
#         if i == 0:
#             # State s0: a0 -> s0, a1 -> s1
#             pomdp.P[i, 0, i] = 1.0  # s0, a0 -> s0
#             pomdp.P[i, 1, i+1] = 1.0  # s0, a1 -> s1
#         elif i == n_states - 1:
#             # State s(n-1): a0 -> s(n-2), a1 -> s(n-1)
#             pomdp.P[i, 0, i-1] = 1.0  # s(n-1), a0 -> s(n-2)
#             pomdp.P[i, 1, i] = 1.0  # s(n-1), a1 -> s(n-1)
#         else:
#             # State s1 to s(n-2): a0 -> s(i-1), a1 -> s(i+1)
#             pomdp.P[i, 0, i-1] = 1.0  # s(i), a0 -> s(i-1)
#             pomdp.P[i, 1, i+1] = 1.0  # s(i), a1 -> s(i+1)
    
#     return pomdp

# def generate_grid_mdp(n_states_per_row) -> pomdps.Mdp:
#     n_states = n_states_per_row**2
#     mdp = pomdps.Mdp(n_states=n_states, n_actions = 4)

#     for i in range(n_states):
#         (x,y) = np.unravel_index(i, (n_states_per_row, n_states_per_row))
#         for j in range(n_states):
#             (x_, y_) = np.unravel_index(i, (n_states_per_row, n_states_per_row))
#             distance = int(((x - x_)**2 + (y - y_)**2)**0.5)
#             if distance == 1:
#                 if x < x_:
#                     if y < y_:

