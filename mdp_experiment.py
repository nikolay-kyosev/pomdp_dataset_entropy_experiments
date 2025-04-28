import numpy as np

import mdp_utilities
import environment
import pomdp_definitions as pomdps
import pomdp_maxent
import mdp_finite_maxent_solver
import scipy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pomdp_parser

from mdp_maxent_common import EntropyDimension as Edim


def conduct_experiment(pomdp: pomdps.Mdp, T: int, n_tranjectories: int, ed: Edim, make_belief_mdp_graph=False):
    policies, expected_s, expected_sa, expected_sas = mdp_finite_maxent_solver.solve_problem(pomdp, T, ed)
    distribution_s = np.zeros((pomdp.S,), dtype=np.float32)
    distribution_sa = np.zeros((pomdp.S, pomdp.A), dtype=np.float32)
    distribution_sas = np.zeros((pomdp.S, pomdp.A, pomdp.S,), dtype=np.float32)
    entropies_s = []
    entropies_sa = []
    entropies_sas = []
    
    env = environment.Env(pomdp)
    for i in range(n_tranjectories):
        b, s = env.reset()
        distribution_s[s] += 1
        for t in range(T):
            policy = policies[t]
            action_dstr = np.zeros((pomdp.A,), dtype=np.float32)
            policy = policy.reshape((pomdp.S, pomdp.A))
            action_distr = policy[s]
            action = np.random.choice(action_distr.shape[0], p=action_distr)
            b_new, z_new, s_new = env.step(action)
            #Update the distribution
            distribution_s[s_new] += 1
            distribution_sa[s, action] += 1
            distribution_sas[s, action, s_new] += 1
            #Update the state and the belief
            b = b_new
            s = s_new
        #Add to entropies array
        d = distribution_s.flatten()
        d = d / np.sum(d)
        entropies_s.append(scipy.stats.entropy(d))
        d = distribution_sa.flatten()
        d = d / np.sum(d)
        entropies_sa.append(scipy.stats.entropy(d))
        d = distribution_sas.flatten()
        d = d / np.sum(d)
        entropies_sas.append(scipy.stats.entropy(d))
    
    return expected_s, expected_sa, expected_sas, entropies_s, entropies_sa, entropies_sas


