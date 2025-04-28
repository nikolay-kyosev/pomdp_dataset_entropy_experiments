import numpy as np

import mdp_utilities
import environment
import pomdp_definitions as pomdps
import pomdp_maxent
import mdp_finite_maxent_solver
import scipy

import pomdp_parser

from mdp_maxent_common import EntropyDimension as Edim
from pomdp_dataset_entropy import DatasetDecision as Dec

def make_distribution(pomdp: pomdps.Pomdp, ed: Edim):
    distribution = None
    if ed == Edim.STATE:
        distribution = np.zeros((pomdp.S,), dtype=np.float32)
    elif ed == Edim.STATE_ACTION:
        distribution = np.zeros((pomdp.S, pomdp.A), dtype=np.float32)
    elif ed == Edim.TRANSITION:
        distribution = np.zeros((pomdp.S, pomdp.A, pomdp.S,), dtype=np.float32)
    else:
        raise Exception('Error: Invalid entropy dimension')
    return distribution

def make_belief_mdp_policy_graph(pomdp, policies, T, entropy_dimension):
    if entropy_dimension == Edim.STATE:
        T = T + 1
    belief_mdp = pomdps.BeliefMdp(pomdp, T)
    dot = belief_mdp.make_graphviz_graph(policy=policies)
    dot.render('belief_mdp_policy', format='png', cleanup=True)

def conduct_experiment(pomdp: pomdps.Pomdp, T: int, n_tranjectories: int, ed: Edim, qmdp_approach=False, make_belief_mdp_graph=False):
    policies = None
    expected_d = None
    expected_s = None
    expected_sa = None
    expected_sas = None
    if not qmdp_approach:
        policies, expected_d = pomdp_maxent.solve_problem(pomdp, T, ed)
        if make_belief_mdp_graph:
            make_belief_mdp_policy_graph(pomdp, policies, T, ed)
    else:
        policies, expected_s, expected_sa, expected_sas = mdp_finite_maxent_solver.solve_problem(pomdp, T, ed)
    entropies_s, entropies_sa, entropies_sas = sample_trajectories_randomized(pomdp, T, n_tranjectories, policies, qmdp_approach)
    if qmdp_approach:
        return expected_s, expected_sa, expected_sas, entropies_s, entropies_sa, entropies_sas
    return expected_d, entropies_s, entropies_sa, entropies_sas


def sample_trajectories_randomized(pomdp: pomdps.Pomdp, T: int, n_tranjectories: int, policies, qmdp_approach=False):
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
            if not qmdp_approach:
                action_distr = policy[tuple(b)]
            else:
                policy = policy.reshape((pomdp.S, pomdp.A))
                action_distr = b @ policy
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
    return entropies_s, entropies_sa, entropies_sas

def sample_dataset_based_policy(pomdp: pomdps.Pomdp, policy, dataset_decision_function):
    action_selection_fn = lambda h : policy[h]

    distribution_s = np.zeros((pomdp.S,), dtype=np.float32)
    distribution_sa = np.zeros((pomdp.S, pomdp.A), dtype=np.float32)
    distribution_sas = np.zeros((pomdp.S, pomdp.A, pomdp.S,), dtype=np.float32)
    entropies_s = []
    entropies_sa = []
    entropies_sas = []
    
    env = environment.Env(pomdp)
    dataset = ()
    action_dataset = []
    collect_data = True
    while collect_data:
        b, s = env.reset()
        actions = []
        action_dataset.append(actions)
        distribution_s[s] += 1
        h = tuple([tuple(b)])
        should_terminate = False
        while True:
            decision = dataset_decision_function(([], action_dataset, []))
            if decision == Dec.RESET:
                break
            elif decision == Dec.TERMINATE:
                collect_data=False
                break
            action = action_selection_fn(h)
            actions.append(action)
            b_new, z, s_new = env.step(action)
            #Update the distribution
            distribution_s[s_new] += 1
            distribution_sa[s, action] += 1
            distribution_sas[s, action, s_new] += 1
            #Update the state and the belief
            b = b_new
            h = h + (action, z)
            s = s_new
        #Update the dataset
        dataset = dataset + h
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
    d_s = distribution_s.flatten()
    d_sa = distribution_sa.flatten()
    d_sas = distribution_sas.flatten()
    return entropies_s[-1], entropies_sa[-1], entropies_sas[-1]


