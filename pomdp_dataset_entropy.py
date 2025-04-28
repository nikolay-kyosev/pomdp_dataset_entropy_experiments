from enum import Enum
import numpy as np
import pomdp_definitions as pomdps
import mdp_utilities as utils
import scipy
import math
from mdp_maxent_common import EntropyDimension as Edim

class DatasetDecision(Enum):
    CONTINUE = 0
    RESET = 1
    TERMINATE = 2


HISTORIES_EVALUATED_ = 0

def calculate_expected_entropy_policy(pomdp: pomdps.Pomdp, policy, dataset, ed: Edim, dataset_decision_function, use_approx=False):
    belief_dataset, action_dataset, observation_dataset = dataset
    belief_history = belief_dataset[-1]
    action_history = action_dataset[-1]
    observation_history = observation_dataset[-1]

    b = belief_history[-1]
    argmax_a = 0
    q_val = 0
    #Immediate reward
    dataset_decision = dataset_decision_function(dataset)
    if dataset_decision == DatasetDecision.TERMINATE:
        reward = calculate_expected_dataest_reward(pomdp, dataset, ed, use_approx)
        return reward
    elif dataset_decision == DatasetDecision.RESET:
        raise Exception('Should not happen!')
        dataset_ = (belief_dataset + [[pomdp.d0]], action_dataset + [[]], observation_dataset + [[]])
        value = calculate_expected_entropy_policy(pomdp, policy, dataset_, ed, dataset_decision_function, use_approx)
        return value
    #If we have continue then we just continue...

    history = make_dataset_tuple(dataset)
    a = policy[history]
    q_val_a = 0
    for z in range(pomdp.Z):
        pr_z, b_, _ = pomdps.belief_update(pomdp, b, a, z)
        if pr_z == 0:
            continue
        b_h = belief_history + [b_]
        a_h = action_history + [a]
        o_h = observation_history + [z]
        dataset_ = (belief_dataset[0:-1] + [b_h], action_dataset[0:-1] + [a_h], observation_dataset[0:-1] + [o_h])
        expected_entropy = calculate_expected_entropy_policy(pomdp, policy, dataset_, ed, dataset_decision_function, use_approx)
        q_val_a += pr_z * expected_entropy

    #Make the history tuple for the policy...

    return q_val_a

def calculate_entropy_fo_history(pomdp: pomdps.Pomdp, action_history, state_history, ed: Edim):
    d = None
    if ed == Edim.STATE:
        d = np.zeros((pomdp.S,), dtype=np.float32)
        for s in state_history:
            d[s] += 1
    elif ed == Edim.STATE_ACTION:
        d = np.zeros((pomdp.S, pomdp.A), dtype=np.float32)
        for t, a in enumerate(action_history):
            s = state_history[t]
            d[s, a] += 1
    elif ed == Edim.TRANSITION:
        d = np.zeros((pomdp.S, pomdp.A, pomdp.S), dtype=np.float32)
        for t, a in enumerate(action_history):
            s = state_history[t]
            s_ = state_history[t+1]
            d[s, a, s_] += 1
    d = d.flatten()
    if np.sum(d) == 0:
        if action_history:
            print(f'Distribution is zeros, returning 0.\nState_history: {state_history}, Action_history: {action_history}')
        return 0
    
    if np.sum(d) == 0:
        return 0
    d = d / np.sum(d)
    return scipy.stats.entropy(d)

def calculate_expected_entropy_hz(pomdp: pomdps.Pomdp, initial_belief, action_history, observation_history, ed: Edim, fb_pass=[], state_history=[]):
    final_entropy = 0
    if len(action_history)+1 == len(state_history):
        return calculate_entropy_fo_history(pomdp, action_history, state_history, ed)
    if not fb_pass:
        forward_part = [initial_belief]
        T = np.transpose(pomdp.P, ((1, 0, 2)))
        for t, a in enumerate(action_history):
            z = observation_history[t]
            O = np.diag(pomdp.O[a, :, z])
            T_ = T[a]
            f = forward_part[t]
            forward_part.append(f@T_@O)
        backward_part = [np.ones((pomdp.S,), dtype=np.float32)]
        for t, a in reversed(list(enumerate(action_history))):
            z = observation_history[t]
            O = np.diag(pomdp.O[a, :, z])
            T_ = T[a]
            length = len(forward_part) - 1
            index = length - (t+1)
            b = backward_part[index]
            result = T_@O@b
            backward_part.append(result)
        backward_part = list(reversed(backward_part))
        transitions_given_hz = []
        for i in range(len(forward_part)-1):
            a = action_history[i]
            z = observation_history[i]
            f = forward_part[i]
            b = backward_part[i+1]
            O = np.diag(pomdp.O[a, :, z])
            T_ = T[a]
            p = np.diag(f) @ T_ @ O @ np.diag(b)
            if np.sum(p) == 0:
                print(f'Error: sum of p = 0, f={f}, b={b}')
            p = p/np.sum(p,axis=1, keepdims=True)
            if np.any(np.isnan(p)):
                pass
                # raise Exception('Wtf, this shit broken!')
            transitions_given_hz.append(p)
        fb_pass = transitions_given_hz
    next_states = None
    if not state_history:
        next_states = initial_belief
    else:
        index = len(state_history) - 1
        s = state_history[index]
        P = fb_pass[index]
        next_states = P[s]
    expectation = 0
    for s_, p in enumerate(next_states):
        if p == 0:
            continue
        sh = state_history + [s_]
        expectation += p * calculate_expected_entropy_hz(pomdp, initial_belief, action_history, observation_history, ed, fb_pass, sh)
    return expectation

def calculate_expected_entropy_history(pomdp: pomdps.Pomdp, belief_history, action_history, observation_history, ed: Edim, state_history=[], test=False):
    # global HISTORIES_EVALUATED_
    if len(state_history) == len(belief_history):
        if test:
            print(len(state_history))
        if len(state_history) <= len(action_history):
            raise Exception('Should never happen!')
        d = None
        if ed == Edim.STATE:
            d = np.zeros((pomdp.S,), dtype=np.float32)
            for s in state_history:
                d[s] += 1
        elif ed == Edim.STATE_ACTION:
            d = np.zeros((pomdp.S, pomdp.A), dtype=np.float32)
            for t, a in enumerate(action_history):
                s = state_history[t]
                d[s, a] += 1
        elif ed == Edim.TRANSITION:
            d = np.zeros((pomdp.S, pomdp.A, pomdp.S), dtype=np.float32)
            for t, a in enumerate(action_history):
                s = state_history[t]
                s_ = state_history[t+1]
                d[s, a, s_] += 1
        d = d.flatten()
        if np.sum(d) == 0:
            if action_history:
                print(f'Distribution is zeros, returning 0.\nState_history: {state_history}, Action_history: {action_history}')
            return 0
        d = d / np.sum(d)
        res = scipy.stats.entropy(d)
        # HISTORIES_EVALUATED_ += 1
        return res
    
    final_entropy = 0
    next_states = None
    if not state_history:
        next_states = belief_history[0]
    else:
        t = len(state_history) - 1
        s = state_history[t]
        a = action_history[t]
        z = observation_history[t]

        next_states = pomdp.P[s, a] * pomdp.O[a, :, z]
        if not np.sum(next_states) == 0:
            next_states = next_states /  np.sum(next_states)
        else:
            return -1
    expected_values = np.zeros((pomdp.S,), dtype=np.float32)
    next_states_new = np.zeros((pomdp.S,), dtype=np.float32)
    for s_ in range(len(next_states)):
        p = next_states[s_]
        if p == 0:
            continue
        sh = state_history + [s_]
        res = calculate_expected_entropy_history(pomdp, belief_history, action_history, observation_history, ed, sh)
        if not res == -1:
            expected_values[s_] = res
            next_states_new[s_] = p
    next_states_new = next_states_new / np.sum(next_states_new)
    return next_states_new@expected_values

def calculate_expected_entropy_approx(pomdp: pomdps.Pomdp, belief_history, action_history, observation_history, ed: Edim):
    # print(f'called')
    T = np.transpose(pomdp.P, (1, 0, 2))
    #Continue
    forward_part = [belief_history[0]]
    for t, a in enumerate(action_history):
        z = observation_history[t]
        O = np.diag(pomdp.O[a, :, z])
        T_ = T[a]
        f = forward_part[t]
        forward_part.append(f@T_@O)
    backward_part = [np.ones((pomdp.S,), dtype=np.float32)]
    for t, a in reversed(list(enumerate(action_history))):
        z = observation_history[t]
        O = np.diag(pomdp.O[a, :, z])
        T_ = T[a]
        length = len(forward_part) - 1
        index = length - (t+1)
        b = backward_part[index]
        result = T_@O@b
        backward_part.append(result)
    backward_part = list(reversed(backward_part))
    d = None
    if ed == Edim.STATE:
        d = np.zeros((pomdp.S,), dtype=np.float32)
        for i in range(len(forward_part)):
            f = forward_part[i]
            b = backward_part[i]
            p = f*b
            p = p/np.sum(p)
            d += p
        d = d / np.sum(d)
        return scipy.stats.entropy(d)
    elif ed == Edim.STATE_ACTION:
        d = np.zeros((pomdp.S, pomdp.A,), dtype=np.float32)
        for i in range(len(forward_part)-1):
            a = action_history[i]
            f = forward_part[i]
            b = backward_part[i]
            p = f*b
            p = p/np.sum(p)
            d[:, a] += p
        d = d.flatten()
        summ = np.sum(d)
        if summ == 0:
            return 0
        d = d / summ
        return scipy.stats.entropy(d)
    elif ed == Edim.TRANSITION:
        d = np.zeros((pomdp.A, pomdp.S, pomdp.S), dtype=np.float32)
        for i in range(len(forward_part)-1):
            a = action_history[i]
            z = observation_history[i]
            f = forward_part[i]
            b = backward_part[i+1]
            O = np.diag(pomdp.O[a, :, z])
            T_ = T[a]
            p = np.diag(f) @ T_ @ O @ np.diag(b)
            if np.sum(p) == 0:
                print(f'Error: sum of p = 0, f={f}, b={b}')
            p = p/np.sum(p)
            d[a] += p
        d = d.flatten()
        summ = np.sum(d)
        if summ == 0 and action_history:
            raise Exception('Error: Should never happen!')
        if summ == 0:
            return 0
        d = d / summ
        return scipy.stats.entropy(d)
    else:
        raise Exception('Error: Other entropies unsupported!')
    






def calculate_expected_dataest_reward(pomdp: pomdps.Pomdp, dataset, ed: Edim, use_approx=False):
    belief_dataset, action_dataset, observation_dataset = dataset
    # global HISTORIES_EVALUATED_
    reward = 0
    for i in range(len(action_dataset)):
        belief_history = belief_dataset[i]
        action_history = action_dataset[i]
        observation_history = observation_dataset[i]
        if not use_approx:
            # HISTORIES_EVALUATED_ = 0
            reward += calculate_expected_entropy_hz(pomdp, belief_history[0], action_history, observation_history, ed)
            # print(f'Evaluated {HISTORIES_EVALUATED_} histories to compute exact reward.')
        else:
            reward += calculate_expected_entropy_approx(pomdp, belief_history, action_history, observation_history, ed)
    return reward

def make_dataset_tuple(dataset):
    belief_dataset, action_dataset, observation_dataset = dataset
    history = []
    for i in range(len(action_dataset)):
        b_h = belief_dataset[i]
        action_h = action_dataset[i]
        o_h = observation_dataset[i]
        history.append(tuple(b_h[0]))
        for t, a in enumerate(action_h):
            history.append(a)
            history.append(o_h[t])
    return tuple(history)

def dfs_solve_pomdp(pomdp: pomdps.Pomdp, dataset, policy, ed: Edim, dataset_decision_function, use_approx=False):
    belief_dataset, action_dataset, observation_dataset = dataset
    belief_history = belief_dataset[-1]
    action_history = action_dataset[-1]
    observation_history = observation_dataset[-1]

    b = belief_history[-1]
    argmax_a = 0
    q_val = 0
    #Immediate reward
    reward = calculate_expected_dataest_reward(pomdp, dataset, ed, use_approx)
    dataset_decision = dataset_decision_function(dataset)
    if dataset_decision == DatasetDecision.TERMINATE:
        return reward
    elif dataset_decision == DatasetDecision.RESET:
        dataset_ = (belief_dataset + [[pomdp.d0]], action_dataset + [[]], observation_dataset + [[]])
        value = dfs_solve_pomdp(pomdp, dataset_, policy, ed, dataset_decision_function, use_approx)
        return reward + value
    #If we have continue then we just continue...

    for a in range(pomdp.A):
        q_val_a = 0
        for z in range(pomdp.Z):
            pr_z, b_, _ = pomdps.belief_update(pomdp, b, a, z)
            if pr_z == 0:
                continue
            b_h = belief_history + [b_]
            a_h = action_history + [a]
            o_h = observation_history + [z]
            dataset_ = (belief_dataset[0:-1] + [b_h], action_dataset[0:-1] + [a_h], observation_dataset[0:-1] + [o_h])
            expected_entropy = dfs_solve_pomdp(pomdp, dataset_, policy, ed, dataset_decision_function, use_approx)
            q_val_a += pr_z * expected_entropy
        if q_val_a > q_val:
            q_val = q_val_a
            argmax_a = a

    #Make the history tuple for the policy...
    history = make_dataset_tuple(dataset)
    policy[history] = argmax_a

    return reward + q_val

def uniform_dataset_history(T, N, dataset):
    belief_dataset, action_dataset, observation_dataset = dataset
    if len(action_dataset) == N:
        if len(action_dataset[-1]) == T:
            return DatasetDecision.TERMINATE
    elif len(action_dataset[-1]) == T:
        return DatasetDecision.RESET
    return DatasetDecision.CONTINUE

def make_dataset_based_policy(pomdp: pomdps.Pomdp, T, N, ed: Edim, use_approx=False):
    policy = {}
    initial_belief = pomdp.d0
    f = lambda dataset : uniform_dataset_history(T, N, dataset)
    dataset = ([[initial_belief]], [[]], [[]])
    dfs_solve_pomdp(pomdp, dataset, policy, ed, f, use_approx)
    initial_b = (tuple(initial_belief),)
    # print(f'{initial_b} -> {policy[initial_b]}')
    # for h in policy:
    #     print(f'{h} -> {policy[h]}')
    # print(f'Expected entropy: {calculate_expected_entropy_policy(pomdp, policy, dataset, ed, f)}')
    return policy

if __name__ == '__main__':
    pomdp = utils.generate_river_swim_unobservable(3)
    policy = make_dataset_based_policy(pomdp, 3, 1, Edim.STATE, use_approx=False)
    for h in policy:
        print(f'{h} -> {policy[h]}')