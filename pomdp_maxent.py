import random
import numpy as np
import cvxpy as cp
import scipy
import pomdp_definitions
import mdp_utilities
from mdp_maxent_common import EntropyDimension as ed

def make_constraints(xi_ba_list, pomdp: pomdp_definitions.Pomdp, belief_mdp:pomdp_definitions.BeliefMdp, T):
    constraints = []
    ba_shape = (belief_mdp.belief_count, pomdp.A)

    b_tuple_list = list(map(lambda x: tuple(x), belief_mdp.belief_list))
    for t in range(T):
        for b_index, b in enumerate(belief_mdp.belief_list):
            # Sum over actions for current belief
            summ = 0
            for a in range(pomdp.A):
                index = np.ravel_multi_index((b_index, a), ba_shape)
                summ += xi_ba_list[t][index]

            #Flow constraint
            second_summ = 0
            if not t == 0:
                for b_index_, b_ in enumerate(belief_mdp.belief_list):
                    for a in range(pomdp.A):
                        for z in range(pomdp.Z):
                            pr_b_new, b_new, _ = pomdp_definitions.belief_update(pomdp, b_, a, z)
                            b_new = tuple(b_new)
                            if not b_new in b_tuple_list:
                                continue
                            if (b_new == b).all():
                                index = np.ravel_multi_index((b_index_, a), ba_shape)
                                second_summ += xi_ba_list[t-1][index] * pr_b_new
            
            #If we are in the initial belief
            if t == 0 and tuple(b) == tuple(pomdp.d0):
                second_summ += 1
            # print(summ == second_summ)
            constraints.append(summ == second_summ)
    return constraints

def make_belief_action_matrix(belief_matrix: np.ndarray, pomdp: pomdp_definitions.Pomdp):
    B = belief_matrix.shape[1]
    S = pomdp.S
    A = pomdp.A
    M = np.zeros((S*A, B*A))
    for b in range(B):
        for a in range(A):
            for s in range(S):
                row = s * A + a        # group rows by (s,a)
                col = b * A + a        # group cols by (b,a)
                M[row, col] = belief_matrix[s, b]
    return M

def make_belief_transition_matrix(belief_matrix: np.ndarray, pomdp: pomdp_definitions.Pomdp):
    B = belief_matrix.shape[1]
    S = pomdp.S
    A = pomdp.A
    M = np.zeros((S*A*S, B*A))
    for b_idx in range(B):
        for a_idx in range(A):
            col = b_idx * A + a_idx

            # Loop over rows: (s, a, s')
            for s_idx in range(S):
                for sprime_idx in range(S):
                    row = s_idx*(A*S) + a_idx*S + sprime_idx

                    # For example, place BeliefMatrix[s, b]
                    # (You could also involve sprime_idx if desired)
                    M[row, col] = pomdp.P[s_idx, a_idx, sprime_idx] * belief_matrix[s_idx, b_idx]
    return M

def make_objective(pomdp: pomdp_definitions.Mdp, belief_mdp: pomdp_definitions.BeliefMdp, xi_ba_list, entropy_dimension: ed, T):
    d_ba = cp.Variable((belief_mdp.belief_count * pomdp.A), name=f'd_ba', nonneg=True)
    constraints = [d_ba == 1/(T) * cp.sum(xi_ba_list)]
    belief_matrix = np.column_stack(belief_mdp.belief_list)
    if entropy_dimension == ed.STATE:
        d_b = cp.Variable((belief_mdp.belief_count,), name=f'd_b', nonneg=True)
        for b_index, b in enumerate(belief_mdp.belief_list):
            summ = 0
            for a in range(pomdp.A):
                index = np.ravel_multi_index((b_index, a), (belief_mdp.belief_count, pomdp.A))
                summ += d_ba[index]
            constraints.append(d_b[b_index] == summ)
        return cp.Maximize(cp.sum(cp.entr(belief_matrix@d_b))), constraints, d_b
    elif entropy_dimension == ed.STATE_ACTION:
        belief_action_matrix = make_belief_action_matrix(belief_matrix, pomdp)
        # print(belief_action_matrix)
        return cp.Maximize(cp.sum(cp.entr(belief_action_matrix@d_ba))), constraints, d_ba
    elif entropy_dimension == ed.TRANSITION:
        giga_matrix = make_belief_transition_matrix(belief_matrix, pomdp)
        return cp.Maximize(cp.sum(cp.entr(giga_matrix@d_ba))), constraints, d_ba
    else:
        raise Exception('Error: Unknown entropy dimension.')

def solve_problem(pomdp: pomdp_definitions.Pomdp, T, entropy_dimension: ed = ed.STATE):
    if entropy_dimension == ed.STATE:
        T = T + 1
    belief_mdp = pomdp_definitions.BeliefMdp(pomdp, T)
    xi_ba_list = []
    for t in range(T):
        xi_ba_t = cp.Variable((belief_mdp.belief_count*pomdp.A), name=f'xi_ba_{t}', nonneg=True)
        xi_ba_list.append(xi_ba_t)
    

    objective, constraints, d = make_objective(pomdp, belief_mdp, xi_ba_list, entropy_dimension, T)
    constraints += make_constraints(xi_ba_list, pomdp, belief_mdp, T)

    prob = cp.Problem(objective, constraints)
    prob.solve()
    if not prob.status == 'optimal':
        if prob.status == 'optimal_inaccurate':
            print(f'Warning: Problem status is {prob.status}!')
        else:
            raise Exception(f'Error: Problem status is not optimal! Problem status: {prob.status}!')

    policy = []
    for t in range(T):
        belief_dict = {}

        xi_ba = xi_ba_list[t].value
        xi_ba = xi_ba.reshape((belief_mdp.belief_count, pomdp.A))
        row_sum = xi_ba.sum(axis=1, keepdims=True)
        np.set_printoptions(suppress=True, precision=2)
        # print(f'x_b_{t} = \n{row_sum}')
        # if t == 1:
        #     print(f'x_ba_{t} = \n{xi_ba}')
        row_sum[row_sum == 0] = 1
        xi_ba = xi_ba / row_sum

        for b_indx, b in enumerate(belief_mdp.belief_list):
            b = tuple(b)
            if sum(xi_ba[b_indx]) != 0:
                belief_dict[b] = xi_ba[b_indx]
        policy.append(belief_dict)

    #make the entropy:
    belief_matrix = np.column_stack(belief_mdp.belief_list)
    np.set_printoptions(suppress=True)
    # print(f'belief matrix: \n{belief_matrix}')
    d_ = d.value
    d_ = d_/sum(d_)
    if entropy_dimension == ed.STATE:
        d_s = belief_matrix@d_
        return policy, d_s
    elif entropy_dimension == ed.STATE_ACTION:
        giga_matrix = make_belief_action_matrix(belief_matrix, pomdp)
        d_sa = giga_matrix@d_
        return policy, d_sa
    elif entropy_dimension == ed.TRANSITION:
        giga_matrix = make_belief_transition_matrix(belief_matrix, pomdp)
        d_sas = giga_matrix@d_
        return policy, d_sas
    else:
        raise Exception('Error: Invalid transition.')

if __name__ == '__main__':
    pomdp = mdp_utilities.generate_river_swim_unobservable(3)
    print(pomdp.is_well_formed())
    
    T = 5
    
    belief_mdp = pomdp_definitions.BeliefMdp(pomdp, T)
    print(belief_mdp.belief_count)
    
    # print(belief_mdp.belief_count)
    # for belief in belief_mdp.belief_list:
    #     print(np.array(belief))

    p, d = solve_problem(pomdp, T, ed.TRANSITION)
    print(d)
    print(pomdp.P.flatten())
    dot = belief_mdp.make_graphviz_graph(policy=p)
    dot.render('belief_mdp_policy', format='png', cleanup=True)
    pass
