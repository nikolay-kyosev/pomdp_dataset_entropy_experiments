
import random
import numpy as np
import cvxpy as cp
import scipy
import pomdp_definitions
import mdp_utilities
import mdp_maxent_common




def make_constriants(d_sa_list, mdp: pomdp_definitions.Mdp, T):
    constraints = []
    sa_shape = (mdp.S, mdp.A)

    for t in range(T):
        for s in range(mdp.S):
            # Sum over actions for current state
            summ = 0
            for a in range(mdp.A):
                index = np.ravel_multi_index((s, a), sa_shape)
                summ += d_sa_list[t][index]

            # Flow conservation constraint
            second_sum = 0
            if not t == 0:
                for s_ in range(mdp.S):
                    for a_ in range(mdp.A):
                        index = np.ravel_multi_index((s_, a_), sa_shape)
                        second_sum += d_sa_list[t-1][index] * mdp.P[s_, a_, s]
            if t == 0:
                second_sum += mdp.d0[s]
            constraints.append(summ == second_sum)
    return constraints

def make_objective(mdp: pomdp_definitions.Mdp, xi_sa_list, entropy_dimension: mdp_maxent_common.EntropyDimension, T):
    d_sa = cp.Variable((mdp.S * mdp.A), name=f'd_sa', nonneg=True)
    constraints = [d_sa == 1/T * cp.sum(xi_sa_list)]
    if entropy_dimension == mdp_maxent_common.EntropyDimension.STATE:
        d_s = cp.Variable((mdp.S,), name=f'd_s', nonneg=True)
        for s in range(mdp.S):
            summ = 0
            for a in range(mdp.A):
                index = np.ravel_multi_index((s, a), (mdp.S, mdp.A))
                summ += d_sa[index]
            constraints.append(d_s[s] == summ)
        return cp.Maximize(cp.sum(cp.entr(d_s))), constraints, d_s
    elif entropy_dimension == mdp_maxent_common.EntropyDimension.STATE_ACTION:
        return cp.Maximize(cp.sum(cp.entr(d_sa))), constraints, d_sa

    elif entropy_dimension == mdp_maxent_common.EntropyDimension.TRANSITION:
        d_sas = cp.Variable((mdp.S*mdp.A*mdp.S,), name=f'd_sas', nonneg=True)
        for s in range(mdp.S):
            summ = 0
            for a in range(mdp.A):
                index_sa = np.ravel_multi_index((s, a), (mdp.S, mdp.A))
                for s_ in range(mdp.S):
                    index_sas = np.ravel_multi_index((s, a, s_), (mdp.S, mdp.A, mdp.S))
                    constraints.append(d_sas[index_sas] == mdp.P[s, a, s_] * d_sa[index_sa])
        return cp.Maximize(cp.sum(cp.entr(d_sas))), constraints, d_sas

    else:
        raise Exception('Error: Unknown entropy dimension.')

def calculate_d_sa(mdp: pomdp_definitions.Mdp, policies):
    distribution = np.zeros((mdp.S * mdp.A, ), dtype=np.float32)
    d_0 = np.zeros((mdp.S,), dtype=np.float32)
    d_0[0] = 1
    matrix = np.identity(mdp.S, dtype=np.float32)
    distributions = []
    for t in range(0, len(policies)):
        policy = np.array(policies[t], dtype=np.float32)
        d_t = d_0 @ matrix
        policy_ = policy.reshape(mdp.S, mdp.A)
        d_t = d_t.reshape((mdp.S, 1))
        distribution += (policy_ * d_t).flatten()
        # print(f'd_sa_actual: {distribution}')
        distributions.append(distribution)

        #Update matrix
        policy_ = policy.reshape((mdp.S, mdp.A, 1))
        policy_matrix = np.sum(mdp.P*policy_, axis=1)
        # print(policy_matrix)
        matrix = matrix @ policy_matrix
    return distributions

def get_all_expected_distributions(mdp, policies):
    d_sa = calculate_d_sa(mdp, policies)[-1]
    d_sa = d_sa/np.sum(d_sa)
    d_sa = np.reshape(d_sa, (mdp.S, mdp.A))
    d_s = np.zeros((mdp.S, ), dtype=np.float32)
    d_sas = np.zeros((mdp.S, mdp.A, mdp.S,), dtype=np.float32)
    for s in range(mdp.S):
        for a in range(mdp.A):
            d_s[s] += d_sa[s, a]
            for s_ in range(mdp.S):
                d_sas[s, a, s_] = d_sa[s, a] * mdp.P[s, a, s_]
    d_s = d_s.flatten()
    d_sa = d_sa.flatten()
    d_sas = d_sas.flatten()
    return d_s, d_sa, d_sas


def get_all_expected_distributions_xi_sa_list(mdp, xi_sa_list, ed:  mdp_maxent_common.EntropyDimension):
    d_sa = np.zeros((mdp.S * mdp.A,), dtype=np.float32)
    for i in range(len(xi_sa_list)):
        # if ed == mdp_maxent_common.EntropyDimension.STATE and i == len(xi_sa_list) - 1:
        #     print('Skipping')
        #     continue
        xi_sa = xi_sa_list[i]
        d_sa += xi_sa.value
    d_sa = d_sa/np.sum(d_sa)
    d_sa = np.reshape(d_sa, (mdp.S, mdp.A))
    d_s = np.zeros((mdp.S, ), dtype=np.float32)
    d_sas = np.zeros((mdp.S, mdp.A, mdp.S,), dtype=np.float32)
    for s in range(mdp.S):
        for a in range(mdp.A):
            d_s[s] += d_sa[s, a]
            for s_ in range(mdp.S):
                d_sas[s, a, s_] = d_sa[s, a] * mdp.P[s, a, s_]
    d_s = d_s.flatten()
    d_sa = d_sa.flatten()
    d_sas = d_sas.flatten()
    return d_s, d_sa, d_sas

def solve_problem(mdp: pomdp_definitions.Mdp, T, entropy_dimension: mdp_maxent_common.EntropyDimension = mdp_maxent_common.EntropyDimension.STATE_ACTION, print_=False):
    xi_sa_list = []
    if entropy_dimension == mdp_maxent_common.EntropyDimension.STATE:
        T = T+1
    for t in range(T):
        xi_sa_t = cp.Variable((mdp.S * mdp.A,), name=f'xi_sa_{t}', nonneg=True)
        xi_sa_list.append(xi_sa_t)
    p_flattened = mdp.P.flatten()
    sa_shape = (mdp.S, mdp.A)
    

    objective, constraints, d = make_objective(mdp, xi_sa_list, entropy_dimension, T)
    constraints += make_constriants(xi_sa_list, mdp, T)

    prob = cp.Problem(objective, constraints)
    prob.solve()
    if not prob.status == 'optimal':
        if prob.status == 'optimal_inaccurate':
            print(f'Warning: Problem status is {prob.status}!')
        else:
            raise Exception(f'Error: Problem status is not optimal! Problem status: {prob.status}!')

    policies = []
    for t in range(0,T):
        policies.append(mdp_maxent_common.make_policy(mdp, xi_sa_list[t].value))
    distributions = get_all_expected_distributions_xi_sa_list(mdp, xi_sa_list, entropy_dimension)
    d_s, d_sa, d_sas = tuple(map(lambda x: scipy.stats.entropy(x), distributions))

    # print(f'Entr1: {entr_1}\nEntr2: {entr_2}')
    return policies, d_s, d_sa, d_sas

    



if __name__ == '__main__':
    mdp = mdp_utilities.generate_river_swim_mdp()

    print(mdp.is_well_formed())

    # print(f'Well formed: {mdp.is_well_formed()}')
    np.set_printoptions(suppress=True,precision=3)

    T = 1
    policies, distr = solve_problem(mdp, T, mdp_maxent_common.EntropyDimension.STATE_ACTION, print_=False)
    print(f'Distribution: {distr}')
    print(f'Entropy: {scipy.stats.entropy(distr)}')