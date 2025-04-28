from enum import Enum
import numpy as np
import pomdp_definitions

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

class EntropyDimension(Enum):
    STATE = 0
    STATE_ACTION = 1
    TRANSITION = 2

def make_policy(mdp: pomdp_definitions.Mdp, d_sa):
    policy = []
    for s in range(mdp.S):
        summ = 0
        for a in range(mdp.A):
            index = np.ravel_multi_index((s,a), (mdp.S, mdp.A))
            summ += d_sa[index]

        use_first_action = summ == 0
        has_assigned_action = False
        for a, allowed in enumerate(mdp.get_allowed_actions(s)):
            if use_first_action and allowed and not has_assigned_action:
                policy.append(1.0)
                has_assigned_action = True
            elif use_first_action:
                policy.append(0.0)
            else:
                index = np.ravel_multi_index((s,a), (mdp.S, mdp.A))
                action_weight = d_sa[index]/summ
                if not allowed and not action_weight == 0:
                    raise Exception('Non-allowed action has positive weight!')
                policy.append(action_weight)
    return np.array(policy, dtype=np.float32)