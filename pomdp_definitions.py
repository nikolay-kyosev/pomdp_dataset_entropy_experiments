import numpy as np
import graphviz

POMDP_DEFINITIONS_PRECISION = 1e-6

class Mdp():
    """Simple MDP class."""
    def __init__(self, n_states, n_actions):
        self.S = n_states
        self.A = n_actions
        self.d0 = np.zeros((n_states,), dtype=np.float32)
        self.P = np.zeros((n_states, n_actions, n_states), dtype=np.float32)
        self.R = np.zeros((n_states, n_actions, n_states), dtype=np.float32)

    def get_allowed_actions(self, s):
        actions = np.zeros((self.A,), dtype=np.bool_)
        for a in range(self.A):
            if np.isclose(np.sum(self.P[s,a]), 1, atol=POMDP_DEFINITIONS_PRECISION):
                actions[a] = True
            else:
                actions[a] = False
        return actions
    
    def is_well_formed(self) -> bool:
        for s in range(self.S):
            actions = self.get_allowed_actions(s)
            if not any(actions):
                return False
            for a in range(self.A):
                if not actions[a] and not sum(self.P[s,a]) == 0:
                    return False
        if not np.isclose(np.sum(self.d0), 1, atol=POMDP_DEFINITIONS_PRECISION):
            return False
        return True

class Pomdp(Mdp):
    """Simple POMDP class."""
    def __init__(self, n_states, n_actions, n_observations):
        super().__init__(n_states, n_actions)
        self.Z = n_observations
        self.O = np.zeros((n_actions, n_states, n_observations), dtype=np.float32)

    def is_well_formed(self) -> bool:
        if not super().is_well_formed():
            return False
        
        #Check that the observation function is well formed.
        for a in range(self.A):
            for s in range(self.S):
                if not np.isclose(np.sum(self.O[a, s]), 1, atol=POMDP_DEFINITIONS_PRECISION):
                    return False
        return True

class BeliefMdp():
    def __init__(self, pomdp: Pomdp, T: int):
        #Eech element of the list corresponds to a level
        self.pomdp = pomdp
        self.S = [{tuple(pomdp.d0)}]
        self.A = pomdp.A
        for t in range(1,T):
            belief_set_t = set({})
            for belief in self.S[-1]:
                for a in range(pomdp.A):
                    for z in range(pomdp.Z):
                        pr_z, new_belief, _ = belief_update(pomdp, np.array(belief, dtype=np.float32), a, z)
                        if pr_z > 0:
                            belief_set_t.add(tuple(new_belief))
            self.S.append(belief_set_t)
        self.belief_list = list(set().union(*self.S))
        self.belief_list = list(map(lambda x : np.array(x, dtype=np.float32), self.belief_list))
        self.belief_count = len(self.belief_list)

    def make_graphviz_graph(self, policy=None):
        node_id = 0
        dot = graphviz.Digraph()
        #First, make the belief nodes
        for i, t in enumerate(self.S):
            for b in t:
                    label = np.array2string(np.array(b, dtype=np.float32), precision=2, separator=', ')
                    dot.node(f'{i}_{b}', label=label)

        #Secondly, make the action nodes and edges
        for i, t in enumerate(self.S):
            if i == len(self.S) - 1:
                continue
            for b in t:
                for a in range(self.pomdp.A):
                    child_beliefs = set({})
                    for z in range(self.pomdp.Z):
                        pr_z, b_, _ = belief_update(self.pomdp, np.array(b, dtype=np.float32), a, z)
                        if pr_z > 0:
                            child_beliefs.add((pr_z, tuple(b_)))
                    if len(child_beliefs) == 0:
                        continue
                    dot.node(f'{i}_{b}_{a}', label=f'a_{a}', shape='point')
                    label = f'a{a}'
                    if policy is not None:
                        p = 0
                        if b in policy[i]:
                            p = np.array2string(policy[i][b][a], precision=2)
                        label+= f' p={p}'
                    dot.edge(f'{i}_{b}', f'{i}_{b}_{a}', label=label)
                    for p, b_ in child_beliefs:
                        p = np.array2string(p, precision=2)
                        dot.edge(f'{i}_{b}_{a}', f'{i+1}_{b_}', label=f'p={p}')
        return dot


def belief_update(pomdp: Pomdp, belief: np.ndarray, a: int, z: int):
    new_belief = np.zeros((pomdp.S), dtype=np.float32)
    transition_belief = np.zeros((pomdp.S, pomdp.S), dtype=np.float32)
    
    reshaped_P = pomdp.P.transpose(1, 0, 2)
    reshaped_O = pomdp.O.transpose(0, 2, 1)

    transition_belief = np.diag(belief) @ reshaped_P[a] @ np.diag(reshaped_O[a, z])
    new_belief = np.sum(transition_belief, axis=0)
    pr_z = np.sum(new_belief)
    if pr_z == 0:
        return 0, new_belief, transition_belief
    elif pr_z == np.nan:
        raise Exception('Error: Probability of an observation is NaN!')
    
    new_belief = new_belief/pr_z
    transition_belief = transition_belief/pr_z

    return pr_z, new_belief, transition_belief

