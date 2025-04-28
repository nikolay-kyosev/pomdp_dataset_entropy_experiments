import numpy as np
import pomdp_definitions as pomdps

class Env:
    def __init__(self, pomdp: pomdps.Pomdp = None , mdp: pomdps.Mdp = None):
        self.mdp = mdp
        self.pomdp = pomdp
        self.s = 0
        self.belief = None
        if self.pomdp is not None:
            self.belief = self.pomdp.d0

    def reset(self):
        if self.mdp is not None:
            self.s = np.random.choice(self.mdp.d0.shape[0], p=self.mdp.d0)
            return self.s
        if self.pomdp is not None:
            self.s = np.random.choice(self.pomdp.d0.shape[0], p=self.pomdp.d0)
            self.belief = self.pomdp.d0
            return self.belief, self.s
        raise Exception('Error: Neither pomdp or mdp was initialized')

    def step(self, action: int):
        a = action
        if self.mdp is not None:
            self.s = np.random.choice(self.mdp.P[self.s, a].shape[0], p=self.mdp.P[self.s,a])
            return self.s
        if self.pomdp is not None:
            self.s = np.random.choice(self.pomdp.P[self.s, a].shape[0], p=self.pomdp.P[self.s,a])
            z = np.random.choice(self.pomdp.O[a, self.s].shape[0], p = self.pomdp.O[a, self.s])
            pr_z, new_belief, _ = pomdps.belief_update(self.pomdp, self.belief, a, z)
            self.belief = new_belief
            return self.belief, z, self.s