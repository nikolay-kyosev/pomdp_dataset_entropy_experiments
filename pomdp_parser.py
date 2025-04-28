import numpy as np
import pomdp_definitions as pomdps
from lark import Lark, Tree, Token

def parse_preamble(preamble: Tree):
    states = 0
    actions = 0
    observations = 0
    for param_type in preamble.children:
        param = param_type.children[0]
        if param.data == 'value_param':
            print(f'Info: Value param found, skipping.')
        elif param.data == 'discount_param':
            print(f'Info: Discount param found, skipping.')
        else:
            smth_tail = param.children[2].children[0]
            if isinstance(smth_tail, Token):
                if param.data == 'state_param':
                    states = int(smth_tail.value)
                elif param.data == 'action_param':
                    actions = int(smth_tail.value)
                elif param.data == 'obs_param':
                    observations = int(smth_tail.value)
                else:
                    raise Exception('Error: Unexpected param')
            else:
                print(f'Error: Identity list not supported')
    if states == 0 or actions == 0 or observations == 0:
        raise Exception('Error: States actions or observations are zero.')
    return states, actions, observations

def parse_start_state(pomdp: pomdps.Pomdp, start_state: Tree):
    #We assume its a probability list!
    u_matrix = start_state.children[2]
    data = parse_u_matrix(u_matrix)
    if not len(data) == len(pomdp.d0):
        raise Exception('Error: Invalid amount of start states.')
    pomdp.d0 = data

def parse_state_action_observation(identifier: Tree):
    id_ = identifier.children[0].value
    if id_ == '*':
        id_ = slice(None)
    else:
        id_ = int(id_)
    return id_

def parse_u_matrix(u_matrix: Tree, prob_check=True):
    prob_matrix = u_matrix.children[0]
    if not prob_matrix.data == 'prob_matrix':
        raise Exception('Error: Other types of u_matrix are not supported.')
    probs = []
    for prob in prob_matrix.children:
        probs.append(float(prob.children[0].value))
    result = np.array(probs, dtype=np.float32)
    # Ensure it sums to one
    if any(result < 0):
        raise Exception('Error: Negative probabilities in initial belief.')
    summ = np.sum(result)
    if prob_check and not np.isclose(summ, 1, atol=0.01):
        raise Exception('Invalid probability matrix')
    if prob_check:
        result = result / np.sum(result)
    return result

def parse_param_list(pomdp: pomdps.Pomdp, param_list: Tree):
    for param_spec_ in param_list.children:
        param_spec = param_spec_.children[0]
        if param_spec.data == 'trans_prob_spec' or param_spec.data == 'obs_prob_spec':
            # Now parse one of the four subtypes...
            tail = param_spec.children[2]
            if len(tail.children) == 6:
                paction = tail.children[0]
                state = tail.children[2]
                id_ = tail.children[4]
                prob = tail.children[5]
                
                a = parse_state_action_observation(paction)
                s = parse_state_action_observation(state)
                id_ = parse_state_action_observation(id_)                
                p = float(prob.children[0].value)
                if param_spec.data == 'trans_prob_spec':
                    pomdp.P[s, a, id_] = p
                else:
                    pomdp.O[a, s, id_] = p
            elif len(tail.children) == 4:
                paction = tail.children[0]
                state = tail.children[2]
                u_matrix = tail.children[3]
                
                a = parse_state_action_observation(paction)
                s = parse_state_action_observation(state)
                data = parse_u_matrix(u_matrix)
                if param_spec.data == 'trans_prob_spec':
                    pomdp.P[s, a] = data
                else:
                    pomdp.O[a, s] = data

            elif len(tail.children) == 2:
                paction = tail.children[0]
                u_matrix = tail.children[1]

                a = parse_state_action_observation(paction)
                data = parse_u_matrix(u_matrix, prob_check=False)
                if param_spec.data == 'trans_prob_spec':
                    data = np.reshape(data, (pomdp.S, pomdp.S,))
                    pomdp.P = np.transpose(pomdp.P, (1, 0, 2))
                    pomdp.P[a] = data
                    pomdp.P = np.transpose(pomdp.P, (1, 0, 2))
                else:
                    data = np.reshape(data, (pomdp.S, pomdp.Z,))
                    pomdp.O[a] = data
            else:
                print(f'Info: unparsed transition/observation spec')
    # Fix rounding errors
    for a in range(pomdp.A):
        for s in range(pomdp.S):
            summ = np.sum(pomdp.P[s, a])
            if not np.isclose(summ, 1, atol=pomdps.POMDP_DEFINITIONS_PRECISION) and np.isclose(summ, 1, atol=0.01):
                pomdp.P[s, a] = pomdp.P[s, a] / summ
                if not np.isclose(summ, 1, atol=pomdps.POMDP_DEFINITIONS_PRECISION):
                    raise Exception('Wtf!!!')
            summ = np.sum(pomdp.O[a, s])
            # Fix rounding
            if not np.isclose(summ, 1, atol=pomdps.POMDP_DEFINITIONS_PRECISION) and np.isclose(summ, 1, atol=0.01):
                # Normalize
                pomdp.O[a, s] = pomdp.O[a, s] / summ
                summ = np.sum(pomdp.O[a, s])
                if not np.isclose(summ, 1, atol=pomdps.POMDP_DEFINITIONS_PRECISION):
                    raise Exception(f'Probability sum rounding error: {summ} for {a} {s}')
            elif not np.isclose(summ, 1, atol=pomdps.POMDP_DEFINITIONS_PRECISION) and not np.isclose(summ, 1, atol=0.01):
                raise Exception(f'Error: {pomdp.O[a, s]} is invalid. s={s} a={a}')

    pass
    

def parse_pomdp_file(path: str):
    grammar = open('./pomdp_file.GRAMMAR')
    f = open(path)
    txt = f.read()
    parser = Lark(grammar, start='pomdp_file')
    parsed = parser.parse(txt);
    # print(parsed.pretty())
    preamble = parsed.children[0]
    start_state = parsed.children[1]
    param_list = parsed.children[2]

    
    #Start with preamble, ignore discount and value parameters.
    S, A, Z = parse_preamble(preamble)
    pomdp = pomdps.Pomdp(S, A, Z)
    parse_start_state(pomdp, start_state)
    parse_param_list(pomdp, param_list)
    print(f'Is well formed: {pomdp.is_well_formed()}')
    if np.isnan(pomdp.P).any():
        raise Exception('Error: Nan value in the transition tensor!')
    if np.isnan(pomdp.O).any():
        raise Exception('Error: NaN value in the observation tensor!')
    if np.isnan(pomdp.d0).any():
        raise Exception('Eror: NaN value in the initial belief!')
    return pomdp




if __name__=='__main__':
    parse_pomdp_file('./tiger-grid.POMDP')
    pass