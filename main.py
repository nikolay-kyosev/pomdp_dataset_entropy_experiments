import numpy as np
import pomdp_experiment
import mdp_utilities
import pomdp_parser
import scipy
from mdp_maxent_common import EntropyDimension as Edim
import pomdp_maxent
import mdp_finite_maxent_solver
import mdp_experiment
import plots
import time
import pomdp_dataset_entropy
import pomdp_definitions

from tabulate import tabulate

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_runtime_plot_experiment(pomdp, name, T, n_tranjectories):
    data = []
    data_approx = []
    data_belief = []
    data_qmdp = []
    for i in range(1, T+1):
        start = time.time()
        policy = pomdp_dataset_entropy.make_dataset_based_policy(pomdp, i, n_tranjectories, Edim.STATE)
        data.append(time.time() - start)
        start = time.time()
        policy = pomdp_dataset_entropy.make_dataset_based_policy(pomdp, i, n_tranjectories, Edim.STATE, use_approx=True)
        data_approx.append(time.time() - start)
        #Belief
        start = time.time()
        pomdp_maxent.solve_problem(pomdp, i, Edim.STATE)
        duration = time.time() - start
        data_belief.append(duration)
        #QMDP
        start = time.time()
        mdp_finite_maxent_solver.solve_problem(pomdp, i, Edim.STATE)
        duration = time.time() - start
        data_qmdp.append(duration)
    df = pd.DataFrame({
        'Timesteps': list(range(len(data))),
        'Exact': data,
        'Forward-Backward approx.': data_approx,
        'Belief approx.': data_belief,
        'QMDP approx.': data_qmdp
    })
    df_melted = df.melt(id_vars='Timesteps', var_name='Algorithm', value_name='Runtime (seconds)')
    plt.figure(figsize=(8, 6))
    sns.lineplot(x='Timesteps', y='Runtime (seconds)', hue='Algorithm', data=df_melted)
    name = f'{name} Runtime Experiment'
    plt.title(name)
    filename = name.lower()
    filename = filename.replace(' ', '_')
    filename += f'_T={T}_D={n_tranjectories}'
    plt.savefig(f'./plots/runtime/{filename}.png', dpi=300, bbox_inches='tight')

def make_runtime_table_experiments(target_time=60.0):
    pomdp = mdp_utilities.generate_river_swim_partially_observable()
    test_exact = True
    exact = 0
    test_fb_approx = True
    fb = 0
    test_belief_approx = True
    belief = 0
    test_qmdp = True
    qmdp = 0

    i = 1

    experiment_start = time.time()
    while True:
        if test_exact:
            start = time.time()
            pomdp_dataset_entropy.make_dataset_based_policy(pomdp, i, 1, Edim.STATE)
            duration = time.time() - start
            exact = i
            if duration > target_time:
                test_exact = False
                exact = i
        if test_fb_approx:
            start = time.time()
            pomdp_dataset_entropy.make_dataset_based_policy(pomdp, i, 1, Edim.STATE, use_approx=True)
            duration = time.time() - start
            fb = i
            if duration > target_time:
                test_fb_approx = False
                fb = i

        if test_belief_approx:
            try:
                start = time.time()
                pomdp_maxent.solve_problem(pomdp, i, Edim.STATE)
                duration = time.time() - start
                belief = i
            except:
                test_belief_approx = False
                belief = i
            if duration > target_time:
                test_belief_approx = False
                belief = i
        if test_qmdp:
            try:
                start = time.time()
                mdp_finite_maxent_solver.solve_problem(pomdp, i, Edim.STATE)
                duration = time.time() - start
                qmdp = i
            except:
                test_qmdp = False
                qmdp = i
            if duration > target_time:
                test_qmdp = False
                qmdp = i
        i = i + 1
        t = time.time()
        if not (test_belief_approx or test_exact or test_fb_approx or test_qmdp):
            break
        if t - experiment_start > 600:
            print(f'Experiment took over 10 minutes, trunctating it to avoid long runtime.')
            break

    data = [['Exact', exact], ['Forward-Backward approx.', fb], ['Belief approx.', belief], ['QMDP approx.', qmdp]]
    headers = ['Algorithm', 'Number of timesteps']
    print(tabulate(data, headers, tablefmt='grid'))
    # return exact, fb, belief, qmdp
        



def gather_large_env_data(pomdp, name, T, n_trajectories, ed: Edim, use_qmdp_only=False):
    args = [pomdp, T, n_trajectories]
    entropy_comparison_entropies = []

    data = []

    exp_s, exp_sa, exp_sas, q_s, q_sa, q_sas = pomdp_experiment.conduct_experiment(*args, ed, qmdp_approach=True)
    if ed == Edim.STATE:
        data.append(f'{exp_s:.2f}')
    elif ed == Edim.STATE_ACTION:
        data.append(f'{exp_sa:.2f}')
    elif ed == Edim.TRANSITION:
        data.append(f'{exp_sas:.2f}')

    # entropy_comparison_entropies.append([q_s[-1], q_sa[-1], q_sas[-1]])
    # exp_s_, exp_sa_, exp_sas_, q_s_, q_sa_, q_sas_ = mdp_experiment.conduct_experiment(*args, ed)
    # print(f'Fully_observable_expectation: {[exp_s_, exp_sa_, exp_sas_]}\nActual: {q_s_[-1], q_sa_[-1], q_sas_[-1]}')
    exp = (exp_s, exp_sa, exp_sas)
    q = (q_s[-1], q_sa[-1], q_sas[-1])

    if not use_qmdp_only:
        expected_d, b_s, b_sa, b_sas = pomdp_experiment.conduct_experiment(*args, ed, make_belief_mdp_graph=False)
        if ed == Edim.STATE:
            data.append(f'{b_s[-1]:.2f}')
            data.append(f'{q_s[-1]:.2f}')
        elif ed == Edim.STATE_ACTION:
            data.append(f'{b_sa[-1]:.2f}')
            data.append(f'{q_sa[-1]:.2f}')
        elif ed == Edim.TRANSITION:
            data.append(f'{b_sas[-1]:.2f}')
            data.append(f'{q_sas[-1]:.2f}')

        belief = (b_s[-1], b_sa[-1], b_sas[-1])
        plots.save_barplot(exp, belief, q, name, T, ed)
        baseline = None
        if ed == Edim.STATE:
            baseline = [exp_s] * len(q_s)
            plots.save_plot(baseline, b_s, q_s, name, T, ed)
        elif ed == Edim.STATE_ACTION:
            baseline = [exp_sa] * len(q_sa)
            plots.save_plot(baseline, b_sa, q_sa, name, T, ed)
        else:
            baseline = [exp_sas] * len(q_sas)
            plots.save_plot(baseline, b_sas, q_sas, name, T, ed)
    else:
        data.append('-')
        if ed == Edim.STATE:
            data.append(f'{q_s[-1]:.2f}')
        elif ed == Edim.STATE_ACTION:
            data.append(f'{q_sa[-1]:.2f}')
        elif ed == Edim.TRANSITION:
            data.append(f'{q_sas[-1]:.2f}')

        if ed == Edim.STATE:
            baseline = [exp_s] * len(q_s)
            plots.save_plot_qmdp_baseline(baseline, q_s, name, T, ed)
        elif ed == Edim.STATE_ACTION:
            baseline = [exp_sa] * len(q_sa)
            plots.save_plot_qmdp_baseline(baseline, q_sa, name, T, ed)
        else:
            baseline = [exp_sas] * len(q_sas)
            plots.save_plot_qmdp_baseline(baseline, q_sas, name, T, ed)
        # plots.save_barplot(exp, None, q, name, T, ed)
    # plots.save_barplot_entropy_comparison(name, T, n_trajectories, *entropy_comparison_entropies)
    return data, q
    
def gather_exact_solution_data(pomdp, name, T, n_trajectories, ed: Edim, n_sample_trajectories=10000):
    data = []

    dataset_decision_f = lambda dataset : pomdp_dataset_entropy.uniform_dataset_history(T, n_trajectories, dataset)
    #Exact
    # print(f'{ed.name}---------------------')
    policy = pomdp_dataset_entropy.make_dataset_based_policy(pomdp, T, n_trajectories, ed)
    exact_list = []
    for i in range(n_sample_trajectories):
        exact = pomdp_experiment.sample_dataset_based_policy(pomdp, policy, dataset_decision_f)
        exact_list.append(exact[ed.value])
    matrix = np.array(exact_list)
    means = np.sum(matrix, axis=0)
    means = means/n_sample_trajectories
    variance = np.square(matrix - means)
    variance = np.sum(variance, axis=0) /  (n_sample_trajectories - 1)
    data.append(f'{means:.2f} ({variance:.2f})')

    policy = pomdp_dataset_entropy.make_dataset_based_policy(pomdp, T, n_trajectories, ed, use_approx=True)
    approximate_list = []
    for i in range(n_sample_trajectories):
        approximate = pomdp_experiment.sample_dataset_based_policy(pomdp, policy, dataset_decision_f)
        approximate_list.append(approximate[ed.value])
    matrix = np.array(approximate_list)
    means = np.sum(matrix, axis=0)
    means = means/n_sample_trajectories
    variance = np.square(matrix - means)
    variance = np.sum(variance, axis=0) /  (n_sample_trajectories - 1)
    data.append(f'{means:.2f} ({variance:.2f})')
    # print(f'F.b. solution means: {means}')
    # print(f'F.b. solution variance: {variance}')
    #Belief
    policy, _ = policies, expected_d = pomdp_maxent.solve_problem(pomdp, T, ed)
    belief_list = []
    for i in range(n_sample_trajectories):
        b_s, b_sa, b_sas = pomdp_experiment.sample_trajectories_randomized(pomdp, T, n_trajectories, policy, qmdp_approach=False)
        belief = (b_s[-1], b_sa[-1], b_sas[-1])
        belief_list.append(belief[ed.value])
    matrix = np.array(belief_list)
    means = np.sum(matrix, axis=0)
    means = means/n_sample_trajectories
    variance = np.square(matrix - means)
    variance = np.sum(variance, axis=0) /  (n_sample_trajectories - 1)
    data.append(f'{means:.2f} ({variance:.2f})')
    # print(f'Belief means: {means}')
    # print(f'Belief variance: {variance}')
    #QMDP
    policy, _, _, _ = mdp_finite_maxent_solver.solve_problem(pomdp, T, ed)
    qmdp_list = []
    for i in range(n_sample_trajectories):
        qmdp_s, qmdp_sa, qmdp_sas = pomdp_experiment.sample_trajectories_randomized(pomdp, T, n_trajectories, policy, qmdp_approach=True)
        qmdp = (qmdp_s[-1], qmdp_sa[-1], qmdp_sas[-1])
        qmdp_list.append(qmdp[ed.value])
    matrix = np.array(qmdp_list)
    means = np.sum(matrix, axis=0)
    means = means/n_sample_trajectories
    variance = np.square(matrix - means)
    variance = np.sum(variance, axis=0) /  (n_sample_trajectories - 1)
    data.append(f'{means:.2f} ({variance:.2f})')
    # print(f'QMDP means: {means}')
    # print(f'QMDP variance: {variance}')
    # print('-----------------------')
    #Make the plot
    return data
    # plots.make_dataset_entropy_comparison(f'{name} {ed.name.capitalize()} T={T}, N_Trajectories = {n_trajectories}', exact, approximate, belief, qmdp, ed )
def make_runtime_experiments(envs, random_seed=42):
    print(f'RUNTIME_EXPERIMENTS---------------------------------')
    np.random.seed(random_seed)
    make_runtime_table_experiments()
    for env in envs:
        np.random.seed(random_seed)
        make_runtime_plot_experiment(env['pomdp'], env['name'], env['T'], 1)
    print(f'----------------------------------------------------')
    
def make_exact_experiments(envs, T=10, random_seed=42):
    print(f'EXACT ALGORITHM COMPARISON--------------------------')
    for ed in Edim:
        print(f'{ed.name}')
        data = []
        for env in envs:
            np.random.seed(random_seed)
            row = gather_exact_solution_data(env['pomdp'], env['name'], T, 1, ed)
            row = [env['name']] + row
            data.append(row)
        headers = ['Environment', 'Exact', 'F.B. approx.', 'Belief approx.', 'QMDP approx.']
        print(tabulate(data, headers, tablefmt='grid'))
    print(f'---------------------------------------------------')

def make_large_experiments(envs, I=10000, random_seed=42):
    print(f'LARGE DATASET ALGORITHM COMPARISON--------------------------')
    entropy_comp_dataset = {}
    for env in envs:
        entropy_comp_dataset[env['name']] = []
    for ed in Edim:
        print(f'{ed.name}')
        data = []
        for env in envs:
            np.random.seed(random_seed)
            row, entropy_comp_data = gather_large_env_data(env['pomdp'], env['name'], env['T'], I, ed, use_qmdp_only=env['large'])
            entropy_comp_dataset[env['name']].append(entropy_comp_data)
            row = [env['name']] + [env['T']] + row
            data.append(row)
        headers = ['Environment', 'T', 'Upper bound', 'Belief approx.', 'QMDP approx.']
        print(tabulate(data, headers, tablefmt='grid'))
    
    for env in envs:
        plots.save_barplot_entropy_comparison(env['name'], env['T'], I, *entropy_comp_dataset[env['name']])
    print(f'------------------------------------------------------------')

def get_envs():
    print('LOADING ENVS.......')
    exact_exp_envs = []
    all_envs = []

    env = {}
    env['pomdp'] = mdp_utilities.generate_chain_mdp_unobservable(3)
    env['name'] = '3State U. O.'
    env['large'] = False #Indicates that we can use the belief-MDP solution in large experiemnts
    env['T'] = 10
    exact_exp_envs.append(env)
    all_envs.append(env)

    env = {}
    env['pomdp'] = mdp_utilities.generate_river_swim_partially_observable()
    env['name'] = 'River Swim P. O.'
    env['large'] = False
    env['T'] = 10
    exact_exp_envs.append(env)
    all_envs.append(env)

    #Large dataseet environments
    env = {}
    env['pomdp'] = mdp_utilities.generate_river_swim_partially_observable_multiple_end_component()
    env['name'] = 'River Swim Rest P.O.'
    env['large'] = True
    env['T'] = 10
    all_envs.append(env)

    env = {}
    env['pomdp'] = pomdp_parser.parse_pomdp_file('./envs/cheese.95.POMDP')
    env['name'] = 'Cheese Maze'
    env['large'] = False
    env['T'] = 25
    all_envs.append(env)

    env = {}
    env['pomdp'] = pomdp_parser.parse_pomdp_file('./envs/aloha.10.POMDP')
    env['name'] = 'Aloha 10'
    env['large'] = True
    env['T'] = 50
    all_envs.append(env)

    env = {}
    env['pomdp'] = pomdp_parser.parse_pomdp_file('./envs/tiger-grid.POMDP')
    env['name'] = 'Tiger Grid 36-state'
    env['large'] = True
    env['T'] = 50
    all_envs.append(env)

    print(f'ENVS LOADED!')

    return exact_exp_envs, all_envs



if __name__ == '__main__':
    exact_envs, all_envs = get_envs()

    make_runtime_experiments([exact_envs[1]])
    make_exact_experiments(exact_envs)
    make_large_experiments(all_envs)
