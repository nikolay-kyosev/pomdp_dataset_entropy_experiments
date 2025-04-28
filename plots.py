
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mdp_maxent_common import EntropyDimension as Edim

def modify_name(name, T, ed: Edim):
    if ed == Edim.STATE:
        name += f' - State Entropy, T={T}'
    elif ed == Edim.STATE_ACTION:
        name += f' - State-action Entropy, T={T}'
    else:
        name += f' - Transition Entropy, T={T}'
    return name

def save_plot_as_file(plt, name, T, ed: Edim = None):
    filename = name.lower()
    filename = filename.replace(' ', '_')
    if ed is not None:
        if ed == Edim.STATE:
            filename += '_state'
        elif ed == Edim.STATE_ACTION:
            filename += '_state_action'
        else:
            filename += '_transition'
    filename = './plots/' + filename + '.png'
    # print(filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def save_plot(baseline, entropies, entropies_qmdp, name, T,  ed: Edim):
    df = pd.DataFrame({
        'Trajectories': list(range(len(entropies))),
        'Belief-based policy': entropies,
        'QMDP-based policy': entropies_qmdp,
        'Expected entropy under full observability': baseline,
    })
    # Melt the DataFrame to make it suitable for seaborn
    df_melted = df.melt(id_vars='Trajectories', var_name='Type', value_name='Value')

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.lineplot(x='Trajectories', y='Value', hue='Type', data=df_melted)
    name_ = modify_name(name, T, ed)
    plt.title(name_)
    plt.xlabel("Trajectories sampled")
    plt.ylabel("Entropy")
    plt.legend(title="Legend")
    save_plot_as_file(plt, '/large_datasets/entropy_growth/'+name, T, ed)
    plt.close()

def save_barplot(expected, belief, qmdp, name, T, ed: Edim):
    e_s, e_sa, e_sas = expected
    if not belief is None:
        b_s, b_sa, b_sas = belief
    q_s, q_sa, q_sas = qmdp
    data = None
    if not belief is None:
        policy_data = (["Expected", "Belief", "QMDP"]*3)
        type_data = (["State"]*3 + ['State Action']*3 + ['Transition']*3)
        data = {
                "Policy": policy_data,
                "Type": type_data,
                "Entropy": ([e_s, b_s, q_s, e_sa, b_sa, q_sa, e_sas, b_sas, q_sas])
            }
        df = pd.DataFrame(data)
    else:
        policy_data = (["Expected", "QMDP"]*3)
        type_data = (["State"]*2 + ['State Action']*2 + ['Transition']*2)
        data = {
                "Policy": policy_data,
                "Type": type_data,
                "Entropy": ([e_s, q_s, e_sa, q_sa, e_sas, q_sas])
            }
        df = pd.DataFrame(data)
    # Create the bar plot
    ax = sns.barplot(
        data=df,
        x="Type",
        y="Entropy",
        hue="Policy",
        # palette={"Belief": "blue", "QMDP": "purple", "Expected": "green" }
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

    # Add a title and labels
    plt.title(modify_name(name, T, ed))
    plt.xlabel("Type")
    plt.ylabel("Entropy")
    save_plot_as_file(plt, 'barplot_' + name, T, ed)
    plt.close()

def save_barplot_entropy_comparison(env_name, T, N, s_results, sa_results, sas_results):
    objective_data = (["State", "State Action", "Transition"]*3)
    entropies = list(zip(s_results, sa_results, sas_results))
    entropies = [e for pair in entropies for e in pair]
    entropy_type_data = (["State"]*3 + ['State Action']*3 + ['Transition']*3)
    data = {
            "Objective": objective_data,
            "Entropy Type": entropy_type_data,
            "Entropy": entropies
        }
    df = pd.DataFrame(data)

    ax = sns.barplot(
        data=df,
        x="Entropy Type",
        y="Entropy",
        hue="Objective",
        # palette={"Belief": "blue", "QMDP": "purple", "Expected": "green" }
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

    # Add a title and labels
    plt.title(env_name + f' Entropy Comparison T={T} N={N}')
    plt.xlabel("Entropy Type")
    plt.ylabel("Entropy")
    name = env_name.lower()
    name = name.replace(' ', '_')
    save_plot_as_file(plt, '/entropy_comparison_barplots/barplot_entropy_comparison_' + name, T)
    plt.close()

def save_plot_qmdp_baseline(baseline, entropies_qmdp, name, T,  ed: Edim):
    df = pd.DataFrame({
        'Trajectories': list(range(len(baseline))),
        'QMDP policy': entropies_qmdp,
        'Expected entropy under full observability': baseline,
    })
    # Melt the DataFrame to make it suitable for seaborn
    df_melted = df.melt(id_vars='Trajectories', var_name='Type', value_name='Value')

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.lineplot(x='Trajectories', y='Value', hue='Type', data=df_melted)
    name_ = modify_name(name, T, ed)
    plt.title(name_)
    plt.xlabel("Trajectories sampled")
    plt.ylabel("Entropy")
    plt.legend(title="Legend")
    save_plot_as_file(plt, '/large_datasets/entropy_growth/' + name, T, ed)
    plt.close()

def make_dataset_entropy_comparison(name, exact, approximate, belief, qmdp, ed: Edim):
    entropies = list(zip(exact, approximate, belief, qmdp))
    entropies = [e for pair in entropies for e in pair]
    policy_data = (["Exact", "Forward-Backward approx.", "Belief approx.", "QMDP approx."]*3)
    type_data = (["State"]*4 + ['State Action']*4 + ['Transition']*4)
    data = {
            "Policy": policy_data,
            "Type": type_data,
            "Entropy": entropies
        }
    
    df = pd.DataFrame(data)

    # Create the bar plot
    ax = sns.barplot(
        data=df,
        x="Type",
        y="Entropy",
        hue="Policy",
        # palette={"Belief": "blue", "QMDP": "purple", "Expected": "green" }
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

    # Add a title and labels
    plt.title(name)
    plt.legend(title='Algorithm')
    plt.xlabel("Type")
    plt.ylabel("Entropy")
    filename = name.lower()
    filename = filename.replace(' ', '_')
    filename = './plots/algorithm_comparison/exact/' + filename + '.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()