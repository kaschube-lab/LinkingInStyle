import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    origin_class = 254
    target_class = 151
    gen_seed = 0

    eps = 1e-4
    lr = 0.5
    lambda_ = 0.6
    manualseed = 1
    input_dir = f'../data/counterfactual_opt/loss_v2_shiftInitSeed={manualseed}_eps={eps}_lambda={lambda_}_lr={lr}/'

    all_r = np.load(input_dir + f'class{origin_class}_seed{gen_seed}_r.npy', allow_pickle=True)
    all_rshifted = np.load(input_dir + f'class{origin_class}_seed{gen_seed}_target{target_class}_rshifted.npy', allow_pickle=True)

    relevant_units = []

    for i in range(len(all_r)):
        shift = all_r[i] - all_rshifted[i]
        sorted_units = np.argsort(abs(shift))[::-1]
        print(sorted_units[:10])
        print(shift[sorted_units[:10]])
        relevant_units.append(sorted_units[:10])

    # plot histogram
    relevant_units = np.array(relevant_units)
    relevant_units = relevant_units.flatten()
    fig, ax = plt.subplots(figsize=(10*0.39, 5*0.39))
    # count number of times each unit appears and plot as barplot
    sns.countplot(x=relevant_units, ax=ax, color='gray')
    ax.set_xlabel('Unit')
    ax.set_ylabel('Count')
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.savefig(input_dir + f'relevant_units_class{origin_class}_target{target_class}_seed{gen_seed}.png', dpi=300, bbox_inches='tight')