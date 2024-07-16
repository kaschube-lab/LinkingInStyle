import os
import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch

from generate_data import load_encoder, val_transform

all_classes = [151, 178, 197, 232, 254]
pal = sns.color_palette(palette='ch:s=-.2,r=.6', n_colors=len(all_classes))
cmap_classes = {c:pal[i] for i, c in enumerate(all_classes)}

def load_data():
    data_r = []
    data_w = []
    labels = []

    # load data
    for c in [232, 151]: #[151, 178, 197, 232, 254]:
        data_c = np.load(f'data/data_noNoise_rnet50_5000N_{c}.npz', allow_pickle=True)
        data_r += data_c['feats'].tolist()
        data_w += data_c['ws'].tolist()
        labels += [c]*len(data_c['feats'])

    labels = np.array(labels)
    data_r = np.array(data_r)
    data_w = np.array(data_w)

    return data_r, data_w, labels

def build_pc_space(data):
    pca_data = PCA().fit(data)

    return pca_data

def subsample_data(data, labels, pca, n_samples=1000):
    mean = np.mean(data, axis=0)
    np.random.seed(0)
    idx_subsample = np.random.choice(np.arange(len(data)), size=n_samples, replace=False)
    subsampled = data[idx_subsample]
    subsampled_pca = pca.transform(subsampled-mean)
    subsample_labels = labels[idx_subsample]

    return subsampled_pca, subsample_labels

def get_pred_class_img(encoder, img):
    img = val_transform(img)
    pred = encoder(img[None, ...].to(device))
    pred = torch.softmax(pred, dim=1)
    pred_class = torch.argmax(pred).item()

    return pred_class

def plot_counterfactuals(example_i, origin_c, target_c, seed):
    # load data from counterfactual examples
    example_i = 0
    origin_c = 232
    target_c = 151
    seed_nb = 0

    eps = 1e-4
    lr = 0.5
    lambda_ = 0.6
    manualseed = 1
    input_dir = f'../data/counterfactual_opt/loss_v2_shiftInitSeed={manualseed}_eps={eps}_lambda={lambda_}_lr={lr}/'

    samples_232 = []
    rtype_labels = []

    rtypes = ['r', 'rshifted', 'rcycled']
    for rtype in rtypes:
        fp = input_dir + f'class{origin_c}_seed{seed_nb}_'
        fp += f'target{target_c}_{rtype}.npy' if rtype != 'r' else f'{rtype}.npy'
        data = np.load(fp, allow_pickle=True)
        data -= mean_r
        samples_232 += data.tolist()
        rtype_labels += [rtype]*len(data)

    samples_232 = np.array(samples_232)
    rtype_labels = np.array(rtype_labels)
    samples_pca = pca_r.transform(samples_232)

    # plot data on the PC space
    fig, ax = plt.subplots(1, 1, figsize=(3*0.39, 3*0.39))
    sns.scatterplot(x=subsampled_r_pca[:, 0], y=subsampled_r_pca[:, 1], 
                    hue=subsampled_labels_r, ax=ax)
    sns.scatterplot(x=samples_pca[:, 0], y=samples_pca[:, 1], 
                    hue=rtype_labels, ax=ax, palette='tab10', marker='x')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.savefig(input_dir + f'PC_Rspace_class{origin_c}_target{target_c}_seed{seed_nb}.png', dpi=300)
    plt.close(fig)
    # plt.show()

    samples_232_w = []
    wtype_labels = []

    wtypes = ['w', 'wcycled']
    for wtype in wtypes:
        if wtype != 'w':
            fp = input_dir + f'class{origin_c}_seed{seed_nb}_target{target_c}_{wtype}.npy'
        else:
            fp = input_dir + '../../samples/' + f'class{origin_c}_seed{seed_nb}_{wtype}.npy'
        data = np.load(fp, allow_pickle=True)
        data = data.reshape((data.shape[0], 32, -1)) if wtype == 'wcycled' else data
        data = data[:,0,:] - mean_w
        samples_232_w += data.tolist()
        wtype_labels += [wtype]*len(data)

    samples_232_w = np.array(samples_232_w)
    wtype_labels = np.array(wtype_labels)
    samples_w_pca = pca_w.transform(samples_232_w-mean_w)

    # plot data on the PC space
    fig, ax = plt.subplots(1, 1, figsize=(10*0.39, 10*0.39))
    sns.scatterplot(x=subsampled_w_pca[:, 0], y=subsampled_w_pca[:, 1], 
                    hue=subsampled_labels_w, ax=ax)
    sns.scatterplot(x=samples_w_pca[:, 0], y=samples_w_pca[:, 1],
                    hue=wtype_labels, ax=ax, palette='tab10', marker='x')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    fig.savefig(input_dir + f'PC_Wspace_class{origin_c}_target{target_c}_seed{seed_nb}.png', dpi=300)
    plt.close(fig)
    # plt.show()

def plot_trajectories(data_traj, data_pca_clusters, labels, 
                      pca, mean_data, pred_classes=None, out_dir=None):

    data_traj_pca = pca.transform(data_traj-mean_data)

    fig, ax = plt.subplots(1, 1, figsize=(10*0.39, 10*0.39)) # (5,5)
    sns.scatterplot(x=data_pca_clusters[:, 0], y=data_pca_clusters[:, 1], 
                    hue=labels, ax=ax, palette=cmap_classes, alpha=0.5)
    sns.scatterplot(x=data_traj_pca[:, 0], y=data_traj_pca[:, 1],
                    hue=pred_classes, ax=ax, palette=cmap_classes,
                    marker='X', s=20, edgecolor='black') #  linewidth=2,

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.get_legend().remove()
    if out_dir is not None:
        fig.savefig(out_dir + f'PC_traj_class{class_c}_target{target_c}.png', dpi=300,
                bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

if __name__ == '__main__':
    data_r, data_w, labels = load_data()
    pca_r = build_pc_space(data_r)
    pca_w = build_pc_space(data_w)
    mean_r = np.mean(data_r, axis=0)
    mean_w = np.mean(data_w, axis=0)

    # subsample data
    subsampled_r_pca, subsampled_labels_r = subsample_data(data_r, labels, pca_r)
    subsampled_w_pca, subsampled_labels_w = subsample_data(data_w, labels, pca_w)

    input_dir = "../data/counterfactual_opt/loss_v4_shiftInitSeed=1_eps=1e-05_lambda=0.6_lr=0.5/"
    sample_nb = 4
    class_c = 232
    target_c = 151
    seed_nb = 0


    fn = f"class{class_c}_seed{seed_nb}_r.npy"
    fn_rshift_opt = f"sample{sample_nb}_class{class_c}_seed{seed_nb}_target{target_c}_rshift_across_opt.npy"
    r = np.load(input_dir + fn, allow_pickle=True)[sample_nb]
    rshift_opt = np.load(input_dir + fn_rshift_opt, allow_pickle=True)
    sampled_r = np.concatenate((r[None, ...], rshift_opt), axis=0)


    pred_classes = None

    plot_trajectories(sampled_r, subsampled_r_pca, subsampled_labels_r, 
                      pca_r, mean_r, out_dir=None, pred_classes=pred_classes)