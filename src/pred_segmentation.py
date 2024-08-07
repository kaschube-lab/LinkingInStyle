import os
import sys
import glob

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from src.utils import get_layers, concat_features_torch
from src.extract_feats_segmentation import FeatureExtractor
from src.metadata import SGXL_MODEL, NETWORK_URL, SEG_LABELS

sys.path.append('src/stylegan-xl/')
import dnnlib, legacy

sys.path.append('src/repurpose_gan/')
from model import FewShotCNN

def predict_segmentation(dataset_name, out_dir, w_path=None, w_dir=None, model_size='S',
                         img_size=128, device='cuda'):
    assert w_path is not None or w_dir is not None, 'Either w_path or w_dir must be provided'
    if w_dir is not None:
        w_files = sorted(glob.glob(os.path.join(w_dir, '*.npz')))
    else:
        w_files = [w_path]

    network_name = os.path.join('models', 'sgxl', os.path.basename(NETWORK_URL[SGXL_MODEL]))
    with dnnlib.util.open_url(network_name) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    G.to(device)
    G.eval()

    seg_labels = SEG_LABELS[dataset_name]
    layers = get_layers(G)
    feat_extractor = FeatureExtractor(G, layers)
    feat_extractor.to(device)

    sample_w = np.load(w_files[0], allow_pickle=True)['ws'][0,0,:] # should have shape (1, 32, 512)
    sample_w = torch.tensor(sample_w).to(device)
    sample_feats = feat_extractor(sample_w)
    sample_feats = [f for f in sample_feats.values()][::2]
    sample_feats = concat_features_torch(sample_feats, img_size, img_size)
    n_channels = sample_feats.shape[1]

    net = FewShotCNN(n_channels, len(seg_labels), model_size)
    model_path = os.path.join('models', 'seg', f'FewShotSegmenter_{model_size}_{img_size}_{img_size}_{dataset_name}.pth')
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint, strict=False)
    net.to(device)
    net.eval()

    cmap = plt.get_cmap('Set3')

    for f in w_files:
        # w_filename = os.path.basename(wf)
        w = np.load(f, allow_pickle=True) # should have shape (1, 32, 512)
        if w.shape[0] == 512*32 or len(w.shape) == 2:
            w = w.reshape(1, 32, 512)
        w = torch.tensor(w).to(device)

        feats = feat_extractor(w)
        feats = [f for f in feats.values()][::2]
        feats = concat_features_torch(feats, img_size, img_size)

        pred = net(feats)
        pred = pred.datamax[1].cpu().numpy().astype(np.uint8)

        pred_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        for l in range(len(seg_labels)):
            color = [0,0,0] if l == 0 else np.array(list(cmap(l-1)[:3]))*255
            pred_rgb[pred == l] = color
        
        pred_rgb = Image.fromarray(pred_rgb, 'RGB')
        pred_rgb.save(os.path.join(out_dir, os.path.basename(f)[:-4] + '.png'))
        # sample_{sample_nb}_class_{class_idx}_unit_{unit_i}


