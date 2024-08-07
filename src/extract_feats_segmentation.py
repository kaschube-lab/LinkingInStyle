import os
import sys
import glob
import pickle

import numpy as np
import torch
import torch.nn as nn

from src.metadata import NETWORK_URL, SGXL_MODEL
from src.utils import concat_features_torch, get_layers

sys.path.append('src/stylegan-xl/')
import dnnlib, legacy

class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model.synthesis(x, noise_mode='none', force_fp32=True)
        return self._features
    
def extract_features(dataset_name, w_path=None, img_size=128, partition='train', device='cuda'):
    if w_path is None:
        w_files = sorted(glob.glob(os.path.join('data', 'seg', partition, dataset_name, 'w', '*.npy')))
        out_dir = os.path.join('data', 'seg', partition, dataset_name, 'features')
    else:
        w_files = [w_path]
        out_dir = os.path.join(os.path.dirname(w_path), '..', 'features')
        os.makedirs(out_dir, exist_ok=True)
            
    network_name = os.path.join('models', 'sgxl', os.path.basename(NETWORK_URL[SGXL_MODEL]))
    with dnnlib.util.open_url(network_name) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        G.to(device)
        G.eval()

        # Create GAN feature extractor
        layers = get_layers(G)
        feat_extractor = FeatureExtractor(G, layers)
        feat_extractor.to(device)

        for f in w_files:
            w = np.load(f, allow_pickle=True)
            if w.shape[0] == 512*32 or len(w.shape) == 2:
                w = w.reshape(1, 32, 512)
            w = torch.tensor(w).to(device)

            feats = feat_extractor(w)
            feats = [f for f in feats.values()][::2]
            feats = concat_features_torch(feats, img_size, img_size)

            feats= feats.cpu().numpy()
            out_path = os.path.join(out_dir, os.path.basename(f)[:-4] + '.npz')
            np.savez(out_path, feats)


    