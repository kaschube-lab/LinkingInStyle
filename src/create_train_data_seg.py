import os
import sys

import numpy as np

sys.path.append('src/stylegan-xl/')
import dnnlib
import legacy
from torch_utils import gen_utils

from src.utils import w_to_img
from src.metadata import CLASSES_DATASET, SGXL_MODEL, NETWORK_URL

def create_data_to_label(dataset_name, gen_seeds, samples_per_class, truncation=.7, partition='train', 
                         device='cuda'):
    network_name = os.path.join('models', 'sgxl', os.path.basename(NETWORK_URL[SGXL_MODEL]))
    with dnnlib.util.open_url(network_name) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    classes = CLASSES_DATASET[dataset_name]
    out_dir = os.path.join('data', 'seg', partition, dataset_name)
    for subdir in ['w', 'imgs', 'features', 'labels']:
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

    for c in classes:
        n_samples = int(np.ceil(samples_per_class/len(gen_seeds)))
        for seed in gen_seeds:
            ws = gen_utils.get_w_from_seed(G, n_samples, device, truncation, class_idx=c, seed=seed)
            for i, w in enumerate(ws):
                np.save(os.path.join(out_dir, 'w', f'sample_{i}_class_{c}_seed_{seed}.npy'), w.cpu().numpy())

                img = w_to_img(G, w, device)
                img.save(os.path.join(out_dir, 'imgs', f'sample_{i}_class_{c}_seed_{seed}.png'))
                
            
            

            
