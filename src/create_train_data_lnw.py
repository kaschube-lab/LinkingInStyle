import os, sys
import argparse

import numpy as np
import torch

sys.path.append('src/stylegan-xl/')
import dnnlib
import legacy
from torch_utils import gen_utils

from src.utils import w_to_img, load_encoder, encode_images_in_r
from src.metadata import VAL_TRANSFORM, NETWORK_URL, SGXL_MODEL, CLASSES_DATASET


def get_correctly_classified_w(G, encoder, class_idx, n_samples, gen_seed, truncation, device='cuda'):
    correct_ws = []
    correct_imgs = []

    while len(correct_ws) < n_samples:
        ws_tensor = gen_utils.get_w_from_seed(G, 
                                              n_samples-len(correct_ws), 
                                              device, 
                                              truncation, 
                                              class_idx=class_idx, 
                                              seed=gen_seed)
        ws = ws_tensor.detach().cpu().numpy()
        imgs = [w_to_img(G, w, device) for w in ws]
        for i, img in enumerate(imgs):
            pred = encoder(VAL_TRANSFORM(img)[None, ...].to(device))
            pred = torch.softmax(pred, dim=1)
            _, pred = torch.max(pred, 1)
            if pred == class_idx:
                correct_ws.append(ws[i])
                correct_imgs.append(img)
    
    correct_ws = np.array(correct_ws)

    return correct_ws, correct_imgs

def create_paired_data(dataset_name, classifier_name='rnet50', partition='train', n_samples=5000, 
                       gen_seed=0, truncation=0.7, device='cuda'):
    network_name = os.path.join('models', 'sgxl', os.path.basename(NETWORK_URL[SGXL_MODEL]))
    with dnnlib.util.open_url(network_name) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    encoder_partial = load_encoder(classifier_name, device, get_full=False)
    encoder_full = load_encoder(classifier_name, device, get_full=True)

    out_dir = os.path.join('data', 'lnw', partition, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    for c in CLASSES_DATASET[dataset_name]:
        ws, imgs = get_correctly_classified_w(G, encoder_full, c, n_samples, gen_seed, truncation, device)
        rs = encode_images_in_r(encoder_partial, imgs, device)
        rs = np.array(rs).squeeze()

        out_path = os.path.join(out_dir, f'data_{dataset_name}_{classifier_name}_N={n_samples}_seed={gen_seed}_trunc={truncation}_class={c}.npz')
        np.savez(out_path, ws=ws, rs=rs)
        
def main_from_args(args):
    create_paired_data(args.dataset_name, args.classifier_name, args.n_samples, args.gen_seed, args.truncation, args.device)

# parser = argparse.ArgumentParser(description='Create paired data to train linking network')
# parser.add_argument('--dataset_name', type=str, default='dogs', choices=CLASSES_DATASET.keys(),
#                     help='Dataset whose classes to use')
# parser.add_argument('--classifier_name', type=str, default='rnet50',
#                     help='Classifier to use for encoding')
# parser.add_argument('--n_samples', type=int, default=5000,
#                     help='Number of samples per class')
# parser.add_argument('--gen_seed', type=int, default=0,
#                     help='Seed for generating images')
# parser.add_argument('--truncation', type=float, default=0.7,
#                     help='Truncation value for generating images')
# parser.add_argument('--device', type=str, default='cuda')
# args = parser.parse_args()