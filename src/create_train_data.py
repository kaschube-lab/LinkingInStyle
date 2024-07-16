import os, sys
import argparse

sys.path.append('./stylegan-xl/')
import dnnlib
import legacy
from torch_utils import gen_utils

import numpy as np
from PIL import Image
from generate_data import load_encoder, network_url, val_transform
from counterfactual_analysis.image_sampling_rshift import generate_img

import torch

classes_dataset = {'dogs': [232,254,151,197,178],
                    'fungi': [992, 993, 994, 997],
                    'birds': [10, 12, 14, 92, 95, 96],
                    'cars': [468, 609, 627, 717, 779, 817]
                    }

def encode_images_in_r(encoder, imgs, device='cuda'):
    all_r = []
    for synth_image in imgs:
        synth_image = val_transform(synth_image)
        r = encoder(synth_image[None, ...].to(device))
        all_r.append(r.detach().cpu().numpy().ravel().squeeze())
    all_r = np.array(all_r)

    return all_r

def w_to_img(G, w, device='cuda'):
    synth_image = G.synthesis(torch.tensor(w.reshape((1, 32, -1))).to(device), noise_mode='none', force_fp32=True)
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    synth_image = Image.fromarray(synth_image, 'RGB')

    return synth_image

def get_correctly_classified_w(G, encoder, class_idx, n_samples, gen_seed, truncation, device='cuda'):
    correct_ws = []
    correct_imgs = []

    while len(correct_ws) < n_samples:
        ws_tensor = gen_utils.get_w_from_seed(G, n_samples-len(correct_ws), device, args.truncation, 
                                        class_idx=class_idx, seed=args.gen_seed)
        ws = ws_tensor.detach().cpu().numpy()
        imgs = [w_to_img(G, w, device) for w in ws]
        # imgs = generate_img(current_w, G)
        for i, img in enumerate(imgs):
            pred = encoder(val_transform(img)[None, ...].to(device))
            pred = torch.softmax(pred, dim=1)
            _, pred = torch.max(pred, 1)
            if pred == class_idx:
                correct_ws.append(ws[i])
                correct_imgs.append(img)
    
    correct_ws = np.array(correct_ws)
    print(correct_ws.shape)

    return correct_ws, correct_imgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create training data for mapping network between classifier and generator")
    parser.add_argument('--encoder', type=str, default='rnet50', 
                        help='Encoder to use')
    parser.add_argument('--dataset', choices=['dogs', 'fungi', 'birds', 'cars'],
                        help='Dataset to use')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of samples to generate per class')
    parser.add_argument('--gen_seed', type=int, default=5000,
                        help='Seed for generator')
    parser.add_argument('--truncation', type=float, default=0.7,
                        help='Truncation psi for generator')
    args = parser.parse_args()

    print(f"Arguments: {args}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load generator
    Model = 'Imagenet-256' 
    network_name = os.path.join('models/', network_url[Model].split("/")[-1])
    with dnnlib.util.open_url(network_name) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # load partial encoder
    encoder_partial = load_encoder(args.encoder, device)
    # load full encoder
    encoder_full = load_encoder(args.encoder, device, get_full=True)

    classes = classes_dataset[args.dataset]

    for c in classes:
        ws, imgs = get_correctly_classified_w(G, encoder_full, c, args.n_samples, args.gen_seed, args.truncation, device)
        rs = []
        for img in imgs:
            # img = w_to_img(G, w, device)
            r = encode_images_in_r(encoder_partial, [img], device)
            rs.append(r)
        rs = np.array(rs).squeeze()
        np.savez(f'./data/data_{args.encoder}_{args.n_samples}N_{c}.npz', 
                    ws=ws, rs=rs)
        print(f'Class {c} done')