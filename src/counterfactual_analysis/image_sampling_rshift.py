import os, sys
import pickle
import argparse
import numpy as np

sys.path.append('./stylegan-xl')
from generate_data import load_encoder, val_transform, network_url
import torchvision.transforms as transforms
import dnnlib
import legacy

import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

classes_datasets = {
    'dogs': [232, 254, 151, 197, 178],
    'fungi': [992, 993, 994, 997],
    'birds': [10, 12, 14, 92, 95, 96]
}

network_url = {
    "Imagenet-1024": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet1024.pkl",
    "Imagenet-512": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet512.pkl",
    "Imagenet-256": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl",
    "Imagenet-128": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet128.pkl",
    "Pokemon-1024": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon1024.pkl",
    "Pokemon-512": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon512.pkl",
    "Pokemon-256": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon256.pkl",
    "FFHQ-256": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/ffhq256.pkl"
}

def load_generator(model_name, device):
    network_name = os.path.join('models/', network_url[model_name].split("/")[-1])

    with dnnlib.util.open_url(network_name) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G.to(device)
    G.eval()

    return G

def generate_img(w, G, height=128, width=128, device='cuda'):
    w = torch.tensor(w.reshape((1, 32, -1))).to(device)
    synth_image = G.synthesis(w, noise_mode='none', force_fp32=True)
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()

    return synth_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_i', type=int)
    parser.add_argument('--target_i', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--w_path', type=str)
    parser.add_argument('--n_imgs', type=int, default=3)
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of the sample image we want to use')
    parser.add_argument('--out', type=str,
                        help='Output directory to save the sampled images')
    parser.add_argument('--show_proba', action='store_true',
                        help='Show the probability of the target class in the image')
    parser.add_argument('--add_steps', type=int, default=0,
                        help='Number of additional steps to add to the interpolation')
    parser.add_argument('--dataset', choices=['dogs', 'fungi', 'birds'],
                        help='Dataset to use', required=True)
    args = parser.parse_args()
    print(args)

    print("CUDA_PATH: ", os.environ.get('CUDA_PATH'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    # load mapping network
    encoder_name = 'rnet50'
    c_ref = ','.join([str(c) for c in classes_datasets[args.dataset]])
    lr_file_name = f'MappingNW_LR_{encoder_name}_noNoise_{c_ref}.pkl'
    save_path='./models'
    if os.path.exists(os.path.join(save_path, lr_file_name)):
        with open(os.path.join(save_path, lr_file_name), 'rb') as file:
            mapping_nw = pickle.load(file)
    else:
        raise ValueError('Trained mapping network not found')
    
    # load encoder
    encoder = load_encoder(encoder_name, device, get_full=True)

    # load generator
    model_name = 'Imagenet-256'
    G = load_generator(model_name, device)

    Rs = np.load(os.path.join(args.w_path, f'class{args.class_i}_seed{args.seed}_r.npy'))
    Rs_counterfactual = np.load(os.path.join(args.w_path, f'class{args.class_i}_seed{args.seed}_target{args.target_i}_rshifted.npy'))
    shifts = np.array([r - rshifted for r, rshifted in zip(Rs, Rs_counterfactual)])

    rs_intermediate = []
    ws_intermediate = []
    probas = []
    alphas = np.linspace(0, 1, args.n_imgs)
    n_imgs = args.n_imgs + args.add_steps
    if args.add_steps > 0:
        step_size = alphas[1] - alphas[0]
        alphas = np.concatenate([alphas, np.arange(1, args.add_steps+1)*step_size + alphas[-1]])
    for k in range(n_imgs):
        current_r = Rs[args.sample_idx] - alphas[k]*shifts[args.sample_idx]
        rs_intermediate.append(current_r)
        current_w = mapping_nw.predict([current_r])
        ws_intermediate.append(current_w)
        current_w = np.tile(current_w, 32)
        current_w = torch.tensor(current_w).to(device)
        imgs = generate_img(current_w, G)
        img = Image.fromarray(imgs[0], 'RGB')
        pred = encoder(val_transform(img)[None, ...].to(device))
        pred = torch.softmax(pred, dim=1)
        class_max = torch.argmax(pred, dim=1).item()
        print(f'Predicted class: {class_max} at alpha k={k}')
        prob = pred[0, int(args.target_i)]
        probas.append(prob.item())

        text_ptarget = f'{prob:.2f}' if prob > 0.01 else f'{prob:.0e}'
        out_fp = f'sample{args.sample_idx}_class{args.class_i}_seed{args.seed}_target{args.target_i}_alpha_{str(k).zfill(3)}.png'
        if args.show_proba:
            img = np.array(img)
            plt.imshow(img)
            # plt.text(5, 23, text_ptarget, color='black', fontsize=23,
            plt.text(-5, 5, text_ptarget, color='black', fontsize=30,
                    bbox=dict(facecolor='white', edgecolor='black', 
                              boxstyle='round'))
            plt.axis('off')
            plt.savefig(os.path.join(args.out, out_fp), bbox_inches='tight',
                        dpi=300, transparent=True, pad_inches=0.3) 
            plt.close()
        else:
            img.save(os.path.join(args.out, out_fp))
        
        # draw = ImageDraw.Draw(img)
        # draw.rectangle([(0, 0), (40, 10)], fill='white')
        # draw.text((0, 0), text_ptarget, fill='black', font=ImageFont.truetype("./fonts/arial.ttf", 10))
        # img.save(os.path.join(args.out, out_fp)
    rs_intermediate = np.array(rs_intermediate)
    ws_intermediate = np.array(ws_intermediate)
    probas = np.array(probas)
    np.save(os.path.join(args.out, f'sample{args.sample_idx}_class{args.class_i}_seed{args.seed}_target{args.target_i}_rintermediate.npy'), rs_intermediate, allow_pickle=True)
    np.save(os.path.join(args.out, f'sample{args.sample_idx}_class{args.class_i}_seed{args.seed}_target{args.target_i}_wintermediate.npy'), ws_intermediate, allow_pickle=True)
    np.save(os.path.join(args.out, f'sample{args.sample_idx}_class{args.class_i}_seed{args.seed}_target{args.target_i}_probas.npy'), probas, allow_pickle=True)
