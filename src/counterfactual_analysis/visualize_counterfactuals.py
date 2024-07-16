import os, sys
import pickle
import argparse
import numpy as np

sys.path.append('./stylegan-xl/')
import dnnlib
import legacy
from generate_data import network_url, Model

from PIL import Image
import torch

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize counterfactual activations')
    parser.add_argument('--encoder', type=str, default='rnet50', help='encoder name')
    parser.add_argument('--activations_file', type=str, help='Path to file with counterfactuals')
    parser.add_argument('--output_dir', type=str, help='Path to directory to save images')

    args = parser.parse_args()

    # load generator
    network_name = os.path.join('models/', network_url[Model].split("/")[-1])

    with dnnlib.util.open_url(network_name) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    encoder_name = args.encoder
    # load mapping network
    lr_file_name = f'MappingNW_LR_{encoder_name}_noNoise_232,151,254,197,178.pkl'
    save_path='./models'
    if os.path.exists(os.path.join(save_path, lr_file_name)):
        with open(os.path.join(save_path, lr_file_name), 'rb') as file:
            mapping_nw = pickle.load(file)

    # load r counterfactuals
    r_counterfactuals = np.load(args.activations_file, allow_pickle=True)
    r_counterfactuals = np.array(r_counterfactuals)

    ws_move = mapping_nw.predict(r_counterfactuals)

    ws_move = np.tile(ws_move, 32)
    synth_image = G.synthesis(torch.tensor(ws_move.reshape((len(ws_move), 32, -1))).to(device), noise_mode='none', force_fp32=True) #.to(device)
    synth_image = (synth_image + 1) * (255/2)
    synth_image_ = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()

    for k, img in enumerate(synth_image_):
        img_name = f"counterfactual_chihuahuha-dog_1.5_k={k}.png"
        pil_img = Image.fromarray(img)
        pil_img.save(args.output_dir + img_name)
