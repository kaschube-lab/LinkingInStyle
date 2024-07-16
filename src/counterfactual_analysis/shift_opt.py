import os, sys
import numpy as np
import pickle
import argparse
from PIL import Image

sys.path.append('../stylegan-xl/')
import dnnlib
import legacy
from torch_utils import gen_utils
import torch

from counterfactual_opt import optimize_shift

sys.path.append('../')
from generate_data import network_url
from generate_data import load_encoder
from create_traindata_mnw import encode_images_in_r, w_to_img, classes_dataset

def generate_ws(G, n_samples, class_idx, gen_seed=0, truncation=.4, save_w=False, output_dir=None, device='cuda'):
    ws = gen_utils.get_w_from_seed(G, n_samples, device, truncation, class_idx=class_idx, seed=gen_seed)
    if save_w:
        if device == 'cuda':
            np.save(output_dir + f'class{class_idx}_seed{gen_seed}_w.npy', ws.detach().cpu().numpy())
        else:
            np.save(output_dir + f'class{class_idx}_seed{gen_seed}_w.npy', ws)
    return ws

def generate_images(G, n_samples, class_idx, gen_seed=0, truncation=.4, save_w=False, output_dir=None, device='cuda'):
    ws = generate_ws(G, n_samples, class_idx, gen_seed, truncation, save_w, output_dir, device)
    imgs = []
    for w in ws:
        synth_image = w_to_img(G, w, device)
        imgs.append(synth_image)

    return imgs

def convert_r_to_imgs(G, mapping_nw, rs, device):
    ws = mapping_nw.predict(rs)
    ws = np.tile(ws, 32)
    imgs = []
    for w in ws:
        synth_image = w_to_img(G, w, device)
        imgs.append(synth_image)
    
    return ws, imgs

def get_prediction(encoder, r, device):
    pred = encoder.fc(torch.tensor(r).to(device))
    class_max = pred.argmax().item()
    pred = torch.softmax(pred, dim=0)
    proba_class_max = pred[class_max].item()

    return pred, class_max, proba_class_max

def run_shift_optimization(G, encoder, encoder_partial, mapping_nw, all_r, ws, origin_class,
                   target_class, gen_seed, lambda_, lr, output_dir, device, initseed, 
                   loss_type):
    all_rshifted = []
    all_rcycled = []
    all_wcycled = []

    for j, r in enumerate(all_r):
        # pred = encoder.fc(torch.tensor(r).to(device))
        # pred = torch.softmax(pred, dim=0)
        shift, rshift_across_opt = optimize_shift(G, all_r, r, ws[j], 
                                origin_class=origin_class, 
                                target_class=target_class,
                                lambda_init=lambda_,
                                lr=lr,
                                encoder=encoder,
                                mapping_nw=mapping_nw,
                                device=device,
                                manualseed=initseed,
                                loss_type=loss_type)
        if shift is None:
            print(f'Target value was not reached for example{j}')
            continue
        r_shifted = r.copy() - shift
        r_shifted = np.where(r_shifted > 0, r_shifted, 0)
        all_rshifted.append(r_shifted)
        # np.save(output_dir + f'sample{j}_class{origin_class}_seed{gen_seed}_target{target_class}_rshift_across_opt.npy', rshift_across_opt)
        
        w_cycled, img_cycled = convert_r_to_imgs(G, mapping_nw, [r_shifted], device)
        img_cycled[0].save(output_dir + f'example{j}_class{origin_class}_seed{gen_seed}_counterfactual_target{target_class}.png')
        all_wcycled.append(w_cycled[0])
        r_cycled = encode_images_in_r(encoder_partial, img_cycled, device)[0]
        all_rcycled.append(r_cycled)

        pred, class_max, proba_class_max = get_prediction(encoder, r_cycled, device)
        print(f'Predicted class: {class_max}')
        # print(f'Predicted class probability: {round(pred_proba,2)}')
        print(f'Probability of target class: {pred[target_class].item()}')
    
    all_rshifted = np.array(all_rshifted)
    all_rcycled = np.array(all_rcycled)
    all_wcycled = np.array(all_wcycled)
    np.save(output_dir + f'class{origin_class}_seed{gen_seed}_r.npy', all_r)
    np.save(output_dir + f'class{origin_class}_seed{gen_seed}_target{target_class}_rshifted.npy', all_rshifted)
    np.save(output_dir + f'class{origin_class}_seed{gen_seed}_target{target_class}_rcycled.npy', all_rcycled)
    np.save(output_dir + f'class{origin_class}_seed{gen_seed}_target{target_class}_wcycled.npy', all_wcycled)

def use_avg_shift_over_samples(G, encoder, encoder_partial, mapping_nw, output_dir, 
                   origin_class, target_class, gen_seed, device):
    all_r = np.load(output_dir + f'class{origin_class}_seed{gen_seed}_r.npy', allow_pickle=True)
    all_rshifted = np.load(output_dir + f'class{origin_class}_seed{gen_seed}_target{target_class}_rshifted.npy', allow_pickle=True)

    shifts = np.array([r - rshifted for r, rshifted in zip(all_r, all_rshifted)])
    mean_shift = np.mean(shifts, axis=0)

    all_rcycled = []
    all_wcycled = []
    all_rshifted = []
    for j, r in enumerate(all_r):
        r_mshifed = r.copy() - mean_shift
        r_mshifed = np.where(r_mshifed > 0, r_mshifed, 0)
        all_rshifted.append(r_mshifed)

        w_cycled, img_cycled = convert_r_to_imgs(G, mapping_nw, [r_mshifed], device)
        all_wcycled.append(w_cycled[0])

        img_cycled[0].save(output_dir + f'example{j}_class{origin_class}_seed{gen_seed}_counterfactual_target{target_class}_meanShift.png')
        r_cycled = encode_images_in_r(encoder_partial, img_cycled, device)[0]
        all_rcycled.append(r_cycled)

        pred, class_max, proba_class_max = get_prediction(encoder, r_cycled, device)
        print(f'Predicted class: {class_max}')
        print(f'Predicted class probability: {round(proba_class_max,2)}')
    
    np.save(output_dir + f'class{origin_class}_seed{gen_seed}_target{target_class}_meanShift_rshifted.npy', all_rshifted)
    np.save(output_dir + f'class{origin_class}_seed{gen_seed}_target{target_class}_meanShift_rcycled.npy', all_rcycled)
    np.save(output_dir + f'class{origin_class}_seed{gen_seed}_target{target_class}_meanShift_wcycled.npy', all_wcycled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Counterfactual optimization')
    parser.add_argument('-oc', '--origin_class', type=int, required=True,
                        help='Origin class')
    parser.add_argument('-tc', '--target_class', type=int, required=True,
                        help='Target class')
    parser.add_argument('-gs', '--gen_seed', type=int, default=0, 
                        help='Generator seed')
    parser.add_argument('-l', '--loss', type=str, default='v1',
                        help='Loss function to use for shift optimization')
    parser.add_argument('-ms', '--manualseed', default=None,
                        help='Manual seed for shift initialization')
    parser.add_argument('-lr', '--lr', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('-lmb', '--lambda_', type=float, default=1e-4,
                        help='Lambda for identity term in loss function')
    parser.add_argument('-o', '--out', type=str, required=True,
                        help='Output directory')
    parser.add_argument('-sd', '--samples_dir', type=str, required=True,
                         help='Directory where samples are stored')
    parser.add_argument('-ds', '--dataset', choices=['dogs', 'fungi', 'birds'], 
                        required=True, help='Dataset to use')
    args = parser.parse_args()

    ############### 
    print(args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    encoder_name = 'rnet50'

    # image generation parameters
    n_samples = 2
    truncation = .4
    # gen_seed = 0

    output_dir = args.out + f'loss_{args.loss}_shiftInitSeed={args.manualseed}_lambda={args.lambda_}_lr={args.lr}/'
    os.makedirs(output_dir, exist_ok=True)
    ###############

    # load generator
    Model = 'Imagenet-256' 
    network_name = os.path.join('../models/', network_url[Model].split("/")[-1])
    with dnnlib.util.open_url(network_name) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # load partial encoder
    encoder_partial = load_encoder(encoder_name, device)

    # load mapping network
    c_ref = ','.join([str(c) for c in classes_dataset[args.dataset]])
    lr_file_name = f'MappingNW_LR_{encoder_name}_noNoise_{c_ref}.pkl'
    save_path='../models'
    if os.path.exists(os.path.join(save_path, lr_file_name)):
        with open(os.path.join(save_path, lr_file_name), 'rb') as file:
            mapping_nw = pickle.load(file)
    else:
        raise ValueError('Trained mapping network not found')

 
    imgs = [Image.open(args.samples_dir + f'example{i}_class{args.origin_class}_seed{args.gen_seed}.png') for i in range(n_samples)]
    ws = np.load(args.samples_dir + f'class{args.origin_class}_seed{args.gen_seed}_w.npy', allow_pickle=True)
    
    # encode images
    all_r = encode_images_in_r(encoder_partial, imgs, device)

    # load full encoder
    encoder = load_encoder(encoder_name, device, get_full=True)

    run_shift_optimization(G, encoder, encoder_partial, mapping_nw, all_r, ws, 
                           origin_class=args.origin_class, 
                           target_class=args.target_class, 
                           gen_seed=args.gen_seed, 
                           lambda_=args.lambda_, 
                           lr=args.lr,
                           output_dir=output_dir,
                           device=device, 
                           initseed=args.manualseed, 
                           loss_type=args.loss)
    # # use_avg_shift_over_samples(G, encoder, encoder_partial, mapping_nw, output_dir,
    # #                origin_class, args.target_class, gen_seed, device)