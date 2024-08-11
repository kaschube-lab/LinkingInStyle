import os
import sys
import pickle

import numpy as np
import torch
from PIL import Image

from src.utils import r_to_w_tensor, w_to_img, load_encoder, \
    encode_images_in_r
from src.metadata import VAL_TRANSFORM, SGXL_MODEL, NETWORK_URL

sys.path.append('src/stylegan-xl/')
import dnnlib, legacy

MAX_STEPS = 2000

def loss_fn(r, w, shift, origin_class, target_class, encoder, linking_nw, device='cuda'):
    r_shifted = r - shift
    pred = encoder.fc(r_shifted).to(device)
    w_cycled = r_to_w_tensor(r, linking_nw, device)
    w_shifted_cycled = r_to_w_tensor(r_shifted, linking_nw, device)
    loss = -pred[target_class] + 0.6*pred[origin_class] - 10*torch.cosine_similarity(w_cycled, w_shifted_cycled, dim=-1)

    return loss

def optimize_shift(G, r, w, origin_class, target_class, lr, encoder_full, 
                   linking_nw, manualseed=0, device='cuda'):
    r_tensor = torch.tensor(r).to(device)
    w_tensor = torch.tensor(w).to(device)

    pred_class = origin_class

    torch.manual_seed(manualseed)
    shift = torch.rand(2048, requires_grad=True, device=device) # initial shift
    steps = 0

    while pred_class != target_class:
        loss = loss_fn(r_tensor, w_tensor, shift, origin_class, target_class, encoder_full, linking_nw, device)
        # print(f'loss: {loss.item()}')
        loss.backward(retain_graph=True)
        
        with torch.no_grad():
            shift -= lr * shift.grad
            shift.grad.zero_()
        
        r_shifted = r_tensor - shift

        w_shifted = r_to_w_tensor(torch.relu(r_shifted), linking_nw, device).squeeze()
        w_shifted = w_shifted.repeat(32)
        img = w_to_img(G, w_shifted, device)
        img = VAL_TRANSFORM(img)
        pred_shifted = encoder_full(img[None, ...].to(device))
        pred_shifted = torch.softmax(pred_shifted, dim=1)[0]
        pred_class = pred_shifted.argmax(dim=0).item()

        steps += 1
        if steps % 100 == 0:
            print(f'Step {steps}: proba origin: {pred_shifted[origin_class].item()}, proba target: {pred_shifted[target_class].item()}')

        if steps > MAX_STEPS:
            return None
    
    print(f'steps needed: {steps}')
    shift = shift.detach().cpu().numpy()
    
    return shift
    
def gen_counterfactual(classifier_name, dataset_name, orig_img_path, orig_w_path,
                        origin_class, target_class, out_dir, lr=0.5, manual_seed=0, 
                        model_type='LR', device='cuda'):
    network_name = os.path.join('models', 'sgxl', os.path.basename(NETWORK_URL[SGXL_MODEL]))
    with dnnlib.util.open_url(network_name) as f:        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()

    # load partial encoder
    encoder_partial = load_encoder(classifier_name, device, get_full=False)
    encoder_partial.eval()

    # load full encoder
    encoder_full = load_encoder(classifier_name, device, get_full=True)
    encoder_full.eval()

    # load linking network
    lnw_path = os.path.join('models', 'lnw', f'LinkingNW_{model_type}_{classifier_name}_{dataset_name}.pkl')

    assert os.path.exists(lnw_path), f"Linking network not found: {lnw_path}"
    with open(lnw_path, 'rb') as f:
        linking_nw = pickle.load(f)
    
    img = Image.open(orig_img_path)
    w = np.load(orig_w_path, allow_pickle=True)
    r = encode_images_in_r(encoder_partial, [img], device)[0]

    shift = optimize_shift(G, r, w, origin_class, target_class, lr, encoder_full,
                           linking_nw, manual_seed, device=device)
    
    if shift is None:
        raise ValueError('Target class not reached')
    
    r_shifted = r.copy() - shift
    r_shifted = np.where(r_shifted > 0, r_shifted, 0) # clip negative values

    w_cycled = r_to_w_tensor(torch.tensor(r_shifted).to(device), linking_nw, device)
    w_cycled = w_cycled.repeat(32).reshape(1, 32, 512)
    img_cycled = w_to_img(G, w_cycled, device)
    r_cycled = encode_images_in_r(encoder_partial, [img_cycled], device)[0]

    out_fp = os.path.join(out_dir, os.path.basename(orig_img_path)[:-4] + f'_ctf_target_{target_class}_mseed_{manual_seed}.png')
    img_cycled.save(out_fp)
    
    out_fp = os.path.join(out_dir, os.path.basename(orig_w_path)[:-4] + f'_ctf_shift_{target_class}_mseed_{manual_seed}_wcycled.npy')
    np.save(out_fp, w_cycled.detach().cpu().numpy())
    np.save(out_fp.replace('wcycled', 'rcycled'), r_cycled)