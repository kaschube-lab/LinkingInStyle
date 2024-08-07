import os
import sys
import io
from zipfile import ZipFile
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.metadata import NETWORK_URL, SGXL_MODEL, CLASSES_DATASET, VAL_TRANSFORM
from src.utils import load_encoder

sys.path.append('src/stylegan-xl/')
import dnnlib, legacy
from torch_utils import gen_utils

def get_rs_for_tuning(G, encoder, dataset_name, samples_per_class, truncation=.7, gen_seed=136, 
                      device='cuda'):
    rs_all, ws_all, class_idx = [], [], []
    for c in CLASSES_DATASET[dataset_name]:
        ws = gen_utils.get_w_from_seed(G, samples_per_class, device, truncation, class_idx=c, seed=gen_seed)
        ws_all.extend([ws[i][0].cpu().numpy().ravel() for i in range(samples_per_class)])
        class_idx.extend([c]*samples_per_class)
        for w in ws:
            img = G.synthesis(w.reshape((1, 32, -1)), noise_mode='none', force_fp32=True)
            img = (img + 1) * (255/2.)
            img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
            img = Image.fromarray(img[0], 'RGB')
            img = VAL_TRANSFORM(img)
            r = encoder(img[None, ...].to(device))
            rs_all.append(r.detach().cpu().numpy().ravel().squeeze())

    rs_all = np.array(rs_all).squeeze()
    ws_all = np.array(ws_all).squeeze()
    class_idx = np.array(class_idx)

    return ws_all, rs_all, class_idx


def tune_single_unit(G, linking_nw, start_rs, unit_i, n_steps, activ_range, class_idx, 
                           sample_nb, outdir=None, save_w=False, device='cuda'):
    rs_traj = []
    rs = start_rs.copy()

    # for each representation r, activate unit_i with sampled value
    for unit_val in np.linspace(activ_range[0], activ_range[1], n_steps):
        rs[unit_i] = unit_val
        rs_traj.append(rs.copy())
    
    rs_traj = np.array(rs_traj)
    ws_traj = linking_nw.predict(rs_traj) 
    ws_traj = np.tile(ws_traj, 32)
    ws_traj = ws_traj.reshape(n_steps, 32, -1)
    ws_traj = torch.tensor(ws_traj).to(device)

    # generate images
    synth_imgs = G.synthesis(ws_traj, noise_mode='none', force_fp32=True)
    synth_imgs = (synth_imgs + 1) * (255/2.)
    synth_imgs = synth_imgs.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()

    if outdir is None:
        return synth_imgs, rs_traj, ws_traj
    
    imgs_zip = []
    ws_zip = []
    zip_name = f'sample_{sample_nb}_class_{class_idx}_unit_{unit_i}' # TODO: decide nomenclature of zip files

    for i, img in enumerate(synth_imgs):
        img_name = f'{zip_name}_step_{i}'
        img_pil = Image.fromarray(img)
        file_object = io.BytesIO()
        img_pil.save(file_object, 'PNG')
        img_pil.close()
        imgs_zip.append([img_name, file_object])
    
    if save_w:
        for i, w in enumerate(ws_traj.cpu().numpy()):
            w_name = f'{zip_name}_step_{i}'
            w_file = io.BytesIO()
            np.save(w_file, w)
            ws_zip.append([w_name, w_file])
        
    zip_files_bytes_io = io.BytesIO()
    with ZipFile(zip_files_bytes_io, 'w') as zipf:
        for img_name, bytes_stream in imgs_zip:
            zipf.writestr(os.path.join(zip_name, f'{img_name}.png'), bytes_stream.getvalue())
        if save_w:
            for w_name, bytes_stream in ws_zip:
                zipf.writestr(os.path.join(zip_name, f'{w_name}.npy'), bytes_stream.getvalue())
    
    with open(os.path.join(outdir, f'{zip_name}.zip'), 'wb') as f:
        f.write(zip_files_bytes_io.getvalue())
    
    return synth_imgs, rs_traj, ws_traj

def run_systematic_tuning(classifier_name, dataset_name, n_steps, samples_per_class, outdir, units=None, 
         model_type='LR', device='cuda'):
    network_name = os.path.join('models', 'sgxl', os.path.basename( NETWORK_URL[SGXL_MODEL]))
    with dnnlib.util.open_url(network_name) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G.eval()

    encoder = load_encoder(classifier_name, device, get_full=False)
    encoder.eval()

    lnw_path = os.path.join('models', 'lnw', f'LinkingNW_{model_type}_{classifier_name}_{dataset_name}.pkl')

    assert os.path.exists(lnw_path), f"Linking network not found: {lnw_path}"
    with open(lnw_path, 'rb') as f:
        linking_nw, mse = pickle.load(f)

    ws_all, rs_all, class_idx = get_rs_for_tuning(G, encoder, dataset_name, samples_per_class, device=device)

    if units is None:
        units = np.arange(rs_all.shape[1])

    ranges = [[np.min(rs_all[:, i]), np.max(rs_all[:, i])] for i in units]
    df = pd.DataFrame({'unit': units, 'min': [r[0] for r in ranges], 'max': [r[1] for r in ranges]})
    df.to_csv(os.path.join(outdir, 'activation_ranges.csv'), index=False)

    for class_i, c in enumerate(CLASSES_DATASET[dataset_name]):
        for sample_nb in range(samples_per_class):
            k = class_i * samples_per_class + sample_nb
            for i_unit, unit in enumerate(units):
                synth_imgs, rs_traj, ws_traj = tune_single_unit(G, linking_nw, rs_all[k], unit, n_steps, 
                                                                ranges[i_unit], c, sample_nb, outdir=outdir, 
                                                                save_w=False, device=device)
def main_from_args(args):
    run_systematic_tuning(args.network_name, args.classifier_name, args.dataset_name, args.n_steps, args.samples_per_class, 
         args.outdir, args.units, args.model_type, args.device)
