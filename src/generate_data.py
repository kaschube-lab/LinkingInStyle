import io
import os, time, glob
import pickle
import shutil
from zipfile import ZipFile
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import requests
import torchvision.transforms as transforms

import argparse


import sys
sys.path.append('./stylegan-xl/')
import dnnlib
import legacy

from torchvision.models.resnet import resnet50
from torchvision.models import vgg16, alexnet
from torchvision.models.densenet import densenet201

from torch_utils import gen_utils
import PIL


from train_linking_network import train_mapping, classes_dataset
from ipirm import Model_Imagenet


# Functions (many must be trimmed too)

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def norm1(prompt):
    "Normalize to the unit sphere."
    return prompt / prompt.square().sum(dim=-1,keepdim=True).sqrt()

Model = 'Imagenet-256' 


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

# Load a resnet model and get the last feature layer before the prediction layer

# Here I use resnet50, but please feel free to try this with any other model (e.g. VGG16, etc.)
# It might even be very interesting to test which models work better, which have in-/equivariances, redundand features
#@title General functions
from sklearn.linear_model import Ridge, ElasticNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,])

def load_encoder(encoder_name, device, get_full=False):
    if encoder_name == 'rnet50':
        print('load rnet50')
        net = resnet50(pretrained=True) #weights="IMAGENET1K_V2")
    elif encoder_name == 'vgg16':
        print('load vgg16')
        net = vgg16(pretrained=True) #weights="IMAGENET1K_V2")
    elif encoder_name == 'alexnet':
        print('load alexnet')
        net = alexnet(pretrained=True) #weights="IMAGENET1K_V2")
    elif encoder_name == 'densenet201':
        print('load densenet201')
        net = densenet201(pretrained=True) #weights="IMAGENET1K_V2")
    elif encoder_name == 'ipirm':
        print('load ip-irm')
        net = Model_Imagenet()
        checkpoint = torch.load('./models/model_ipirm.pth', map_location=device)
        net.load_state_dict(checkpoint, strict=False)
    if not get_full:
        net = torch.nn.Sequential(*(list(net.children())[:-1]))
    net.to(device)
    net.eval();
    return net


def get_features_for_mapping(G, encoder, N, classes, batch_size, 
                             truncation=.7, gen_seed=136, device='cuda'):
    """
    Get the features from the encoder model, i.e., the classifier.
    G: GAN
    encoder: classifier
    N: number of generated images per class
    classes: imagenet class index (https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)
    batch_size: batch size to sample images from GAN
    truncation: truncation for GAN (usually around .5 to .7). High values cause higher variability but also more bullshit

    Return:
    ws_all: all sampled variables in W-space of GAN
    feat_all: all encoder features 
    c_indices: array of the corresponding class indices to the feats and ws.
    """
    encoder.eval()
    G.eval()
    feat_all, ws_all, c_indices = [], [], []
    # np.random.seed(0)
    # gen_seed = 136
    print(f"Seed for generator: {gen_seed}")
    for c_idx in classes:
        print(c_idx, end=',')
        counter = 0
        while counter < N:
            n_samples = np.min([batch_size, N - counter])
            ws = gen_utils.get_w_from_seed(G, n_samples, device, truncation, class_idx=c_idx, seed=gen_seed)
            ws_list = [ws[i][0].cpu().numpy().ravel() for i in range(len(ws))]
            assert ws_list[0].shape == (512, ), print(ws_list[0].shape)
            ws_all.extend(ws_list)
            for i, w in enumerate(ws):
                synth_image = G.synthesis(w.reshape((1, 32, -1)), noise_mode='none', force_fp32=True)
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                synth_image = PIL.Image.fromarray(synth_image, 'RGB')
                synth_image = val_transform(synth_image)
                feats = encoder(synth_image[None, ...].to(device))
                feat_all.append(feats.detach().cpu().numpy().ravel().squeeze())
                c_indices.append(c_idx)
            counter += batch_size
        
    return np.array(ws_all).squeeze(), np.array(feat_all).squeeze(), np.array(c_indices)

def get_features_from_real_images(encoder, image_paths):
    encoder.eval()
    feat_all = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = val_transform(img)
        feats = encoder(img[None, ...].to(device))
        feat_all.append(feats.detach().cpu().numpy().ravel().squeeze())

    return np.array(feat_all)

def get_rep_layer_size(model_name):
    sizes = {
    'rnet50': 2048,
        'ipirm': 2048,
        'alexnet': 9216,
        'densenet201': 94080,
        'vgg16': 25088
    }
    return sizes[model_name]

def create_image(w):
    synth_image = G.synthesis(torch.tensor(np.tile(w, 32).reshape((1, 32, -1))).to(device))
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return PIL.Image.fromarray(synth_image, 'RGB')


def create_ws(G, N, classes, device, truncation_psi=.7):
    """
    Creates a list of W-space variables.
    """
    ws_all = []
    gen_seed = 0
    print(f"Seed for generator: {gen_seed}")
    for c_idx in classes:
        ws = gen_utils.get_w_from_seed(G, N, device, truncation_psi, class_idx=c_idx, seed=gen_seed)
        ws_all.extend([ws[i][0].cpu().numpy().ravel() for i in range(len(ws))])
    return ws_all
  

def move_along_pc(w_start, component, range, N):
    """
    Move along prinicple component in the W-space
    """
    ws_move = np.array([w_start + (x*component) for x in np.linspace(range[0], range[1], N)])
    ws_move = np.tile(ws_move, 32)
    synth_image = G.synthesis(torch.tensor(ws_move.reshape((len(ws_move), 32, -1))).to(device))
    synth_image = (synth_image + 1) * (255/2)
    synth_image_ = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
    img_array = np.concatenate([img for img in synth_image_], axis=1)
    PIL.Image.fromarray(img_array, 'RGB').show()


def move_along_single_unit(G, start_feat, mapping_nw, unit_i, n_offsets, 
                             ranges, class_j, example_k,output_dir=None, 
                             save_w=False, device='cuda'):
    """
    Move along one unit in the feature space of encoder, map this to W-space and show images (if True)
    start_feat: features in encoder from which to start
    mapping_nw: mapping linear regression
    unit_i: index of unit to manipulate
    n_offests: number of offsets to change unit by
    ranges: list of min and max range to manipulate unit with.
    show_imgs: boolean if a grid with the manipulated images should get displayed.
    """
    feat_move = []
    feat = start_feat.copy()
    unit_arr = np.zeros(len(start_feat))
    feat[unit_i] = 1
    unit_arr[unit_i] = 1
    for offset in np.linspace(ranges[0], ranges[1], n_offsets):

        feat_move.append(feat + offset * unit_arr)
    feat_move = np.array(feat_move)
    ws_move = mapping_nw.predict(feat_move) 
    ws_move = np.tile(ws_move, 32)
    synth_image = G.synthesis(torch.tensor(ws_move.reshape((len(ws_move), 32, -1))).to(device), noise_mode='none', force_fp32=True) #.to(device)
    synth_image = (synth_image + 1) * (255/2)
    synth_image_ = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
    
    if output_dir is None:
      return synth_image_, feat_move, ws_move

    images = []
    ws_tuned = []
    zip_name = f"example_{example_k}_class{class_j}_unit{unit_i}"

    for k, img in enumerate(synth_image_):
        img_name = f"{zip_name}_move{k}"
        pil_img = PIL.Image.fromarray(img)
        # pil_img.save(img_name)
        file_object = io.BytesIO()
        pil_img.save(file_object, "PNG")
        pil_img.close()
        images.append([img_name, file_object])
    
    if save_w:
        for k, w in enumerate(ws_move):
            img_name = f"{zip_name}_move{k}"
            file_object = io.BytesIO()
            np.save(file_object, w)
            ws_tuned.append([img_name, file_object])
    
    zip_file_bytes_io = io.BytesIO()
    with ZipFile(zip_file_bytes_io, 'w') as zip_file:
        for img_name, bytes_stream in images:
            zip_file.writestr(zip_name+'/'+img_name+".png", bytes_stream.getvalue())
        for img_name, bytes_stream in ws_tuned:
            zip_file.writestr(zip_name+'/'+img_name+".npy", bytes_stream.getvalue())
        
    # save the zip file to disk
    with open(os.path.join(output_dir, zip_name + '.zip'), 'wb') as f:
        f.write(zip_file_bytes_io.getvalue())

    return synth_image_, feat_move, ws_move

    
def generate(encoder_name, output_dir, classes, 
             n_offsets, n_units, samples_per_class,
             device):
    # encoder_name = 'rnet50'
    encoder = load_encoder(encoder_name, device)

    N = 5000
    batch_size = 10
    
    # ## 
    c_ref = ','.join([str(c) for c in classes])
    lr_file_name = f'MappingNW_LR_{encoder_name}_{c_ref}.pkl'
    save_path='./models'

    if os.path.exists(os.path.join(save_path, lr_file_name)):
        print('Loading existing mapping network')
        with open(os.path.join(save_path, lr_file_name), 'rb') as file:
            mapping_nw = pickle.load(file)
    else:
        ws, feats, c_indices = [], [], []
        for c in classes:
            print('data path', f'data/data_{encoder_name}_{N}N_{c}.npz')
            if os.path.exists(f'data/data_{encoder_name}_{N}N_{c}.npz'):
                d = np.load(f'data/data_{encoder_name}_{N}N_{c}.npz', allow_pickle=True)
                ws_ = d['ws']
                feats_ = d['feats']
                cindices_ = d['c_indices']
                # print(ws.shape, feats.shape, cindices.shape)
                ws.extend([w for w in ws_])
                feats.extend([f for f in feats_])
                c_indices.extend([c for c in cindices_])
            else:
                print('Could not find existing data file')
                ws, feats, c_indices = get_features_for_mapping(G, encoder, N, classes, batch_size, truncation=.4)
        ws = np.array(ws)
        feats = np.array(feats)
        c_indices = np.array(c_indices)

        mapping_nw = train_mapping(feats, ws, model_type='LR')
        with open(os.path.join(save_path, lr_file_name), 'wb') as file:
            pickle.dump(mapping_nw, file)


    seed = 136
    np.random.seed(seed)
    print(f">>> Random seed for reproducibility: {seed}")
    n = 100
    ws_test, feats_test, c_test = get_features_for_mapping(G, encoder, n, classes, batch_size, truncation=.5)

    t0 = time.time()

    print(f">> n_offsets = {n_offsets}")
    print(f"Manipulating {n_units} units for {samples_per_class} samples of each class in {classes}, which gives {n_units*samples_per_class*len(classes)} sequences in total.")

    min_max = np.array([[np.min(feats_test[:, i]), np.max(feats_test[:, i])] for i in range(feats_test.shape[1])])
    for class_i in range(len(classes)): # index of class in classes list (above) that you want to look at

        print(f'Class {classes[class_i]}')
        for k in range(samples_per_class):
            example_k = class_i * n + k

            for unit_i in np.arange(get_rep_layer_size(encoder_name))[:n_units]: # selected_units: # np.arange(get_rep_layer_size(encoder_name)):
                if os.path.exists(os.path.join(output_dir, f'example_{example_k}_class{classes[class_i]}_unit{unit_i}.zip')):
                    continue
                imgs, feat_move, ws_move = move_along_single_unit(G, 
                                                                  feats_test[example_k], 
                                                                  mapping_nw, 
                                                                  unit_i, 
                                                                  n_offsets, 
                                                                  min_max[unit_i], 
                                                                  class_j=classes[class_i], 
                                                                  example_k=example_k, 
                                                                  output_dir=output_dir,
                                                                  save_w=True,
                                                                  device=device)
        
    tf = time.time()- t0
    print(f">>> Total runtime: {tf}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train mapping network')
    parser.add_argument('--encoder', type=str, default='rnet50', 
                        help='Encoder to use')
    parser.add_argument('--dataset', choices=['dogs', 'fungi', 'birds', 'cars'], 
                        required=True, help='Dataset to use')
    parser.add_argument('-o', '--outdir', type=str, default='output/',
                        help='Directory to save resulting images.')
    parser.add_argument('-nu', '--nunits', type=int,
                        help='Number of units to tune')
    parser.add_argument('-noff', '--noffsets', type=int, 
                        help='Number of values to sample within the unit range')
    parser.add_argument('-ns', '--nsamples', type=int,
                        help='Number of images to sample per class')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device, file=sys.stderr)
    
    print(f">>> Generating data, using encoder {args.encoder}. \n Resulting images will be saved in {args.outdir}")

    network_name = os.path.join('models/', network_url[Model].split("/")[-1])

    with dnnlib.util.open_url(network_name) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    outdir = os.path.join(args.outdir, args.encoder)
    os.makedirs(outdir, exist_ok=True)

    classes = classes_dataset[args.dataset]
    generate(args.encoder, outdir, classes, args.noffsets, args.nunits, args.nsamples, device)