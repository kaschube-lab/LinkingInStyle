import shutil
import os
import sys 
sys.path.append('./stylegan-xl')
sys.path.append('./repurposeGANs')
from repurposeGANs.img_segmentation_model import FewShotCNN

import torch
import dnnlib
import legacy
from torch_utils import gen_utils
import argparse
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import PIL
import matplotlib.pyplot as plt
import pickle
import numpy as np
import zipfile


import os
print('cuda path', os.environ.get('CUDA_PATH'))


print('Cuda available?', torch.cuda.is_available())

# encoders
from torchvision.models.resnet import resnet50
from torchvision.models import vgg16, alexnet
from torchvision.models.densenet import densenet201


# ToDO
from train_maren import load_encoder

from imageSegmentation_maren import get_layers, make_segmentation_new, FeatureExtractor
from 


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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])

device = torch.device('cuda')
print('Using device:', device, file=sys.stderr)

def get_feats(G, encoder, c_idx, seed, n_feats=100):
    ws = gen_utils.get_w_from_seed(G, n_feats, device, .7, class_idx=c_idx, seed=seed)
    ws_test = [ws[i][0].cpu().numpy().ravel() for i in range(len(ws))]
    feats_test = []
    print(np.array(ws_test).shape)
    for i in range(0, len(ws), 10):
        w = ws[i:i+10]
        synth_image = G.synthesis(w.reshape((10, 32, -1)).to(device), noise_mode='none', force_fp32=True)
        synth_image = (synth_image + 1) * (255/2)
        synth_images = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
        imgs = []
        for synth_image in synth_images:
            synth_image = PIL.Image.fromarray(synth_image, 'RGB')
            synth_image = val_transform(synth_image)
            # print(synth_image.shape)
            imgs.append(synth_image[None, ...])
        if 'vit' in encoder_name:
            synth_img_preproc = vit_imp_preproc(torch.cat(imgs, dim=0).to(device), encoder)
            feats_ = encoder.encoder(synth_img_preproc)
            if int(args.layer_i) >= 0:
                feats_ = feats_.squeeze()[:, int(args.layer_i)]     
        else:
            feats_ = encoder(torch.cat(imgs, dim=0).to(device)) #synth_image[None, ...].to(device))
        feats_test.extend([f.ravel().squeeze() for f in feats_.detach().cpu().numpy()])
    return feats_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train Mapping Network')
    parser.add_argument('-e', '--encoder_name', type=str, default='')
    parser.add_argument('-c', '--c_idx', type=str, default='')
    parser.add_argument('-i', '--layer_i', type=int, default=-1)
    args = parser.parse_args()
    print(args)

    

    # Load GAN
    print('Load GAN')
    Model = 'Imagenet-256' 
    network_name = os.path.join('models/', network_url[Model].split("/")[-1])

    with dnnlib.util.open_url(network_name) as f:
        print('f:', f)
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        print('type(G)', type(G))
    G.to(device)
    G.eval()

    # Load GAN feature extractor
    print('Load GAN feat extractor')
    layers = get_layers(G) #'synthesis.L21_148_861'
    G_features = FeatureExtractor(G, layers=layers)
    G_features.to(device)

    # Load encoder
    print('Load encoder')
    encoder_name = args.encoder_name
    encoder = load_encoder(encoder_name, device, get_full=False)

    # Load linking NW (Linear Regression Model)
    print('Load Linking NW')

    classes = classes_dataset[args.dataset]

    c_ref = ','.join([str(c) for c in classes])

    lr_file_name = f'MappingNW_LR_{encoder_name}_232,254,151,197,178.pkl'
    # lr_file_name = f'MappingNWLR{encoder_name}_noNoise_232,151,254,197,178.pkl'
    save_path='./models'
    with open(os.path.join(save_path, lr_file_name), 'rb') as file:
        linking_nw = pickle.load(file)

    
    
    model_size, width, height = 'S',  128, 128
    img_seg_classes = ['background', 'legs', 'body', 'tail', 'tongue', 'eyes', 'nose', 'snout', 'ears', 'head']
    n_classes = len(img_seg_classes)
    n_channels = 13588 

    # for c_idx in [232,151]: #,254,197,178]: #232,
    c_idx_ImgSegmenter = '_'.join(['232','254','151', '197','178'])
    # c_idx = 178# 197 #254 #151#232 
    # Load Segmentation model
    print('load ImgSegmentation model')
    ImgSegmenter = FewShotCNN(n_channels, n_classes, size=model_size)
    checkpoint = torch.load(f'./repurposeGANs/trained_models/FewShotSegmenter_{model_size}_{c_idx_ImgSegmenter}_{width}_{height}.pth', map_location=device)
    ImgSegmenter.load_state_dict(checkpoint, strict=False)
    ImgSegmenter.to(device)

    print('get feats')
    c_idx = int(args.c_idx)
    feats_test = get_feats(G, encoder, c_idx, 0, n_feats=100)


    save_path = f'./results/featureSegmentation/{encoder_name}/{c_idx}'
    
    if os.path.exists(save_path + '.zip'):
        with zipfile.ZipFile(save_path + '.zip', 'r') as zip_ref:
            zip_ref.extractall(save_path)
    else:
        os.makedirs(save_path, exist_ok=True)
    ws = gen_utils.get_w_from_seed(G, 100, device, .4, class_idx=c_idx)
    #for seed, feat in enumerate(feats_test):
    for seed, w in enumerate(ws):
        print(seed, end=',', flush=True)
        save_name = f'Segmentation_{encoder_name}_{seed}' 
        #ws = linking_nw.predict(feat.reshape(1, -1))
        #ws = np.tile(ws, 32)
        #ws = torch.tensor(ws.reshape((len(ws), 32, -1))).to(device)
        #synth_images = G.synthesis(ws, noise_mode='none', force_fp32=True)
        synth_images = G.synthesis(w.reshape(1, 32, -1), noise_mode='none', force_fp32=True)
        synth_images = torch.nn.functional.interpolate(synth_images, (height,width), mode='nearest')
        synth_images = (synth_images + 1) * (255/2)
        synth_images = synth_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()

        img_rgb, _ = make_segmentation_new(w.reshape(1, 32, -1), G_features, ImgSegmenter, height, width, n_classes)
        img_concat = np.concatenate([synth_images[0], img_rgb], axis=0)
        synth_image = PIL.Image.fromarray(img_concat, 'RGB')
        synth_image.save(os.path.join(save_path, save_name + '.png'))