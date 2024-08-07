import os
import numpy as np

import torch
from PIL import Image
from torchvision.models.resnet import resnet50
from torchvision.models import vgg16, alexnet
from torchvision.models.densenet import densenet201
from torch import nn

from src.metadata import VAL_TRANSFORM

class Model_Imagenet(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model_Imagenet, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        # out = self.g(feature)
        return feature # F.normalize(feature, dim=-1) #, F.normalize(out, dim=-1)

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
        checkpoint = torch.load(os.path.join('models', 'classif', 'model_ipirm.pth'), map_location=device)
        net.load_state_dict(checkpoint, strict=False)
    if not get_full:
        net = torch.nn.Sequential(*(list(net.children())[:-1]))
    net.to(device)
    net.eval()
    return net

def encode_images_in_r(encoder, imgs, device='cuda'):
    all_r = []
    for synth_image in imgs:
        synth_image = VAL_TRANSFORM(synth_image)
        r = encoder(synth_image[None, ...].to(device))
        all_r.append(r.detach().cpu().numpy().ravel().squeeze())
    all_r = np.array(all_r)

    return all_r

def w_to_img(G, w, device='cuda'):
    synth_image = G.synthesis(torch.tensor(w.reshape((1, 32, -1))).to(device), noise_mode='none', force_fp32=True)
    synth_image = (synth_image + 1) * (255/2.)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    synth_image = Image.fromarray(synth_image, 'RGB')

    return synth_image

def r_to_w_tensor(r, linking_nw, device):
    coef = torch.tensor(linking_nw.coef_).to(device)
    intercept = torch.tensor(linking_nw.intercept_).to(device)
    w_cycled = (torch.matmul(r, coef.T) + intercept).to(device)

    return w_cycled

@torch.no_grad()
def concat_features_torch(features, h=256, w=256):
    return torch.cat([torch.nn.functional.interpolate(f, (h,w), mode='nearest') for f in features], dim=1)

def get_layers(G):
    layers = []
    for name, m in G.named_children():
      if name == 'synthesis':
        for name, module in m.named_children():
            layers.append('synthesis.' + name)
    return layers

def get_layer_names(file_path):
    d = np.load(os.path.join(file_path))
    layer_names = list(d.keys())
    return layer_names

def get_features(file_path, layer_names, h, w):
    d = np.load(os.path.join(file_path))
    features = [torch.tensor(d[layer_name]) for layer_name in layer_names]
    return concat_features_torch(features, h=h, w=w)
