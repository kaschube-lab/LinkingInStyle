import sys
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F

from src.utils import get_layer_names, get_features

sys.path.append('src/repurpose_gan/')
from model import FewShotCNN

from src.metadata import SEG_LABELS

def train_seg_model(dataset_name, n_epochs, model_size='S', img_size=128, device='cuda'):
    seg_labels = SEG_LABELS[dataset_name]

    feats_dir = os.path.join('data', 'seg', 'train', dataset_name, 'features')
    labels_dir = os.path.join('data', 'seg', 'train', dataset_name, 'labels')

    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.npy')))
    layer_names = get_layer_names(os.path.join(feats_dir, os.path.basename(label_files[0])[:-4] + '.npz'))
    n_channels = get_features(os.path.join(feats_dir, os.path.basename(label_files[0])[:-4] + '.npz'),
                                layer_names[::2], img_size, img_size).shape[1]

    train_feats, train_labels = [], []
    for lf in label_files:
        filename = os.path.basename(lf)
        feat = get_features(os.path.join(feats_dir, filename[:-4] + '.npz'), layer_names[::2], img_size, img_size)
        feat = feat.to(device)

        label = np.load(os.path.join(labels_dir, filename))
        label = torch.tensor(label)
        label = torch.nn.functional.interpolate(label[None, ...], (img_size, img_size), mode='nearest')
        label = label[:,0].long().to(device)

        train_feats.append(feat)
        train_labels.append(label)
    
    net = FewShotCNN(n_channels, len(seg_labels), size=model_size)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    net.train()
    
    # start training
    for epoch in range(1, n_epochs+1):
        sample_order = np.arange(len(train_feats))
        np.random.shuffle(sample_order)
    
        for idx in sample_order:
            feat = train_feats[idx]
            label = train_labels[idx]

            optimizer.zero_grad()
            out = net(feat)
            loss = F.cross_entropy(out, label, reduction='mean')

            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{n_epochs} - Loss: {loss.item():6.4f}')

        scheduler.step()
    
    out_path = os.path.join('models', 'seg', f'FewShotSegmenter_{model_size}_{img_size}_{img_size}_{dataset_name}.pth')
    torch.save(net.state_dict(), out_path)
