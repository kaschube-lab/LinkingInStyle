import os
import sys
import argparse

import numpy as np
import cv2

from src.motion_visualization import plot_corres

sys.path.append('src/pump/')
from tools.common import image
from post_filter import densify_corres

import test_singlescale as pump

def load_imgs(src_img_path, dst_img_path, resize=0):
    args = argparse.Namespace(img1=src_img_path,
                              img2=dst_img_path, 
                              resize=resize)

    imgs = tuple(map(image, (pump.Main.load_images(args))))
    return imgs

def compute_sparse_corres(src_img_path, dst_img_path, out_path, resize=0, device='cuda'):
    args = argparse.Namespace(img1=src_img_path,
                              img2=dst_img_path,
                              output=out_path,
                              resize=resize,
                              device=device,
                              backward=device,
                              forward=device,
                              reciprocal=device,
                              post_filter=True,
                              desc='PUMP-stytrf',
                              levels=99,
                              nlpow=1.5,
                              border=0.9,
                              dtype='float16',
                              first_level='torch',
                              activation='torch',
                              verbose=0,
                              dbg=(),
                              min_shape=5)
    pump.Main().run_from_args(args)

def load_corres(corres_path):
    return np.load(corres_path)['corres']

def get_dense_corres_from_sparse(sparse_corres_path, img_size, out_path=None):
    sparse_corres = load_corres(sparse_corres_path)
    dense_corres = densify_corres(sparse_corres, img_size)
    
    if out_path is not None:
        np.savez(open(out_path, 'wb'), corres=dense_corres)
    
    return dense_corres

def homography_from_corres(corres):
    corres = corres[:, :4]
    # convert to OpenCV format
    corres = corres[:, [0, 1, 2, 3]].astype(np.float32)

    homography, _ = cv2.estimateAffine2D(corres[:,:2], corres[:,2:4])
    return homography

def warp_and_save(img_path, homography, out_path):
    img = cv2.imread(img_path)
    img_warped = cv2.warpAffine(img, homography, (img.shape[1], img.shape[0]))
    cv2.imwrite(out_path, img_warped)

def comp_vis_motion(src_img_path, dst_img_path, sparse_corres_path, percentile=None, neigh=0, 
                    step_sparsify=1, device='cuda'):
    # example of sparse_corres_path: 'output/img1_to_img2_sparse.npy'
    if not os.path.exists(sparse_corres_path):
        compute_sparse_corres(src_img_path, dst_img_path, sparse_corres_path, device=device)
    sparse_corres = load_corres(sparse_corres_path)

    assert len(sparse_corres) >= 4, 'Not enough correspondences to estimate homography'

    out_dir = os.path.dirname(sparse_corres_path)

    imgs = load_imgs(src_img_path, dst_img_path)
    homography = homography_from_corres(sparse_corres)

    src_warped_path = os.path.join(out_dir, os.path.basename(src_img_path).replace('.png', '_warped.png'))
    warp_and_save(src_img_path, homography, src_warped_path)

    sparse_corres_warped_path = os.path.join(out_dir, os.path.basename(sparse_corres_path).replace('.npy', '_warped.npy'))
    if not os.path.exists(sparse_corres_warped_path):
        compute_sparse_corres(src_warped_path, dst_img_path, sparse_corres_warped_path, device=device)
    sparse_corres_warped = load_corres(sparse_corres_warped_path)

    assert len(sparse_corres_warped) >= 4, 'Not enough correspondences to densify'

    dense_corres_path = os.path.join(out_dir, os.path.basename(sparse_corres_path).replace('_sparse.npy', '_dense.npy'))
    if os.path.exists(dense_corres_path):
        dense_corres = load_corres(dense_corres_path)
    else:
        dense_corres = get_dense_corres_from_sparse(sparse_corres_path, imgs[0].shape[:2], dense_corres_path)

    # visualization
    img_size = imgs[0].shape[:2]
    imgs = load_imgs(src_warped_path, dst_img_path)
    out_paths = [os.path.join(out_dir, f'{os.path.basename(f)[:-4]}_vis.png') for f in [src_warped_path, dst_img_path]]

    plot_corres(dense_corres, img_size, out_paths, imgs, percentile, neigh, step_sparsify)