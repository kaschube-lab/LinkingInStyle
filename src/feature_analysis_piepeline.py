import argparse
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy as calc_entropy
import skimage.io as sio # import imread, imshow
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage.transform import resize
import pandas as pd
import time
import PIL
import torch

sys.path.append('./stylegan-xl/')
from generate_data import load_encoder, val_transform

labels_datasets = {
    'dogs':  ['bg', 'leg', 'body', 'tail', 'tongue', 'eye', 'nose', 'snout', 'ear', 'head'],
    'fungi': ['bg', 'cap', 'stem'],
    'birds': ['bg', 'beak', 'eye', 'head', 'leg', 'wing', 'body', 'tail']
}

def binarize_mask(mask, color):
    bin_im = mask.copy()
    bin_im[np.where(bin_im != color)] = 0
    bin_im[np.where(bin_im == color)] = 1
    if bin_im.ndim == 3:
        bin_im = np.mean(bin_im, axis=-1)
    return bin_im


def get_area(bin_mask):
    return np.sum(bin_mask)


def get_luminance(lum_img, bin_mask):
    return np.mean(lum_img[bin_mask == 1])


def get_entropy_results(entropy_img, bin_mask):
    e_im = np.where(bin_mask == 1, entropy_img, np.nan)
    return np.nanmean(e_im), np.nanstd(e_im), np.nanmin(e_im), np.nanmax(e_im)


def calc_eccentricity(ellipse):
    w, h = (ellipse[1])
    e = np.sqrt(1 - (w**2/h**2))
    return e


def get_ellipses_results(bin_mask, values, i, label):
    results = []
    contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour_i, cnt in enumerate(contours):
        try:
            ellipse = cv2.fitEllipse(cnt)
            [x, y], [w, h], angle = ellipse
            ellipse_i = np.nan

            for comp_i in range(1, len(values)):
                x1 = values[comp_i, cv2.CC_STAT_LEFT] 
                y1 = values[comp_i, cv2.CC_STAT_TOP] 
                w1 = values[comp_i, cv2.CC_STAT_WIDTH] 
                h1 = values[comp_i, cv2.CC_STAT_HEIGHT] 
                if x1 <= x <= x1+w1 and y1 <= y <= y1+h1:
                    ellipse_i = comp_i
                    break
            e = calc_eccentricity(ellipse)
            res = [i, label, ellipse_i, contour_i, e, angle, x, y, w, h]
            results.append(res)
        except:
            pass
    return results


def analysis_loop(imgs, masks, labels, class_colors):
    imgs_gray = [rgb2gray(img) for img in imgs]
    imgs_entropy = [calc_entropy(img_gray, disk(5)) for img_gray in imgs_gray]
    lum_imgs = [(img[..., 0]*0.299 + img[..., 1]*0.587 + img[..., 2]*0.114)/3 for img in imgs]
    results, results_ellipses = [], []
    for l, label in enumerate(labels[1:]):
        
        if masks.ndim == 4:
            color = class_colors[labels.index(label)]
        else:
            color = l
        bin_masks = [binarize_mask(mask, color) for mask in masks]
        overlap = (bin_masks[0]*bin_masks[-1]).astype(np.uint8)
        analysis = cv2.connectedComponentsWithStats(overlap,4,cv2.CV_32S) 
        (_, _, values, _) = analysis 

        for i, (bin_mask, lum_img, img_entropy) in enumerate(zip(bin_masks, lum_imgs, imgs_entropy)):
            if np.sum(bin_mask) == 0:
                results.append([i, label, 0, 0, 0, 0, 0, 0])
                continue
            area = get_area(bin_mask)
            luminance = get_luminance(lum_img, bin_mask)
            entropy_res = get_entropy_results(img_entropy, bin_mask)
            ellipses_res = get_ellipses_results(bin_mask, values, i, label)
            res = [i, label, area, luminance]
            res.extend(entropy_res)
            results.append(res)
            # if len(ellipses_res) > 0:
            for ellips_res_ in ellipses_res:
                results_ellipses.append(ellips_res_)

    return results, results_ellipses, imgs_entropy


def get_class_relevance(imgs, classifier, device):
    print('get class relevance')
    predictions = []
    for img in imgs:
        img_resized = resize(img, (256, 256), anti_aliasing=True)
        img_resized = val_transform(PIL.Image.fromarray(np.uint8(img_resized)))
        # print('img resized', img_resized.shape)
        preds = classifier(img_resized[None, ...])
        # print('preds', preds.shape)
        predictions.append(preds.detach().cpu().numpy().squeeze())
    return predictions

def rename_files(file_path):  
    print('rename images at', file_path)  
    offsets = np.zeros((2048, 2))
    img_names = [f for f in os.listdir(file_path) if '.npy' not in f and f != '.DS_Store']
    for img_name in img_names:
        try:
            old_file = os.path.join(file_path, img_name)
            unit = img_name.split('__')[-1].split('_')[0]  
            # FixedUnitChangeRange_rnet50_17_0.0_6.699999809265137_allClassSeg__299_all_imgs.png
            offsets[int(unit)] = [float(o) for o in img_name.split('_allClassSeg')[0].split('_')[-2:]]
            new_file = os.path.join(file_path, unit + '.png')
            os.rename(old_file, new_file)
        except ValueError:
            print(img_name)
            continue
    np.save(os.path.join(file_path, 'offsets.npy'), offsets)
    


def add_results_to_df(df, results, seed, unit):
    df_keys = list(df.keys())
    for res in results:
        df['seed'].append(seed)
        df['unit'].append(unit)
        for k, key in enumerate(df_keys[2:]):
            df[key].append(res[k])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train Mapping Network')
    parser.add_argument('--class_i', type=str, default='254') 
    parser.add_argument('--seeds', type=str, default='0', 
                        help='if several seeds then separate them by ;') 
    parser.add_argument('--seeds_list', type=int, default=0, 
                        help='if 1 then np.arange(args.seed) as seeds_list') 
    parser.add_argument('--dir', type=str,
                        help='Directory where the images are stored')
    parser.add_argument('--dataset', choices=['dogs', 'fungi', 'birds'],
                        help='Subclasses dataset to use')
    parser.add_argument('--units', type=str help='Units to analyze separated by ";"')
    args = parser.parse_args()
    print(args)
    labels = labels_datasets[args.dataset]
    
    # save_path = os.path.join('./results/featureSegmentation/', args.class_i)
    save_path = os.path.join(args.dir, args.class_i) #'../data/pump/' # '/Volumes/MyPassport/XAI/results/rnet50/', args.class_i)
    save_entropy_imgs = False
    im_width, im_height = 128, 128
    cmap =  plt.get_cmap('Set3')
    class_colors = [[0, 0, 0]]
    class_colors.extend([(np.array(cmap(i)[:3])*255).astype(int) for i in range(9)])
    if args.seeds_list:
        seeds = np.arange(int(args.seeds))
    else:
        seeds = [int(s) for s in args.seeds.split(';')]
    
    units = [int(u) for u in args.units.split(';')]
    df_keys = ['seed', 'unit', 'seq_i', 'label', 'area', 'luminance', 'mean_entropy', 'std_entropy', 
            'min_entropy', 'max_entropy']
    df_ellips_keys = ['seed', 'unit', 'seq_i', 'label', 'ellips_i', 'contour_i', 'eccentricity', 'angle', 
                    'center_x', 'center_y', 'width', 'height']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)
    
    classifier = load_encoder('rnet50', device, get_full=True)   
    print('loaded classifier')

    for seed in seeds:
        start_time = time.time()
        predictions = [] 
        print(seed, end=':', flush=True)
        df = {key: [] for key in df_keys}
        df_ellips = {key: [] for key in df_ellips_keys}
        curr_path = os.path.join(save_path, str(seed))
        if not os.path.exists(os.path.join(curr_path, '0.png')):
            rename_files(curr_path)
        for unit in units:
            print(unit, end=';', flush=True)
            img_and_mask = sio.imread(os.path.join(curr_path, f'{unit}.png'))
            imgs, masks = [], []
            for i in range(img_and_mask.shape[1]//im_width):
                imgs.append(img_and_mask[:im_height, i*im_width: (i+1)*im_width])
                masks.append(img_and_mask[im_height:, i*im_width: (i+1)*im_width])
            imgs = np.array(imgs); masks = np.array(masks)
            if not os.path.exists(os.path.join(curr_path, 'predictions.npy')):
                preds = get_class_relevance(imgs, classifier, device)
                predictions.append(preds)
            if not os.path.exists(os.path.join(curr_path, 'ellips_results.csv')):
                results, results_ellips, imgs_entropy = analysis_loop(imgs, masks, labels, class_colors)
                df = add_results_to_df(df, results, seed, unit)
                df_ellips = add_results_to_df(df_ellips, results_ellips, seed, unit)
            if save_entropy_imgs:
                pass
        if not os.path.exists(os.path.join(curr_path, 'ellips_results.csv')):
            df = pd.DataFrame.from_dict(df)
            df.to_csv(os.path.join(curr_path, 'analysis_results.csv'), index=False)
            df_ellips = pd.DataFrame.from_dict(df_ellips)
            df_ellips.to_csv(os.path.join(curr_path, 'ellips_results.csv'), index=False)
        if not os.path.exists(os.path.join(curr_path, 'predictions.npy')):
            np.save(os.path.join(curr_path, 'predictions.npy'), predictions)
        print('')
        print('Time (sec)', time.time() - start_time)