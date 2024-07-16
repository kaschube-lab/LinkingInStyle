import os, sys
import glob
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

sys.path.append('./stylegan-xl/')
from generate_data import load_encoder, val_transform

import torch

def get_pred_from_img(encoder, img):
    img = val_transform(img)
    pred = encoder(img[None, ...].to(device))
    pred = torch.softmax(pred, dim=1)

    return pred

def draw_proba_on_img(img, text):
    img = np.array(img)
    hw = img.shape[:2]
    plt.imshow(img)
    plt.text(5, 18, str(text), color='black', fontsize=17,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img = img.resize(hw)
    img = np.array(img)

    return img

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    samples_dir = "../data/samples/"
    counterfactual_dir = "../data/counterfactual_opt/loss_v2_shiftInitSeed=1_eps=0.0001_lambda=0.6_lr=0.5/"
    output_dir = counterfactual_dir + 'paired/'
    os.makedirs(output_dir, exist_ok=True)

    samples_fps = sorted(glob.glob(samples_dir + "*class254*.png"))

    # load partial encoder
    encoder_name = 'rnet50'
    encoder = load_encoder(encoder_name, device, get_full=True)

    for fps in samples_fps:
        sample = Image.open(fps).convert('RGB')
        pred_sample = get_pred_from_img(encoder, sample)
        # search for its corresponding counterfactual
        counterfactual_fps = sorted(glob.glob(counterfactual_dir + fps.split('/')[-1][:-4] + '_counterfactual_*.png'))
        for fpc in counterfactual_fps:
            target_class = int(fpc.split('target')[1].split('.png')[0])
            counterfactual = Image.open(fpc).convert('RGB')
            pred_counterfactual = get_pred_from_img(encoder, counterfactual)  
            ptarget_sample = pred_sample[0, target_class].item()
            ptarget_counterfactual = pred_counterfactual[0, target_class].item()
            # add box with probability of target class on top left corner
            text_ptarget_sample = f'{ptarget_sample:.2f}' if ptarget_sample > 0.01 else f'{ptarget_sample:.0e}'
            text_ptarget_counterfactual = f'{ptarget_counterfactual:.2f}' if ptarget_counterfactual > 0.01 else f'{ptarget_counterfactual:.0e}'
            sample = draw_proba_on_img(sample, text_ptarget_sample)
            counterfactual = draw_proba_on_img(counterfactual, text_ptarget_counterfactual)
            # draw = ImageDraw.Draw(sample)
            # draw.rectangle([(0, 0), (80, 20)], fill='white')
            # draw.text((0, 0), text_ptarget_sample, fill='black', font=ImageFont.truetype("./fonts/arial.ttf", 20))
            # draw = ImageDraw.Draw(counterfactual)
            # draw.rectangle([(0, 0), (80, 20)], fill='white')
            # draw.text((0, 0), text_ptarget_counterfactual, fill='black', font=ImageFont.truetype("./fonts/arial.ttf", 20))

            # put one image on top of the other
            img_pair = np.concatenate((sample, counterfactual), axis=0)
            img_pair = Image.fromarray(img_pair)
            out_fp = output_dir + os.path.basename(fpc)[:-4] + f'_probas.png'
            print(out_fp)
            img_pair.save(out_fp)