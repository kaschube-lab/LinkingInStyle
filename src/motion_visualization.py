import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def show_hsv_wheel():
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')

    rho = np.linspace(0,1,100) # Radius of 1, distance from center to outer edge
    phi = np.linspace(0, np.pi*2.,1000 ) # in radians, one full circle

    RHO, PHI = np.meshgrid(rho,phi) # get every combination of rho and phi

    h = (PHI-PHI.min()) / (PHI.max()-PHI.min()) # use angle to determine hue, normalized from 0-1
    h = np.flip(h)        
    s = RHO               # saturation is set as a function of radias
    v = np.ones_like(RHO) # value is constant

    h,s,v = h.flatten().tolist(), s.flatten().tolist(), v.flatten().tolist()
    c = [hsv_to_rgb(*x) for x in zip(h,s,v)]
    c = np.array(c)

    ax.scatter(PHI, RHO, c=c)
    _ = ax.axis('off')
    plt.show()

def hsv_from_r_theta(r, theta, max_abs):
    theta = theta + np.pi # offset to invert the direction of vector
    h = theta % (2 * np.pi)
    h_neg = np.where(h < 0)
    h[h_neg] += 2 * np.pi
    h = h / (2 * np.pi) # hue, normalized to [0, 1]

    s = r/max_abs
    s = np.where(s > 0.5, s, 0.5)
    v = np.ones_like(r)

    return h, s, v

def plot_motion_with_arrows(corres, dx, dy, colors, out_path, img_size=None, img=None):
    fig, ax = plt.subplots(1,1, figsize=(10,10)) #figsize=(15*0.39, 15*0.39))

    if img is not None:
        img = Image.blend(img, Image.new('RGB', img_size, 0), alpha=0.75)
        ax.imshow(img, alpha=0.7)
    else:
        ax.imshow(Image.new('RGB', img.size, 'white')) # white background

    for i, dxi in enumerate(dx):
        ax.arrow(corres[i,0], corres[i,1], dxi, dy[i], color=colors[i],
                 head_width=1.7, alpha=0.95, width=1)
        
    ax.axis('off')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def plot_corres(corres, img_size, out_paths, imgs=None, percentile=None, neigh=0, step_sparsify=1):
    # filter correspondences to avoid out-of-frame arrows
    corres = corres[np.where((corres[:, 0] >= neigh) &
                (corres[:, 0] <= img_size[1] - 1 - neigh) &
                (corres[:, 1] >= neigh) &
                (corres[:, 1] <= img_size[0] - 1 - neigh) &
                (corres[:, 2] >= neigh) &
                (corres[:, 2] <= img_size[1] - 1 - neigh) &
                (corres[:, 3] >= neigh) &
                (corres[:, 3] <= img_size[0] - 1 - neigh))[0]]

    corres = corres[np.arange(len(corres))[::step_sparsify]]

    dx = corres[:, 2] - corres[:, 0]
    dy = corres[:, 3] - corres[:, 1]
    r = np.sqrt(dx**2 + dy**2) # radii
    theta = np.arctan2(dy, dx) # angles in radians

    if percentile is not None:
        perc = np.percentile(r, percentile)
        idx = np.where(r > perc)[0]
    else:
        idx = np.arange(len(r))

    h, s, v = hsv_from_r_theta(r[idx], theta[idx], np.max(r))
    colors = np.array([hsv_to_rgb((h[i], s[i], v[i])) for i in range(len(h))])

    if imgs is None:
        imgs = [None]

    for i, img in enumerate(imgs):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        plot_motion_with_arrows(corres[idx], dx[idx], dy[idx], colors, out_paths[i],
                                img_size, img)
