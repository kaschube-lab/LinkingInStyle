import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys

sys.path.append('../')
from generate_data import val_transform
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# class CounterfactualOptimizer:
#     def __init__(self, G, encoder, mapping_nw, device):

def w512_to_img(G, w):
    w = w.repeat(32)
    w = torch.reshape(w, (1, 32, 512))
    synth_image = G.synthesis(w, noise_mode='none', force_fp32=True)
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.clamp(0, 255).to(torch.uint8)[0]
    synth_image = transforms.ToPILImage()(synth_image)

    return synth_image

def r_to_w(r, mapping_nw, device):
    coef = torch.tensor(mapping_nw.coef_).to(device)
    intercept = torch.tensor(mapping_nw.intercept_).to(device)
    w_cycled = (torch.matmul(r, coef.T) + intercept).to(device)

    return w_cycled

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def watcher_loss(all_r, r, shift, target_class, encoder, device, lambda_=0.1):
    # median absolute deviation of each feature j over the dataset
    mad = torch.median(torch.stack([torch.abs(all_r[i] - torch.median(all_r[i])) for i in range(all_r.shape[0])]), dim=0)[0]

    r_shifted = r - shift
    pred = encoder.fc(torch.relu(r_shifted)).to(device)
    # proba_target = torch.softmax(pred, dim=0)[target_class]
    # manhattan distance of r and r_shifted, weighted by the median absolute deviation
    x_dist = torch.sum(torch.stack([torch.abs(r[j]-r_shifted[j]).div(mad[j]) for j in range(r.shape[0])]))
    # x_dist = spherical_dist_loss(r, r_shifted)
    # difference between the probability of the target class and desired probability
    prob_dist = (torch.softmax(pred, dim=0)[target_class] - torch.tensor(0.99).to(device)).pow(2).mul(lambda_)
    # prob_dist = -pred[target_class]

    loss = -prob_dist + x_dist
    return loss

def loss_fn_multiseed(all_r, shift, origin_class, target_class, encoder, device):
    loss = torch.sum(torch.stack([loss_fn(r, shift, origin_class, target_class, encoder, device) for r in all_r]))
    
    return loss
        
def loss_fn(r, shift, origin_class, target_class, encoder, device, lambda_):
    r_shifted = r - shift
    pred = encoder.fc(r_shifted).to(device)  # torch.relu(r_shifted)).to(device)
    # pred = torch.softmax(pred, dim=0) # probability version
    # lambda_ = 0.7
    loss = -pred[target_class] + lambda_*pred[origin_class] # + 0.1*torch.sum(torch.relu(-r_shifted)) # punish negative values of r
    # loss = -pred[target_class]
    return loss

def loss_fn_v2(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_):
    r_shifted = r-shift
    pred = encoder.fc(r_shifted).to(device)
    w_cycled = r_to_w(r_shifted, mapping_nw, device)
    loss = -pred[target_class] + lambda_*pred[origin_class] - 10*torch.cosine_similarity(w[0], w_cycled, dim=-1)

    return loss

def loss_fn_v3(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_):
    r_shifted = r-shift
    pred = encoder.fc(r_shifted).to(device)
    loss = -pred[target_class] + lambda_*pred[origin_class] - 10*torch.cosine_similarity(r_shifted, r, dim=-1)

    return loss

def loss_fn_v4(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_):
    r_shifted = r-shift
    pred = encoder.fc(r_shifted).to(device)
    pred = torch.softmax(pred, dim=0)
    loss = -pred[target_class] #- lambda_*torch.cosine_similarity(r_shifted, r, dim=-1)
    # print(loss)

    return loss

def loss_fn_v5(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_):
    r_shifted = r-shift
    pred = encoder.fc(r_shifted).to(device)
    pred = torch.softmax(pred, dim=0)
    w_cycled = r_to_w(r, mapping_nw, device)
    w_shifted_cycled = r_to_w(r_shifted, mapping_nw, device)
    loss = -pred[target_class] - lambda_*torch.cosine_similarity(w_cycled, w_shifted_cycled, dim=-1) # lambda_*

    return loss

def loss_fn_v6(G, r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_):
    r_shifted = r-shift
    coef = torch.tensor(mapping_nw.coef_).to(device)
    intercept = torch.tensor(mapping_nw.intercept_).to(device)
    w_cycled = (torch.matmul(r_shifted, coef.T) + intercept).to(device)
    img = w512_to_img(G, w_cycled)
    img = val_transform(img)
    pred = encoder(img[None, ...].to(device))
    proba = torch.softmax(pred, dim=1)[0]

    sim_w = torch.cosine_similarity(w[0], w_cycled, dim=-1)
    # print(sim_w)
    loss = -proba[target_class] - sim_w.mul(lambda_)
    print(proba)
    print('Loss: ', loss.item())

    return loss

def loss_fn_v7(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_):
    r_shifted = r-shift
    pred = encoder.fc(r_shifted).to(device)
    w_shifted_cycled = r_to_w(r_shifted, mapping_nw, device)
    w_cycled = r_to_w(r, mapping_nw, device)
    loss = -pred[target_class] + lambda_*pred[origin_class] - 10*torch.cosine_similarity(w_cycled, w_shifted_cycled, dim=-1)

    return loss

def optimize_shift(G, all_r, r, w, origin_class, target_class, 
                   lambda_init, lr, encoder, mapping_nw, device, 
                   manualseed=0, loss_type='v2'):
    """
    Find the optimal r' that maximizes the probability of target_class, knowing that r maximizes the probability of origin_class.
    
    Parameters
    ----------
    r : np.array of size 2048
        The original activation vector.
    weights : np.array of size 1000x2048
        The weights of the last layer of the encoder.
    bias : np.array of size 1000
        The bias of the last layer of the encoder.
    origin_class : int
        The class of the original activation vector.
    target_class : int
        The class of the target activation vector.
    pred_proba : np.array of size 1000
        The probability vector of the original activation vector.
    """
    steps = 0
    if manualseed is not None:
        torch.manual_seed(manualseed)
        shift = torch.rand(2048, requires_grad=True, device=device)
    else:
        shift = torch.zeros(2048, requires_grad=True, device=device)

    r_tensor = torch.tensor(r).to(device)
    w_tensor = torch.tensor(w).to(device)
    all_r_tensor= torch.tensor(all_r).to(device)
    # proba_target = torch.softmax(encoder.fc(torch.relu(r)).to(device), dim=0)[target_class].item()
    pred_class = origin_class

    # opt = torch.optim.AdamW([shift], lr=lr, betas=betas, weight_decay=0.25)
    lambda_ = lambda_init
    rshift_across_opt = []

    while pred_class != target_class:
        if loss_type == 'v2':
            loss = loss_fn_v2(r_tensor, w_tensor, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_)
        elif loss_type == 'v3':
            loss = loss_fn_v3(r_tensor, w_tensor, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_)
        elif loss_type == 'v1':
            loss = loss_fn(r_tensor, shift, origin_class, target_class, encoder, device, lambda_)
        elif loss_type == 'watcher':
            loss = watcher_loss(all_r_tensor, r_tensor, shift, target_class, encoder, device, lambda_)
        elif loss_type == 'v4':
            loss = loss_fn_v4(r_tensor, w_tensor, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_)
        elif loss_type == 'v5':
            loss = loss_fn_v5(r_tensor, w_tensor, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_)
        elif loss_type == 'v6':
            loss = loss_fn_v6(G, r_tensor, w_tensor, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_)
        elif loss_type == 'v7':
            loss = loss_fn_v7(r_tensor, w_tensor, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_)
        else:
            raise ValueError('Unknown loss type')
        loss.backward(retain_graph=True)
        # opt.step()
        with torch.no_grad():
            shift -= shift.grad * lr
            shift.grad.zero_()

        r_shifted = r_tensor - shift
        # lambda_ *= 0.9

        w_shifted = mapping_nw.predict([torch.relu(r_shifted).detach().cpu().numpy()])[0]
        w_shifted = torch.tensor(w_shifted).to(device)
        img = w512_to_img(G, w_shifted)
        img = val_transform(img)
        pred_shifted = encoder(img[None, ...].to(device))
        pred_shifted = torch.softmax(pred_shifted, dim=1)[0]
        proba_target = pred_shifted[target_class].item()
        proba_origin = pred_shifted[origin_class].item()
        pred_class = pred_shifted.argmax(dim=0).item()

        # pred_shifted = encoder.fc(torch.relu(r_shifted)).to(device)
        # pred_shifted = torch.softmax(pred_shifted, dim=0)
        # proba_target = pred_shifted[target_class].item()
        # proba_origin = pred_shifted[origin_class].item()
        steps += 1
        rshift_across_opt.append(r_shifted.detach().cpu().numpy())
        if steps%100==0:
            print(f'Iteration {steps}:')
            print(f'Proba origin: {pred_shifted[origin_class].item()}')
            print(f'Proba target: {pred_shifted[target_class].item()}')

        if steps > 2000:
            return None, None
        # if steps > 2000:
        #     manualseed = int(manualseed)+1 if manualseed is not None else 0
        #     return optimize_shift(G, all_r, r, w, origin_class, target_class, eps, 
        #            lambda_init, lr, encoder, mapping_nw, device, manualseed, loss_type)
    rshift_across_opt = np.array(rshift_across_opt)
    return shift.detach().cpu().numpy(), rshift_across_opt

def optimize_shift_multiseed(rs, origin_class, target_class, encoder, device):
    stylemc_shift = np.load('./stylemc_mean_change/mean_diff_n=101_chihuahua-dog.npy', allow_pickle=True)
    # # r_shifted_v2 = np.where(r_shifted_v2 > 0, r_shifted_v2, 0)

    steps = 0
    rs = torch.tensor(rs).to(device)
    shift = torch.zeros(2048, requires_grad=True, device=device)
    mean_ptarget = np.mean([torch.softmax(encoder.fc(torch.relu(r - shift)).to(device), dim=0)[target_class].item() for r in rs])

    while mean_ptarget < .99: #and steps < 1000:
        # mean of tensors
        loss = loss_fn_multiseed(rs, shift, origin_class, target_class, encoder, device)
        loss.backward()
        with torch.no_grad():
            shift -= shift.grad * 0.2
            shift.grad.zero_()

        mean_ptarget = np.mean([torch.softmax(encoder.fc(torch.relu(r - shift)).to(device), dim=0)[target_class].item() for r in rs])
        steps += 1
        # rs_shifted = [r - shift for r in rs]

        # pred_shifted = encoder.fc(torch.relu(r_shifted)).to(device)
        # pred_shifted = torch.softmax(pred_shifted, dim=0)
        # mean_ptarget = pred_shifted[target_class].item()
        # diff_v2 = np.dot(stylemc_shift, shift.detach().cpu().numpy()) / (np.linalg.norm(stylemc_shift) * np.linalg.norm(shift.detach().cpu().numpy()))
        # print(diff_v2)

        return shift.detach().cpu().numpy()