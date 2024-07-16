import numpy as np
import torch
import torch.nn.functional as F

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

def loss_function_multiseed(all_r, shift, origin_class, target_class, encoder, device):
    loss = torch.sum(torch.stack([loss_function(r, shift, origin_class, target_class, encoder, device) for r in all_r]))
    
    return loss
        
def loss_function(r, shift, origin_class, target_class, encoder, device, lambda_):
    r_shifted = r - shift
    pred = encoder.fc(r_shifted).to(device)  # torch.relu(r_shifted)).to(device)
    # pred = torch.softmax(pred, dim=0) # probability version
    # lambda_ = 0.7
    loss = -pred[target_class] + lambda_*pred[origin_class] # + 0.1*torch.sum(torch.relu(-r_shifted)) # punish negative values of r
    # loss = -pred[target_class]
    return loss

def loss_function_v2(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_):
    r_shifted = r-shift
    pred = encoder.fc(r_shifted).to(device)
    coef = torch.tensor(mapping_nw.coef_).to(device)
    intercept = torch.tensor(mapping_nw.intercept_).to(device)
    w_cycled = (torch.matmul(r_shifted, coef.T) + intercept).to(device)
    loss = -pred[target_class] + lambda_*pred[origin_class] - 10*torch.cosine_similarity(w[0], w_cycled, dim=-1)

    return loss

def loss_function_v3(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_):
    r_shifted = r-shift
    pred = encoder.fc(r_shifted).to(device)
    loss = -pred[target_class] + lambda_*pred[origin_class] - 10*torch.cosine_similarity(r_shifted, r, dim=-1)

    return loss

def loss_function_v4(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_):
    r_shifted = r-shift
    pred = encoder.fc(r_shifted).to(device)
    pred = torch.softmax(pred, dim=0)
    loss = -pred[target_class] # - 0.001*torch.cosine_similarity(r_shifted, r, dim=-1)

    return loss

def optimize_shift(all_r, r, w, origin_class, target_class, eps, 
                   lambda_init, lr, encoder, mapping_nw, device, 
                   manualseed=0, loss_type='v2'):
    """
    Find the optimal r' that maximizes the probability of target_class, knowing that r maximizes the probability of origin_class.
    pred_proba = softmax(rW+b)
    
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
    stylemc_shift = np.load('./stylemc_mean_change/mean_diff_n=101_chihuahua-dog.npy', allow_pickle=True)
    # torch generator
    torch.manual_seed(manualseed)
    shift = torch.rand(2048, requires_grad=True).to(device)
    r = torch.tensor(r).to(device)
    w = torch.tensor(w).to(device)
    all_r = torch.tensor(all_r).to(device)
    proba_target = torch.softmax(encoder.fc(torch.relu(r)).to(device), dim=0)[target_class].item()
    proba_origin = torch.softmax(encoder.fc(torch.relu(r)).to(device), dim=0)[origin_class].item()

    opt = torch.optim.AdamW([shift], lr=0.5, betas=(0.9, 0.99))
    lambda_ = lambda_init
    rshift_across_opt = []

    while 1-proba_target > eps: #or proba_origin > 1e-6:# and steps < 1000:
        if loss_type == 'v2':
            loss = loss_function_v2(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_)
        elif loss_type == 'v3':
            loss = loss_function_v3(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_)
        elif loss_type == 'v1':
            loss = loss_function(r, shift, origin_class, target_class, encoder, device, lambda_)
        elif loss_type == 'watcher':
            loss = watcher_loss(all_r, r, shift, target_class, encoder, device, lambda_=lambda_)
        elif loss_type == 'v4':
            loss = loss_function_v4(r, w, shift, origin_class, target_class, encoder, mapping_nw, device, lambda_)
        else:
            raise ValueError('Unknown loss type')
        loss.backward()
        opt.step()
        # # shift.grad.data.zero_()
        # with torch.no_grad():
        #     shift -= shift.grad * lr
        #     shift.grad.zero_()

        r_shifted = r - shift
        # if proba_target > 0.99:
        #     lambda_ *= 0.9

        pred_shifted = encoder.fc(torch.relu(r_shifted)).to(device)
        pred_shifted = torch.softmax(pred_shifted, dim=0)
        proba_target = pred_shifted[target_class].item()
        proba_origin = pred_shifted[origin_class].item()
        steps += 1
        rshift_across_opt.append(r_shifted.detach().cpu().numpy())
        # cosine distance between r and r_shifted
        # diff_v2 = np.dot(stylemc_shift, shift.detach().cpu().numpy()) / (np.linalg.norm(stylemc_shift) * np.linalg.norm(shift.detach().cpu().numpy()))
        # print(diff_v2)
        # diff_v2 = np.linalg.norm(stylemc_shift - shift.detach().cpu().numpy())
        # print(diff_v2)

    print(f'Iteration {steps}:')
    print(f'Proba origin: {pred_shifted[origin_class].item()}')
    print(f'Proba target: {pred_shifted[target_class].item()}')
    rshift_across_opt = np.array(rshift_across_opt)
    return shift.detach().cpu().numpy(), rshift_across_opt

def optimize_shift_multiseed(rs, origin_class, target_class, encoder, device):
    stylemc_shift = np.load('./stylemc_mean_change/mean_diff_n=101_chihuahua-dog.npy', allow_pickle=True)
    # # r_shifted_v2 = np.where(r_shifted_v2 > 0, r_shifted_v2, 0)

    steps = 0
    rs = torch.tensor(rs).to(device)
    shift = torch.zeros(2048, requires_grad=True).to(device)
    mean_ptarget = np.mean([torch.softmax(encoder.fc(torch.relu(r - shift)).to(device), dim=0)[target_class].item() for r in rs])

    while mean_ptarget < .99: #and steps < 1000:
        # mean of tensors
        loss = loss_function_multiseed(rs, shift, origin_class, target_class, encoder, device)
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