import os
import argparse
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet

classes_dataset = {'dogs': [232,254,151,197,178],
                    'fungi': [992, 993, 994, 997],
                    'birds': [10, 12, 14, 92, 95, 96],
                    'cars': [468, 609, 627, 717, 779, 817]
                    }

def train_mapping(feats, ws, model_type='LR'):
    """
    Train a linear regression to map from encoder space to W-space in GAN
    """
    if model_type == 'LR':
      model  = LinearRegression()
    elif model_type == 'Ridge':
      model = Ridge()
    elif model_type == 'ElasticNet':
      model = ElasticNet()
    model.fit(feats, ws)
    mse = np.mean((ws - model.predict(feats))**2)
    print(f'Mean error in feature space: {mse}')

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train mapping network')
    parser.add_argument('--encoder', type=str, default='rnet50', 
                        help='Encoder to use')
    parser.add_argument('--dataset', choices=['dogs', 'fungi', 'birds', 'cars'], 
                        required=True, help='Dataset to use')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of samples generated per class')
    args = parser.parse_args()
    print(args)

    classes = classes_dataset[args.dataset]

    c_ref = ','.join([str(c) for c in classes])
    lr_file_name = f'MappingNW_LR_{args.encoder}_{c_ref}.pkl'
    save_path='./models'

    # load data
    ws, rs = [], []
    for c in classes:
        data_fp = f'./data/data_{args.encoder}_{args.n_samples}N_{c}.npz'
        data = np.load(data_fp, allow_pickle=True)
        ws += list(data['ws'][:,0,:])
        rs += list(data['rs'])
    ws = np.array(ws)
    rs = np.array(rs)

    # shuffle data just in case
    idx = np.arange(len(ws))
    np.random.shuffle(idx)
    ws = ws[idx]
    rs = rs[idx]

    trained_nw = train_mapping(rs, ws, model_type='LR')
    with open(os.path.join(save_path, lr_file_name), 'wb') as file:
        pickle.dump(trained_nw, file)