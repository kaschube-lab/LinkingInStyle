import os
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet

from src.metadata import CLASSES_DATASET

def train_linear(feats, ws, model_type='LR'):
    """
    Train a linear regression to link from encoder space to W-space in GAN
    """
    if model_type == 'LR':
      model  = LinearRegression()
    elif model_type == 'Ridge':
      model = Ridge()
    elif model_type == 'ElasticNet':
      model = ElasticNet()
    else:
       raise NotImplementedError(f"Model type {model_type} not implemented")
    
    model.fit(feats, ws)
    mse = np.mean((ws - model.predict(feats))**2)

    return model, mse

def train_linking_nw(dataset_name, classifier_name, model_type='LR', n_samples=5000, gen_seed=0, truncation=0.7):
    nw_path = os.path.join('models', 'lnw', f'LinkingNW_{model_type}_{classifier_name}_{dataset_name}.pkl')

    # load training data
    ws, rs = [], []
    for c in CLASSES_DATASET[dataset_name]:
        data_path = os.path.join('data', 'lnw', 'train', dataset_name, f'data_{dataset_name}_{classifier_name}_N={n_samples}_seed={gen_seed}_trunc={truncation}_class={c}.npz')
        assert os.path.exists(data_path), f"Data file not found: {data_path}"

        data = np.load(data_path, allow_pickle=True)
        ws += list(data['ws'][:,0,:])
        rs += list(data['rs'])
    
    ws = np.array(ws)
    rs = np.array(rs)

    # shuffle
    idx = np.arange(len(ws))
    np.random.shuffle(idx)
    ws = ws[idx]
    rs = rs[idx]

    trained_nw, mse = train_linear(rs, ws, model_type=model_type)
    print(f'MSE: {mse}')
    with open(nw_path, 'wb') as f:
        pickle.dump(trained_nw, f)





