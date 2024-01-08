
import math
import os
import random

import matplotlib
import numpy as np
import scipy
import sklearn.metrics
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) #
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def getdata(dataname):

    data = np.load(f'./dataset/{dataname}.npz', allow_pickle=True)
    label = data['y'].astype('float32')
    data = data['X'].astype('float32')
    # Standardization
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # if the dataset is too large, subsampling for considering the computational cost
    if len(label) > 9000:
        idx_sample = np.random.choice(np.arange(len(label)), 5000, replace=False)
        data = data[idx_sample]
        label = label[idx_sample]

    normal_data = data[label == 0]
    normal_label = label[label == 0]
    anom_data = data[label == 1]
    anom_label = label[label == 1]

    test_ano = np.random.choice(np.arange(0, len(anom_data)), int(len(anom_data) * 0.2), replace=False)
    test_no = np.random.choice(np.arange(0, len(normal_data)), int(len(normal_data) * 0.2), replace=False)

    train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_no)

    train_x = normal_data[train_idx]
    train_y = normal_label[train_idx]

    test_x = np.concatenate((normal_data[test_no], anom_data[test_ano]))
    test_y = np.concatenate((normal_label[test_no], anom_label[test_ano]))

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y



def shuffle(X, Y):
    """
    Shuffle the datasets
    Args:
        X: input data
        Y: labels

    Returns: shuffled sets
    """
    random_index = np.random.permutation(X.shape[0])
    return X[random_index], Y[random_index]

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        return torch.linspace(1e-6, 0.02, num_diffusion_timesteps)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t+1e-10) / (1+1e-10) * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.02
                        ):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []

    for i in range(num_diffusion_timesteps):
        t1 = i / (num_diffusion_timesteps)
        t2 = (i + 1) / (num_diffusion_timesteps)
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)

def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    dr = tpr[right_index]
    far = fpr[right_index]
    return dr, far, best_th, right_index

def CalMetrics(test_y, error):
    auc = sklearn.metrics.roc_auc_score(test_y.cpu(), error.cpu())
    pr = sklearn.metrics.average_precision_score(test_y.cpu(), error.cpu())

    return auc, pr

def Metrics(test_y, error):
    auc = sklearn.metrics.roc_auc_score(test_y, error)
    pr = sklearn.metrics.average_precision_score(test_y, error)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_y, error, pos_label=1)
    dr, far, best_th, _ = get_err_threhold(fpr, tpr, thresholds)
    test_labels = np.where(error > best_th, 1, 0)
    f1 = sklearn.metrics.f1_score(test_y, test_labels)

    return auc, pr, f1

