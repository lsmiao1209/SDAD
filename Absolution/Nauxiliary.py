import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Absolution.Diffusion import Diffusion
from Tool.utils import *

def train(args):
    test_x = args.test_x
    args.test_x = test_x
    train_x = torch.tensor(args.train_x).float()
    train_y = torch.tensor(args.train_y).float()
    Diffusion(args, train_x, train_y)

def launch_DAD(dataname,train_x, train_y, test_x, test_y, T, lamda):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Parameters required
    args.nonauxiliary = True

    args.num_class = 1
    # Clustering learning rate
    args.lr_cluster = 0.01

    # dataname
    args.dataname = dataname

    # train dataset
    args.train_x = train_x
    args.train_y = train_y

    # test dataset
    args.test_x = test_x
    args.test_y = test_y

    # auxiliary learning module
    args.feature_dimension = args.train_x.shape[1]
    args.embedded_dimension = 24 if args.feature_dimension < 100 else 64

    #diffusion model parameters
    args.units = 500
    args.num_steps = T
    args.lamda = lamda
    args.device = "cuda"
    args.betas = get_named_beta_schedule("cosine", args.num_steps)
    args.betas = args.betas.to(args.device)

    args.alphas = 1 - args.betas
    args.alphas_bar = torch.cumprod(args.alphas, 0)
    args.alphas_bar_sqrt = torch.sqrt(args.alphas_bar)
    args.one_minus_alphas_bar_sqrt = torch.sqrt(1 - args.alphas_bar)

    # training
    train(args)


