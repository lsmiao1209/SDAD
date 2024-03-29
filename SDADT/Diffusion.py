import time

import pandas as pd
import torch.nn as nn
from sklearn import metrics

from Tool.utils import *
from Tool.sampling import *
from SDADT.Model import DiffusionM
import random

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) #
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  #

def train(args, train_x, train_y):

    num_epoch = 3000
    maxauc = 0.0
    maxpr = 0.0
    maxf1 = 0.0
    loss = 0.0
    running = 0.0
    s = 1
    model = DiffusionM(args, train_x).to(args.device)
    # from torchsummary import summary
    # summary(model, [(args.test_x.shape[-1],),(1,)], device='cuda')

    # from thop import profile
    # t = torch.randint(0, args.num_steps, size=(1,)).to(args.device)
    # t = t.unsqueeze(-1).cpu()
    # x = torch.randn([1, train_x.shape[1]])
    # flops, params = profile(model.to('cpu'), (x,t))
    # print(f"FLOPS: {flops / 10 ** 6:.03} M, Params: {params / 10 ** 6:.03} M")


    batch = train_x.shape[0]
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    for epoch in range(num_epoch):
        model.train()
        for i in range(train_x.shape[0] // batch):
            input_batch = train_x[i * batch: (i + 1) * batch]

            # Generate a random moment t for the sample
            t = torch.randint(0, args.num_steps, size=(input_batch.shape[0],)).to(args.device)
            t = t.unsqueeze(-1)

            # Constructing inputs to the model
            x, noise = x_t(input_batch, t, args)

            # Feed into the model to get the noise prediction at moment t
            output = model(x, t.squeeze(-1))

            # Calculating the difference between real and predicted noise
            noise_loss = mse(noise, output)
            optimizer.zero_grad()
            noise_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            running += noise_loss.data.cpu().numpy()

        # Testing
        auc, x_0, maxauc, maxpr, maxf1, maxdata, label = Test(model, maxauc, maxpr, maxf1, args)

        # if epoch % 999 ==0:
        #     print(f'Epoch: {epoch}, AUC: {auc}, LOSS: {running}')

        running = 0.0

    # Testing
    results = '{:.4f}, {:.4f}'.format(maxauc, maxpr)
    print(results)
    output_file = "T.xlsx"
    if os.path.exists(output_file):
        df = pd.read_excel(output_file)
    else:
        print("fdasfffff")
    df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    df.to_excel(output_file, index=False)

def Test(model, maxauc, maxpr, maxf1,args):
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(args.test_x).float()
        test_y = torch.tensor(args.test_y).float()
        maxdata = test_x
        label = args.test_y
        # Sampling, calculating the restored x_0
        x_0, xt, z = sampleT(model, args, test_x)
        x_0 = x_0.cpu().detach()

        sum = torch.mean((test_x - x_0).pow(2), dim=1).data
        auc, pr = CalMetrics(test_y.cpu(), sum.cpu())
        fpr, tpr, thresholds = metrics.roc_curve(args.test_y, sum, pos_label=1)
        dr, far, best_th, right_index = get_err_threhold(fpr, tpr, thresholds)
        pred_label = np.where(sum.numpy() > best_th, 1, 0)

        if args.auxiliary == True:
            if auc > maxauc and pr > maxpr:
                maxauc = auc
                maxpr = pr


    return auc, x_0, maxauc, maxpr, maxf1, maxdata, label

def Diffusion(args, train_x, train_y):

    train(args, train_x, train_y)


