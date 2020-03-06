from sklearn.manifold import TSNE
from datetime import datetime
from tensorboardX import SummaryWriter
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb 
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
from itertools import chain
from util.sampler import InfiniteSampler

import utils
import Networks2 as net

parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', type=int, default='3')
parser.add_argument('--lr_decay', type=int, default='100')
parser.add_argument('--digitroot', type=str, default='~/dataset/digits/')
parser.add_argument('--prefix', type=str, default='bitranslation')
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2048, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--gpu", type=int, default=3)
parser.add_argument('--model', type=str, default='mnist_svhn')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--cla_plus_weight', type=float, default=3e-1)
parser.add_argument('--cyc_loss_weight',type=float,default=0.01)
parser.add_argument('--weight_in_loss_g',type=str,default='1,0.01,0.1,0.1')
parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')
parser.add_argument('--random', type=bool, default=False, help='whether to use random')
parser.add_argument('--norm', type=bool, default=True)
parser.add_argument('--modelload', type=str, default=None)
opt = parser.parse_args()

torch.cuda.set_device(opt.gpu)
### Model
input_size = 512
num_feature = 1024
class_num = 10
gen_st = net.Generator_digits(input_size, num_feature).cuda()
gen_ts = net.Generator_digits(input_size, num_feature).cuda()
D_s = net.Discriminator_digits().cuda()
D_t = net.Discriminator_digits().cuda()
model = net.DTN().cuda()

classifier1 = net.Classifier(512, class_num)
classifier1 = classifier1.cuda()
classifier1_optim = torch.optim.Adam(classifier1.parameters(), lr=0.0003)


### Loss & Optimizers
# Loss functions
adversarial_loss = torch.nn.BCELoss()
loss_CE = torch.nn.CrossEntropyLoss().cuda()
criterion_CE = torch.nn.CrossEntropyLoss().cuda()
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_Sem = torch.nn.L1Loss()
criterion_percep = torch.nn.MSELoss()

optimizer_G = torch.optim.Adam(chain(gen_st.parameters(), gen_ts.parameters()), lr=0.0003)
optimizer_D_s = torch.optim.Adam(D_s.parameters(), lr=0.0003)
optimizer_D_t = torch.optim.Adam(D_t.parameters(), lr=0.0003)
optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
# optimizer_ad = torch.optim.SGD(ad_net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)


### Data_load
trainset, trainset2, testset = utils.load_data(opt=opt)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, drop_last=True, sampler=InfiniteSampler(len(trainset))) # model
train_loader2 = torch.utils.data.DataLoader(trainset2, batch_size=opt.batch_size, drop_last=True, sampler=InfiniteSampler(len(trainset2))) # model
test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, drop_last=True) # model

n_sample = max(len(trainset), len(trainset2))
iter_per_epoch = n_sample // opt.batch_size + 1

src_train_iter = iter(train_loader)
tgt_train_iter = iter(train_loader2)

### SummaryWriter
now = datetime.now()
curtime = now.isoformat() 
modelname = '{0}_{1}_{2}_{3}_{4:0.3f}_{5:0.1f}_{6}_{7}'.format(
    opt.prefix, opt.model, opt.lr, opt.weight_in_loss_g, opt.cyc_loss_weight, opt.cla_plus_weight, opt.start_epoch, opt.lr_decay)
run_dir = "runs/{0}_{1}_ongoing".format(curtime[0:16], modelname)
writer = SummaryWriter(run_dir)


### Looping
niter = 0
while True:
    niter += 1
    x_s, y_s = next(src_train_iter)
    x_t, tgt_y = next(tgt_train_iter)
    x_s = x_s.cuda()
    y_s = y_s.cuda()
    x_t = x_t.cuda()

    optimizer.zero_grad()
    # optimizer_ad.zero_grad()

    ########### Networks Forward Propagation
    f_s, p_s = model(x_s)
    f_t, p_t = model(x_t)
    features = torch.cat((f_s, f_t), dim=0)
    outputs = torch.cat((p_s, p_t), dim=0)
    loss = nn.CrossEntropyLoss()(outputs.narrow(0, 0, x_s.size(0)), y_s)

    ### TSNE
    tsne_model = TSNE(learning_rate=100)
    transformed = tsne_model.fit_transform(f_s.detach().cpu())

    xs = transformed[:, 0]
    ys = transformed[:, 1]

    fig = plt.figure()
    plt.scatter(xs, ys, c=y_s.cpu())
    # fig.savefig('f_s_tsne.png')



    writer.add_figure('f_s_tsne', fig, niter)
    

    # Identity loss
    same_t = gen_st(f_t)
    loss_identity_t = criterion_identity(same_t, f_t)
    same_s = gen_ts(f_s)
    loss_identity_s = criterion_identity(same_s, f_s)


    # Gan loss
    f_st = gen_st(f_s)
    pred_f_st, aux_f_st = D_t(f_st)
    loss_G_s2t = criterion_GAN(pred_f_st, y_s.float())
    
    f_ts = gen_ts(f_t)
    pred_f_ts, aux_f_ts = D_s(f_ts)
    loss_G_t2s = criterion_GAN(pred_f_ts, y_s.float())

    # cycle loss
    f_sts = gen_ts(f_st)
    loss_cycle_sts = criterion_cycle(f_sts, f_s)   
    
    f_tst = gen_st(f_ts)
    loss_cycle_tst = criterion_cycle(f_tst, f_t)
    
    # softmax
    outputs_fake = classifier1(f_st.detach())

