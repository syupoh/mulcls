import argparse
import os
import numpy as np
import math
import pdb 
from tensorboardX import SummaryWriter
from itertools import chain

import torchvision.transforms as transforms
from torchvision.utils import save_image
from util.sampler import InfiniteSampler

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
from datetime import datetime

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='acgan_tran')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--gpu", type=int, default=3)
parser.add_argument('--norm', type=bool, default=True)
parser.add_argument('--digitroot', type=str, default='~/dataset/digits/')
parser.add_argument('--model', type=str, default='mnist_mnist')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--display', type=int, default=10)
opt = parser.parse_args()
print(opt)

now = datetime.now()
curtime = now.isoformat() 
modelname = '{0}_{1}'.format(opt.prefix, opt.model)
run_dir = "runs/{0}_{1}_ongoing".format(curtime[0:16], modelname)
writer = SummaryWriter(run_dir)

cuda = True if torch.cuda.is_available() else False
if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)

# Configure data loader

import utils
trainset, trainset2, testset = utils.load_data(opt=opt)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, drop_last=True, sampler=InfiniteSampler(len(trainset))) # model
train_loader2 = torch.utils.data.DataLoader(trainset2, batch_size=opt.batch_size, drop_last=True, sampler=InfiniteSampler(len(trainset2))) # model
test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, drop_last=True) # model

n_sample = max(len(trainset), len(trainset2))
iter_per_epoch = n_sample // opt.batch_size + 1

src_train_iter = iter(train_loader)
tgt_train_iter = iter(train_loader2)

if opt.norm == True:
    X_min = -1 # 0.5 mormalize 는 0~1
    X_max = 1
else:
    X_min = trainset.data.min()
    X_max = trainset.data.max()

# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "~/dataset/digits/MNIST_data/",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

for X, Y in train_loader: 
    res_x = X.shape[-1]
    break

for X, Y in train_loader2: 
    res_y = X.shape[-1]
    break

modelsplit = opt.model.split('_')
if (modelsplit[0] == 'mnist' or modelsplit[0] == 'usps') and modelsplit[1] != 'svhn':
    n_c_in = 1 # number of color channels
else:
    n_c_in = 3 # number of color channels
    
if (modelsplit[1] == 'mnist' or modelsplit[1] == 'usps') and modelsplit[0] != 'svhn':
    n_c_out = 1 # number of color channels
else:
    n_c_out = 3 # number of color channels
    opt.channels = 3

n_ch = 64
n_hidden = 5
n_resblock = 4

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class GenResBlock(nn.Module):
    def __init__(self, n_out_ch):
        super(GenResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_out_ch, n_out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(n_out_ch, n_out_ch, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(n_out_ch)
        self.bn2 = nn.BatchNorm2d(n_out_ch)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        return x + self.bn2(self.conv2(h))


class Generator(nn.Module):
    def __init__(self, n_hidden, n_resblock, n_ch, res, n_c_in, n_c_out):
        super(Generator, self).__init__()
        self.n_resblock = n_resblock
        self.n_hidden = n_hidden
        self.res = res
        self.fc = nn.Linear(n_hidden, self.res * self.res)
        
        self.conv1 = nn.Conv2d(1 + n_c_in, n_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_ch)
        
        for i in range(1, self.n_resblock + 1):
            setattr(self, 'block{:d}'.format(i), GenResBlock(n_ch))
        self.conv2 = nn.Conv2d(n_ch, n_c_out, 3, padding=1)

    def gen_noise(self, batchsize):
        return torch.randn(batchsize, self.n_hidden)  # z_{i} ~ N(0, 1)

    def __call__(self, x):
        z = self.gen_noise(x.size(0)).to(x.device)
        h = torch.cat(
            (x, F.relu(self.fc(z)).view(-1, 1, self.res, self.res)),
            dim=1)
        h = F.relu(self.bn1(self.conv1(h)))
        for i in range(1, self.n_resblock + 1):
            h = getattr(self, 'block{:d}'.format(i))(h)
        return torch.tanh(self.conv2(h))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

loss_CE = torch.nn.CrossEntropyLoss().cuda()
loss_KLD = torch.nn.KLDivLoss(reduction='batchmean').cuda()
# loss_LS = GANLoss(device, use_lsgan=True)



# Initialize generator and discriminator


gen_st = Generator(n_hidden=n_hidden, n_resblock=n_resblock, \
    n_ch=n_ch, res=res_x, n_c_in=n_c_in, n_c_out=n_c_out).cuda()
    
gen_ts = Generator(n_hidden=n_hidden, n_resblock=n_resblock, \
    n_ch=n_ch, res=res_y, n_c_in=n_c_out, n_c_out=n_c_in).cuda()
    
discriminator = Discriminator()


if cuda:
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
gen_st.apply(weights_init_normal)
gen_ts.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers

config2 = {'lr': opt.lr, 'weight_decay': opt.weight_decay, 'betas': (0.5, 0.999)}

opt_gen_st = torch.optim.Adam(gen_st.parameters(), **config2)
opt_gen_ts = torch.optim.Adam(gen_ts.parameters(), **config2)
# opt_gen = torch.optim.Adam(
#     chain(gen_st.parameters(), gen_ts.parameters()), **config2) 
        
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


from torchvision.utils import make_grid

# ----------
#  Training
# ----------
print('')
niter = 0
epoch = 0
while True:
    niter += 1
    src_x, src_y = next(src_train_iter)
    tgt_x, tgt_y = next(tgt_train_iter)
    src_x = src_x.cuda()
    src_y = src_y.cuda()
    tgt_x = tgt_x.cuda()

    # Adversarial ground truths
    batch_size = src_x.shape[0]
    valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

    
    # Networks
    fake_tgt_x = gen_st(src_x)
    fake_src_x = gen_ts(tgt_x)
    fake_back_src_x = gen_ts(fake_tgt_x)

    # -----------------
    #  Train Generator
    # -----------------

    opt_gen_st.zero_grad()

    # Loss measures generator's ability to fool the discriminator
    validity, pred_label = discriminator(fake_tgt_x)
    g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, src_y))

    g_loss.backward()
    opt_gen_st.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------
    # pdb.set_trace()
    optimizer_D.zero_grad()

    # Loss for real images
    # real_pred, real_aux = discriminator(src_x)
    # d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, src_y)) / 2

    real_pred, real_aux = discriminator(src_x)
    d_real_loss = 0.5 * (auxiliary_loss(real_aux, src_y)) 

    tgt_pred, tgt_aux = discriminator(tgt_x)
    d_real_loss += 0.5 * (adversarial_loss(tgt_pred, valid)) 

    # Loss for fake images
    fake_pred, fake_aux = discriminator(fake_tgt_x.detach())
    d_fake_loss = 0.5 * (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, src_y)) 

    # Total discriminator loss
    d_loss = 0.5 * (d_real_loss + d_fake_loss) 

    # Calculate discriminator accuracy
    # pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
    # gt = np.concatenate([src_y.data.cpu().numpy(), src_y.data.cpu().numpy()], axis=0)
    # d_acc = np.mean(np.argmax(pred, axis=1) == gt)

    d_acc_src = np.mean(np.argmax(real_aux.data.cpu().numpy(), axis=1) == src_y.data.cpu().numpy())
    d_acc_tgt = np.mean(np.argmax(fake_aux.data.cpu().numpy(), axis=1) == src_y.data.cpu().numpy())
    d_loss.backward()
    optimizer_D.step()

    print('epoch {0} ({1}/{2}) '.format(epoch, (niter % iter_per_epoch), iter_per_epoch ) \
    + 'D loss {0:02.4f}, acc_s {1:02.4f}, acc_t {2:02.4f}, G loss {3:02.4f}'.format(d_loss.item(), 100 * d_acc_src, 100*d_acc_tgt, g_loss.item()), end='\r')

    writer.add_scalar('{0}/d_loss'.format(opt.prefix), d_loss.item(), niter)
    writer.add_scalar('{0}/d_acc_src'.format(opt.prefix), 100 * d_acc_src.item(), niter)
    writer.add_scalar('{0}/d_acc_tgt'.format(opt.prefix), 100 * d_acc_tgt.item(), niter)
    writer.add_scalar('{0}/g_loss'.format(opt.prefix), g_loss.item(), niter)

    if niter % iter_per_epoch == 0 and niter > 0:
        with torch.no_grad(): 
            epoch = niter // iter_per_epoch

            avgaccuracy1 = 0
            n = 0
            nagree = 0
        
            for X, Y in test_loader: 
                n += X.size()[0]
                X_test = X.cuda() 
                Y_test = Y.cuda() 

                val, prediction1 = discriminator(X_test) #
                predicted_classes1 = torch.argmax(prediction1, 1) 
                correct_count1 = (predicted_classes1 == Y_test) 
                testaccuracy1 = correct_count1.float().sum()
                avgaccuracy1 += testaccuracy1

            avgaccuracy1 = (avgaccuracy1/n) *100
            writer.add_scalar('{0}/d_acc_test'.format(opt.prefix), avgaccuracy1, niter)
                

    if niter % (opt.sample_interval) == 0 :
        data_grid = []
        for x in [src_x, fake_tgt_x, tgt_x]:
            x = x[0:opt.display].to(torch.device('cpu'))
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)  # grayscale2rgb
            data_grid.append(x)
        grid = make_grid(torch.cat(tuple(data_grid), dim=0),
                        normalize=True, range=(X_min, X_max), nrow=opt.display) # for SVHN?
        writer.add_image('generated_{0}'.format(opt.prefix), grid, niter)

    if epoch >= opt.n_epochs:
        print('')
        print('train complete')
        os.rename(run_dir, run_dir[:-8])
        break

