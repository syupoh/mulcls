import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from itertools import chain

import argparse
import os
import numpy as np
import math
import pdb 

from datetime import datetime
from util.sampler import InfiniteSampler
from utils import ReplayBuffer
import Networks2 as net


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--digitroot', type=str, default='~/dataset/digits/')
parser.add_argument('--prefix', type=str, default='bitranslation')
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
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
opt = parser.parse_args()

now = datetime.now()
curtime = now.isoformat() 
modelname = '{0}_{1}_{2}_{3:0.2f}'.format(
    opt.prefix, opt.model, opt.weight_in_loss_g, opt.cyc_loss_weight)
run_dir = "runs/{0}_{1}_ongoing".format(curtime[0:16], modelname)
writer = SummaryWriter(run_dir)


prompt = ''
prompt += ('====================================\n')
prompt += run_dir + '\n'
for arg in vars(opt):
    prompt = '{0}{1} : {2}\n'.format(prompt, arg, getattr(opt, arg))
prompt += ('====================================\n')
print(prompt, end='')

cuda = False
if torch.cuda.is_available():
    cuda = True
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{0}'.format(opt.gpu))
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

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-1*entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 


# Loss functions
adversarial_loss = torch.nn.BCELoss()

loss_CE = torch.nn.CrossEntropyLoss().cuda()
# loss_KLD = torch.nn.KLDivLoss(reduction='batchmean').cuda()
# loss_LS = GANLoss(device, use_lsgan=True)

### Model
input_size = 512
num_feature = 1024
class_num = 10
gen_st = net.Generator_digits(input_size, num_feature).cuda()
gen_ts = net.Generator_digits(input_size, num_feature).cuda()
D_s = net.Discriminator_digits().cuda()
D_t = net.Discriminator_digits().cuda()
model = net.DTN().cuda()

if opt.random:
    random_layer = net.RandomLayer([model.output_num(), class_num], 500)
    ad_net = net.AdversarialNetwork(500, 500).cuda()
    random_layer.cuda()
else:
    random_layer = None
    ad_net = net.AdversarialNetwork(model.output_num() * class_num, 500).cuda()

classifier1 = net.Classifier(512, class_num)
classifier1 = classifier1.cuda()
classifier1_optim = torch.optim.Adam(classifier1.parameters(), lr=0.0003)

f_ts_buffer = ReplayBuffer()
f_st_buffer = ReplayBuffer()

### Loss & Optimizers
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_Sem = torch.nn.L1Loss()
criterion_percep = torch.nn.MSELoss()

vgg_model = net.VGG16()
if torch.cuda.is_available():
    vgg_model.cuda()

vgg_model.eval()

optimizer_G = torch.optim.Adam(chain(gen_st.parameters(), gen_ts.parameters()), lr=0.0003)
optimizer_D_s = torch.optim.Adam(D_s.parameters(), lr=0.0003)
optimizer_D_t = torch.optim.Adam(D_t.parameters(), lr=0.0003)
optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
optimizer_ad = torch.optim.SGD(ad_net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)

### Initialize weights

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

from torchvision.utils import make_grid

# ----------
#  Training
# ----------
print('')
niter = 0
epoch = 0
best_test = 0
while True:
    niter += 1
    x_s, y_s = next(src_train_iter)
    x_t, tgt_y = next(tgt_train_iter)
    x_s = x_s.cuda()
    y_s = y_s.cuda()
    x_t = x_t.cuda()

    optimizer.zero_grad()
    optimizer_ad.zero_grad()

    # Networks
    f_s, p_s = model(x_s)
    f_t, p_t = model(x_t)

    features = torch.cat((f_s, f_t), dim=0)
    outputs = torch.cat((p_s, p_t), dim=0)

    loss = nn.CrossEntropyLoss()(outputs.narrow(0, 0, x_s.size(0)), y_s)
    softmax_output = nn.Softmax(dim=1)(outputs)

    output1 = classifier1(features)
    softmax_output1 = nn.Softmax(dim=1)(output1)
    softmax_output = (1-opt.cla_plus_weight)*softmax_output+ opt.cla_plus_weight*softmax_output1

    
    loss += CDAN([features, softmax_output], ad_net, None, None, random_layer)
    

    # if epoch > start_epoch:
    #     if method == 'CDAN-E':
    #         entropy = loss_func.Entropy(softmax_output)
    #         loss += loss_func.CDAN([features, softmax_output], ad_net, entropy, network.calc_coeff(num_iter*(epoch-start_epoch)+batch_idx), random_layer)
    #     elif method == 'CDAN':
    #         loss += loss_func.CDAN([features, softmax_output], ad_net, None, None, random_layer)
    #     elif method == 'DANN':
    #         loss += loss_func.DANN(features, ad_net)
    #     else:
    #         raise ValueError('Method cannot be recognized.')

    num_feature = features.size(0)
    # =================train discriminator T
    real_label = Variable(torch.ones(num_feature)).cuda()
    fake_label = Variable(torch.zeros(num_feature)).cuda()

    optimizer_G.zero_grad()
    
    # Identity loss
    same_t = gen_st(f_t)
    loss_identity_t = criterion_identity(same_t, f_t)

    same_s = gen_ts(f_s)
    loss_identity_s = criterion_identity(same_s, f_s)

    # Gan loss
    f_st = gen_st(f_s)
    pred_f_st = D_t(f_st)
    loss_G_s2t = criterion_GAN(pred_f_st, y_s.float())

    f_ts = gen_ts(f_t)
    pred_f_ts = D_s(f_ts)
    loss_G_t2s = criterion_GAN(pred_f_ts, y_s.float())
    loss_G_t2s = 0

    ##### cycle loss
    f_sts = gen_ts(f_st)
    # loss_cycle_sts = criterion_cycle(f_sts, f_s)
    loss_cycle_sts = criterion_percep(f_sts, f_s)
    # pdb.set_trace()

    f_tst = gen_st(f_ts)
    # loss_cycle_tst = criterion_cycle(f_tst, f_t)
    loss_cycle_tst = criterion_percep(f_tst, f_t)
 
    # loss_cycle_tst = criterion_percep(vgg_model(f_t), vgg_model(f_tst))
    

    # sem loss
    pred_f_sts = model.classifier(f_sts)
    pred_f_st = model.classifier(f_st)
    loss_sem_t2s = criterion_Sem(pred_f_sts, pred_f_st)

    pred_f_tst = model.classifier(f_tst)
    pred_f_ts = model.classifier(f_ts)
    loss_sem_s2t = criterion_Sem(pred_f_tst, pred_f_ts)

    loss_cycle = loss_cycle_tst + loss_cycle_sts
    weight_in_loss_g = opt.weight_in_loss_g.split(',')
    loss_G = float(weight_in_loss_g[0]) * (loss_identity_s + loss_identity_t) + \
                float(weight_in_loss_g[1]) * (loss_G_s2t + loss_G_t2s) + \
                float(weight_in_loss_g[2])* loss_cycle + \
                float(weight_in_loss_g[3]) * (loss_sem_s2t + loss_sem_t2s)
    
    # softmax
    outputs_fake = classifier1(f_st.detach())

    # classifier
    classifier_loss1 = nn.CrossEntropyLoss()(outputs_fake, y_s)
    classifier1_optim.zero_grad()
    classifier_loss1.backward()
    classifier1_optim.step()


    total_loss = loss + opt.cyc_loss_weight * loss_G
    total_loss.backward()
    optimizer.step()
    optimizer_G.step()


    ###### Discriminator S ######
    optimizer_D_s.zero_grad()

    # Real loss
    pred_real = D_s(f_s.detach())
    loss_D_real = criterion_GAN(pred_real, real_label)

    # Fake loss
    f_ts = f_ts_buffer.push_and_pop(f_ts)
    pred_f_ts = D_s(f_ts.detach())
    loss_D_fake = criterion_GAN(pred_f_ts, fake_label)

    # Total loss
    loss_D_s = loss_D_real + loss_D_fake
    loss_D_s.backward()

    optimizer_D_s.step()
    ###################################

    ###### Discriminator t ######
    optimizer_D_t.zero_grad()

    # Real loss
    pred_real = D_t(f_t.detach())
    loss_D_real = criterion_GAN(pred_real, real_label)

    # Fake loss
    f_st = f_st_buffer.push_and_pop(f_st)
    pred_f_st = D_t(f_st.detach())
    loss_D_fake = criterion_GAN(pred_f_st, fake_label)

    # Total loss
    loss_D_t = loss_D_real + loss_D_fake
    loss_D_t.backward()
    optimizer_D_t.step()
    optimizer_ad.step()

    acc_src = 100*(np.mean(np.argmax((nn.Softmax(dim=1)(p_s)).data.cpu().numpy(), axis=1) == y_s.data.cpu().numpy()))
    
    print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tAccuracy: {6:.2f}\tLoss: {4:.6f}\tLoss+G: {5:.6f}'.format(
        epoch, niter%iter_per_epoch, iter_per_epoch,
        100. * niter / n_sample, loss.item(), total_loss.item(), acc_src.item()), end='\r')

    writer.add_scalar('{0}/Loss'.format(opt.prefix), loss.item(), niter)
    writer.add_scalar('{0}/Loss+G'.format(opt.prefix), total_loss.item(), niter)
    writer.add_scalar('{0}/src_accuracy'.format(opt.prefix), acc_src.item(), niter)

    if niter % iter_per_epoch == 0 and niter > 0:
        with torch.no_grad(): 
            epoch = niter // iter_per_epoch

            n = 0
            nagree = 0
            correct = 0
            test_loss = 0 
            for X, Y in test_loader: 
                n += X.size()[0]
                X_test = X.cuda() 
                Y_test = Y.cuda() 

                feature, output = model(X_test) #
                test_loss += nn.CrossEntropyLoss()(output, Y_test).item()
                pred = output.data.cpu().max(1, keepdim=True)[1]
                correct += pred.eq(Y_test.data.cpu().view_as(pred)).sum().item()
                
            test_loss /= len(test_loader.dataset)
            test_accuracy = 100. * correct / len(test_loader.dataset)

            if best_test < test_accuracy:
                best_test = test_accuracy

            writer.add_scalar('{0}/test_loss'.format(opt.prefix), test_loss, niter)
            writer.add_scalar('{0}/test_accuracy'.format(opt.prefix), test_accuracy, niter)
                


        if epoch >= opt.n_epochs:
            print('')
            print('train complete')
            run_dir_complete = '{0}_{1:0.2f}'.format(run_dir[:-8], best_test)
            os.rename(run_dir, run_dir_complete)
            break
