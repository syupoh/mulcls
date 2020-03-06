import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from itertools import chain
from vat import VATLoss

import argparse
import os
import numpy as np
import math
import pdb 

from datetime import datetime
from util.sampler import InfiniteSampler
import Networks2 as net


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', type=int, default='3')
parser.add_argument('--lr_decay', type=int, default='100')
parser.add_argument('--digitroot', type=str, default='~/dataset/digits/')
parser.add_argument('--prefix', type=str, default='vat_entropy')
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
parser.add_argument('--select', type=str, default='translation')
opt = parser.parse_args()

now = datetime.now()
curtime = now.isoformat() 
modelname = '{0}_{1}_{2}_{3}_{4:0.3f}_{5:0.1f}_{6}_{7}'.format(
    opt.prefix, opt.model, opt.lr, opt.weight_in_loss_g, opt.cyc_loss_weight, opt.cla_plus_weight, opt.start_epoch, opt.lr_decay)
run_dir = "runs/{0}_{1}_ongoing".format(curtime[0:16], modelname)
writer = SummaryWriter(run_dir)


prompt = ''
prompt += ('====================================\n')
prompt += run_dir + '\n'
for arg in vars(opt):
    prompt = '{0}{1} : {2}\n'.format(prompt, arg, getattr(opt, arg))
prompt += ('====================================\n')
print(prompt, end='')

f = open('{0}/opt.txt'.format(run_dir), 'w')
f.write(prompt)
f.close()

if torch.cuda.is_available():
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(opt.gpu)
    torch.cuda.set_device(opt.gpu)
    # device = torch.device('cuda:{0}'.format(opt.gpu))
# Configure data loader

import utils
trainset, trainset2, testset = utils.load_data(opt=opt)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, drop_last=True, sampler=InfiniteSampler(len(trainset))) # model
train_loader2 = torch.utils.data.DataLoader(trainset2, batch_size=opt.batch_size, drop_last=True, sampler=InfiniteSampler(len(trainset2))) # model
test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, drop_last=True) # model

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


### Model
input_size = 512
num_feature = 1024
class_num = 10


model = net.DTN().cuda()
classifier_j = net.Classifier(512, class_num*2).cuda()

### Loss & Optimizers
# Loss functions
criterion_CE = torch.nn.CrossEntropyLoss().cuda()
criterion_VAT = VATLoss().cuda()

# Optimizers

model_optim = torch.optim.Adam(model.parameters(), lr=0.0003)
classifier_j_optim = torch.optim.Adam(classifier_j.parameters(), lr=0.0003)

# ----------
#  Training
# ----------
print('')
niter = 0
epoch = 0
best_test = 0


# zerotensor = torch.zeros(opt.batch_size, dtype=torch.long).cuda()
# torch.cat( (p_c_t, zerotensor))

acc_src = 0
while True:
    niter += 1
    x_s, y_s = next(src_train_iter)
    x_t, y_t = next(tgt_train_iter)
    x_s = x_s.cuda()
    y_s = y_s.cuda()
    x_t = x_t.cuda()

    
    model_optim.zero_grad()
    classifier_j_optim.zero_grad()

    for child in classifier_j.children():
        for param in child.parameters():
            param.requires_grad = True
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True

    # Networks
    f_s, p_c_s = model(x_s)
    f_t, p_c_t = model(x_t)
    p_j_s = classifier_j(f_s)
    p_j_t = classifier_j(f_t)
    y_t_hat = torch.Tensor.argmax(F.softmax(p_c_t, dim=0), dim=1)

    # pdb.set_trace()
    ###########

    # classifier_c update
    cls_loss = criterion_CE(p_c_s, y_s)
    vat_loss = criterion_VAT(model, x_s)
    vat_loss += criterion_VAT(model, x_t)
    
    # entropy_loss

    print(model.fc_params[0].weight)
    print(model.classifier.weight)
    print(classifier_j.fc[0].weight)
    pdb.set_trace()

    losses = cls_loss + vat_loss
    losses.backward(retain_graph=True)

    model_optim.step()

    print(model.fc_params[0].weight)
    print(model.classifier.weight)
    print(classifier_j.fc[0].weight)
    pdb.set_trace()


    # only classifier_j update
    ## freeze encoder, classifier_c

    classifier_j_optim.zero_grad()
    
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
            
    joint_loss = criterion_CE(p_j_s, y_s)
    if epoch > opt.start_epoch:
        joint_loss += criterion_CE(p_j_t, y_t_hat+10 ) 

    joint_loss.backward(retain_graph=True)

    classifier_j_optim.step()


    print(model.fc_params[0].weight)
    print(model.classifier.weight)
    print(classifier_j.fc[0].weight)
    pdb.set_trace()

    # only encoder update
    ## freeze classifier_j, classifier_c
    
    model_optim.zero_grad()

    for child in classifier_j.children():
        for param in child.parameters():
            param.requires_grad = False
    ct = 0
    for child in model.children():
        ct += 1
        if ct == 3:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True

    encoder_loss = criterion_CE(p_j_s, y_s+10 )
    if epoch > opt.start_epoch:
        encoder_loss += criterion_CE(p_j_t, y_t_hat)
    encoder_loss.backward()
    model_optim.step()    

    print(model.conv_params[0].weight)
    print(model.fc_params[0].weight)
    print(classifier_j.fc[0].weight)
    pdb.set_trace()

    acc_tgt = 100*(np.mean(np.argmax((nn.Softmax(dim=1)(p_j_t.detach())).data.cpu().numpy(), axis=1) == (y_t.data.cpu().numpy()+10)))
    acc_src = 100*(np.mean(np.argmax((nn.Softmax(dim=1)(p_c_s.detach())).data.cpu().numpy(), axis=1) == y_s.data.cpu().numpy()))
    
    print('Train Epoch: {0} [{1}/{2} ({3:.01f}%)] cls_loss: {4:.6f} joint_loss: {5:.6f} encoder_loss: {6:.6f} acc_src: {7:.2f} Best_Test {8:.2f} acc_tgt {9:.2f}'.format(
        epoch, niter%iter_per_epoch, iter_per_epoch, \
            100. * niter / (iter_per_epoch*opt.n_epochs), cls_loss.item(), joint_loss.item(), encoder_loss.item(), acc_src.item(), best_test, acc_tgt), end='\r')

    writer.add_scalar('vat_entropy/cls_loss', cls_loss.item(), niter)
    writer.add_scalar('vat_entropy/joint_loss', joint_loss.item(), niter)
    writer.add_scalar('vat_entropy/encoder_loss', encoder_loss.item(), niter)
    writer.add_scalar('vat_entropy/acc_src', acc_src.item(), niter)

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
                Y_test = Y.cuda() + 10

                feature, output2 = model(X_test) #
                output = classifier_j(f_t)

                test_loss += nn.CrossEntropyLoss()(output, Y_test).item()
                pred = output.data.cpu().max(1, keepdim=True)[1]
                correct += pred.eq(Y_test.data.cpu().view_as(pred)).sum().item()
                
            test_loss /= len(test_loader.dataset)
            test_accuracy = 100. * correct / len(test_loader.dataset)

            if best_test < test_accuracy:
                best_test = test_accuracy

                modelsave = '{0}/{1}_{2}_{3:.1f}.pth'.format(run_dir, opt.prefix, epoch, best_test)

                # if best_test > 50:
                #     torch.save({
                #         'epoch': epoch,
                #         'best_test': best_test,
                #         'niter': niter,
                #         'gen_st': gen_st.state_dict(),
                #         'gen_ts': gen_ts.state_dict(),
                #         'D_s': D_s.state_dict(),
                #         'D_t': D_s.state_dict(),
                #         'model': model.state_dict(),
                #         'ad_net': ad_net.state_dict(),
                #         'optimizer_G': optimizer_G.state_dict(),
                #         'optimizer_D_s': gen_ts.state_dict(),
                #         'optimizer_D_t': optimizer_D_t.state_dict(),
                #         'optimizer': optimizer.state_dict(),
                #         'optimizer_ad': optimizer_ad.state_dict(),
                #         }, modelsave)

            print('')
            print('Test loss: {0:.6f}\t accuracy : {1:.1f} {2}'.format(
                test_loss, test_accuracy, run_dir))

            writer.add_scalar('vat_entropy/test_loss', test_loss, epoch)
            writer.add_scalar('vat_entropy/test_accuracy', test_accuracy, epoch)
                


        # if epoch >= opt.n_epochs:
        #     print('')
        #     print('train complete')
            
        #     modelsave = '{0}/{1}_{2}_{3:.1f}.pth'.format(run_dir, opt.prefix, epoch, best_test)

        #     torch.save({
        #         'epoch': epoch,
        #         'best_test': best_test,
        #         'niter': niter,
        #         'gen_st': gen_st.state_dict(),
        #         'gen_ts': gen_ts.state_dict(),
        #         'D_s': D_s.state_dict(),
        #         'D_t': D_s.state_dict(),
        #         'model': model.state_dict(),
        #         'ad_net': ad_net.state_dict(),
        #         'optimizer_G': optimizer_G.state_dict(),
        #         'optimizer_D_s': gen_ts.state_dict(),
        #         'optimizer_D_t': optimizer_D_t.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'optimizer_ad': optimizer_ad.state_dict(),
        #         }, modelsave)

        #     run_dir_complete = '{0}_{1:0.2f}'.format(run_dir[:-8], best_test)
        #     os.rename(run_dir, run_dir_complete)

        #     break

