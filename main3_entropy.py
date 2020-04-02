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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import argparse
import os
import numpy as np
import math
import pdb 

from datetime import datetime
from util.sampler import InfiniteSampler
from utils import ReplayBuffer
import Networks2 as net


parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', type=int, default='0')
parser.add_argument('--start_acc', type=int, default='50')
parser.add_argument('--start_acc2', type=int, default='0')
parser.add_argument('--digitroot', type=str, default='~/dataset/digits/')
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4096, help="size of the batches")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--gpu", type=int, default=3)
parser.add_argument('--model', type=str, default='svhn_mnist')
parser.add_argument('--prefix', type=str, default='adEntPlus_2cls')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument("--lr", type=float, default=1e-2, help="adam: learning rate")
parser.add_argument("--lr2", type=float, default=1e-2, help="adam: learning rate2")
# parser.add_argument('--lr_decay', type=int, default='100')
# parser.add_argument('--cla_plus_weight', type=float, default=3e-1)
# parser.add_argument('--cyc_loss_weight',type=float,default=0.01)
# parser.add_argument('--weight_in_loss_g',type=str,default='1,0.01,0.1,0.1')
# parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')
parser.add_argument('--random', type=bool, default=False, help='whether to use random')
parser.add_argument("--seedfix", type=bool, default=False, help="seedfix")
parser.add_argument('--norm', type=bool, default=True)
parser.add_argument('--modelload', type=str, default=None)
opt = parser.parse_args()

now = datetime.now()
curtime = now.isoformat() 
modelname = '{prefix}_{model}_{lr}_{lr2}_{start_acc}_{start_acc2}_{weight_decay:0.4f}'.format(
    prefix=opt.prefix, model=opt.model, lr=opt.lr, lr2=opt.lr2, \
        start_acc=opt.start_acc, start_acc2=opt.start_acc2, weight_decay=opt.weight_decay)
run_dir = "runs/{0}_{1}_ongoing".format(curtime[0:16], modelname)
writer = SummaryWriter(run_dir)

seedCPU = torch.initial_seed()
seedGPU = torch.cuda.initial_seed()

if opt.seedfix:
    seedCPU = 1341862488249197428
    seedGPU = 1976746675577478

    torch.manual_seed(seedCPU)
    torch.cuda.manual_seed_all(seedGPU)

prompt = ''
prompt += ('====================================\n')
prompt += run_dir + '\n'
for arg in vars(opt):
    prompt = '{0}{1} : {2}\n'.format(prompt, arg, getattr(opt, arg))
prompt += ("Current CPU seed : {0}\n".format(seedCPU))
prompt += ("Current GPU seed : {0}\n".format(seedGPU))
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
test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, drop_last=True) # model

n_sample = max(len(trainset), len(trainset2))
iter_per_epoch = n_sample // opt.batch_size + 1

src_train_iter = iter(train_loader)
tgt_train_iter = iter(train_loader2)

if len(opt.prefix.split('_')) > 1:
    modelprefix = opt.prefix.split('_')
    MODELTYPE = modelprefix[1]
    MODELNAME = modelprefix[0]
else:
    MODELTYPE = None

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

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

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

# coeff = calc_coeff(num_iter*(epoch-start_epoch)+batch_idx)
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

def writer_tsne(writer, feature, ylabel, epoch, nametag):
    transformed = tsne_model.fit_transform(feature.detach().cpu())
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    fig = plt.figure()
    plt.scatter(xs, ys, c=ylabel.cpu())
    plt.xlim((-30, 30))
    plt.ylim((-30, 30))
    writer.add_figure(nametag, fig, epoch)

### Model
input_size = 512
num_feature = 1024
class_num = 10

model = net.DTN().cuda()
classifier1 = net.Classifier(input_size, class_num).cuda()
classifier2 = net.Classifier(input_size, class_num).cuda()

if opt.random:
    random_layer = net.RandomLayer([model.output_num(), class_num], 500)
    ad_net = net.AdversarialNetwork(500, 500).cuda()
    random_layer.cuda()
else:
    random_layer = None
    ad_net = net.AdversarialNetwork(model.output_num() * class_num, 500).cuda()


### Loss & Optimizers
optimizer_model = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
optimizer_classifier1 = torch.optim.Adam(classifier1.parameters(), lr=opt.lr2)
optimizer_classifier2 = torch.optim.Adam(classifier2.parameters(), lr=opt.lr2)
criterion_GAN = torch.nn.MSELoss()

### Initialize weights

real_label = Variable(torch.ones(num_feature)).cuda()
fake_label = Variable(torch.zeros(num_feature)).cuda()
if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

from torchvision.utils import make_grid

if opt.modelload is not None:
    if not opt.modelload.endswith(".pth"):
        file_list = os.listdir(opt.modelload)
        file_list_pt = [file for file in file_list if file.endswith(".pth")]

        best = 0
        for file in file_list_pt:
            besttemp = file.split('_')
            besttemp = besttemp[-1].split('.')[0] + '.' + besttemp[-1].split('.')[1]
            if float(best) < float(besttemp):
                best = besttemp
                bestpt = file
        

        opt.modelload = '{0}/{1}'.format(opt.modelload, bestpt)

    print('model load')
    print(' -> ', opt.modelload)   
    checkpoint = torch.load(opt.modelload, map_location='cuda:{0}'.format(opt.gpu))
    epoch = checkpoint['epoch']
    best_test = checkpoint['best_test']
    niter = checkpoint['niter']
    model.load_state_dict(checkpoint['model'])
    classifier1.load_state_dict(checkpoint['classifier1'])
    classifier2.load_state_dict(checkpoint['classifier2'])
    optimizer_model.load_state_dict(checkpoint['optimizer_model'])
    optimizer_classifier1.load_state_dict(checkpoint['optimizer_classifier1'])
    optimizer_classifier2.load_state_dict(checkpoint['optimizer_classifier2'])

# ----------
#  Training
# ----------
print('')
niter = 0
epoch = 0
best_test = 0

acc_src = 0
acc_src2 = 0
tsne_model = TSNE(learning_rate=100)
loss_adent = 0
loss_cos = 0

while True:
    niter += 1
    x_s, y_s = next(src_train_iter)
    x_t, tgt_y = next(tgt_train_iter)
    x_s = x_s.cuda()
    y_s = y_s.cuda()
    x_t = x_t.cuda()

    model.train()
    classifier1.train()
    classifier2.train()
    optimizer_model.zero_grad()
    optimizer_classifier1.zero_grad()
    optimizer_classifier2.zero_grad()


    # Networks Forward Propagation
    f_s, p_s = model(x_s)
    f_t, p_t = model(x_t)

    output_s = classifier1(f_s)
    output_s2 = classifier2(f_s)
    softmax_output_s = nn.Softmax(dim=1)(output_s)
    softmax_output_s2 = nn.Softmax(dim=1)(output_s2)
    output_t = classifier1(f_t, reverse=True)
    output_t2 = classifier2(f_t, reverse=True)
    softmax_output_t = nn.Softmax(dim=1)(output_t)   
    softmax_output_t2 = nn.Softmax(dim=1)(output_t2)   

    # ----------
    #  CE Loss
    # ----------    
    loss_ce1 = nn.CrossEntropyLoss()(output_s.narrow(0, 0, x_s.size(0)), y_s)    
    loss_ce2 = nn.CrossEntropyLoss()(output_s2.narrow(0, 0, x_s.size(0)), y_s)    
    loss_ce = loss_ce1 + loss_ce2

    loss_ce.backward(retain_graph=True)

    optimizer_model.step()
    optimizer_classifier1.step()
    optimizer_classifier2.step()

    optimizer_model.zero_grad()
    optimizer_classifier1.zero_grad()
    optimizer_classifier2.zero_grad()
    model.zero_grad()
    classifier1.zero_grad()
    classifier2.zero_grad()

    # ----------
    #  2cls Loss
    # ----------
    
    if min(acc_src, acc_src2) > opt.start_acc2 and (MODELTYPE.find('2cls') > -1):
        if MODELTYPE == '2clsA':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc[0].weight, classifier2.fc[0].weight)))
        elif MODELTYPE == '2clsB':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc[2].weight, classifier2.fc[2].weight)))
        elif MODELTYPE == '2clsC':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc[4].weight, classifier2.fc[4].weight)))
        elif MODELTYPE == '2clsD':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc1.weight, classifier2.fc1.weight)))
        elif MODELTYPE == '2clsE':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc1.weight, classifier2.fc1.weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[4].weight, classifier2.fc[4].weight)))/2
        elif MODELTYPE == '2clsF':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc1.weight, classifier2.fc1.weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[2].weight, classifier2.fc[2].weight)))/2
        elif MODELTYPE == '2clsG':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc1.weight, classifier2.fc1.weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[0].weight, classifier2.fc[0].weight)))/2
        elif MODELTYPE == '2clsH':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc[4].weight, classifier2.fc[4].weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[2].weight, classifier2.fc[2].weight)))/2
        elif MODELTYPE == '2clsI':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc[4].weight, classifier2.fc[4].weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[0].weight, classifier2.fc[0].weight)))/2
        elif MODELTYPE == '2clsJ':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc[2].weight, classifier2.fc[2].weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[0].weight, classifier2.fc[0].weight)))/2
        elif MODELTYPE == '2clsK':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc1.weight, classifier2.fc1.weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[2].weight, classifier2.fc[2].weight)) + \
                        torch.mean(nn.CosineSimilarity()(classifier1.fc[4].weight, classifier2.fc[4].weight)))/3
        elif MODELTYPE == '2clsL':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc1.weight, classifier2.fc1.weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[0].weight, classifier2.fc[0].weight)) + \
                        torch.mean(nn.CosineSimilarity()(classifier1.fc[4].weight, classifier2.fc[4].weight)))/3
        elif MODELTYPE == '2clsM':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc1.weight, classifier2.fc1.weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[0].weight, classifier2.fc[0].weight)) + \
                        torch.mean(nn.CosineSimilarity()(classifier1.fc[2].weight, classifier2.fc[2].weight)))/3
        elif MODELTYPE == '2clsN':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc[4].weight, classifier2.fc[4].weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[0].weight, classifier2.fc[0].weight)) + \
                        torch.mean(nn.CosineSimilarity()(classifier1.fc[2].weight, classifier2.fc[2].weight)))/3
        elif MODELTYPE == '2cls':
            loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc[0].weight, classifier2.fc[0].weight)) + \
                torch.mean(nn.CosineSimilarity()(classifier1.fc[2].weight, classifier2.fc[2].weight)) + \
                    torch.mean(nn.CosineSimilarity()(classifier1.fc[4].weight, classifier2.fc[4].weight)) + \
                        torch.mean(nn.CosineSimilarity()(classifier1.fc1.weight, classifier2.fc1.weight)))/4

        # A = classifier1.fc1.weight.data.clone()
        # AA = classifier1.fc1.weight.grad.data.clone()
        if (opt.gpu == 0 or opt.gpu == "0") and loss_cos < 0:
            loss_cos = -1 * loss_cos
        loss_cos.backward()
        
        optimizer_classifier1.step()
        optimizer_classifier2.step()

        # sum(sum(classifier1.fc1.weight.data-A))
        # sum(sum(classifier1.fc1.weight.grad.data-AA))

        optimizer_classifier1.zero_grad()
        optimizer_classifier2.zero_grad()

        classifier1.zero_grad()
        classifier2.zero_grad()

    # ----------
    #  adcls Loss
    # ----------
    if min(acc_src, acc_src2) > opt.start_acc and (MODELTYPE.find('adcls') > -1):
        # loss_adcos 
        # loss_cos = -1 * (torch.mean(nn.CosineSimilarity()(classifier1.fc[0].weight, classifier2.fc[0].weight)) + \
        #     torch.mean(nn.CosineSimilarity()(classifier1.fc[2].weight, classifier2.fc[2].weight)) + \
        #         torch.mean(nn.CosineSimilarity()(classifier1.fc[4].weight, classifier2.fc[4].weight)) + \
        #             torch.mean(nn.CosineSimilarity()(classifier1.fc1.weight, classifier2.fc1.weight)))/4
        # loss_adcls_D = criterion_GAN(pred_f_st, D)

        optimizer_classifier1.step()
        optimizer_classifier2.step()

        optimizer_classifier1.zero_grad()
        optimizer_classifier2.zero_grad()
        
        classifier1.zero_grad()
        classifier2.zero_grad()

    # ----------
    #  adEnt Loss
    # ----------
    if min(acc_src, acc_src2) > opt.start_acc and (MODELNAME.find('adEnt')) > -1:
    # if epoch > opt.start_epoch:
        if MODELNAME == 'adEntPlus':
            loss_adent = torch.mean(Entropy(softmax_output_t)) 
            if (MODELTYPE.find('2cls')) > -1:
                loss_adent = (loss_adent / 2) + torch.mean(Entropy(softmax_output_t2)) /2
        elif MODELNAME == 'adEntMinus':
            loss_adent = 0.1 * torch.mean(torch.sum(softmax_output_t * \
                (torch.log(softmax_output_t + 1e-5)), 1))
            if (MODELTYPE.find('2cls')) > -1:
                loss_adent = (loss_adent / 2) + torch.mean(torch.sum(softmax_output_t * \
                    (torch.log(softmax_output_t + 1e-5)), 1)) / 2
        
        loss_adent.backward()
        optimizer_model.step()
        optimizer_classifier1.step()
        optimizer_classifier2.step()

        model.zero_grad()
        classifier1.zero_grad()
        classifier2.zero_grad()
        optimizer_model.zero_grad()
        optimizer_classifier1.zero_grad()
        optimizer_classifier2.zero_grad()

    # ----------
    #  Save history
    # ----------
    acc_src = 100*(np.mean(np.argmax((nn.Softmax(dim=1)(output_s.detach())).data.cpu().numpy(), axis=1) == y_s.data.cpu().numpy()))        
    acc_src2 = 100*(np.mean(np.argmax((nn.Softmax(dim=1)(output_s2.detach())).data.cpu().numpy(), axis=1) == y_s.data.cpu().numpy()))        
    
    prompt = 'Train Epoch: {epoch} [{progress}/{iter_per_epoch}] {total_progress:.01f}% ' \
     'Src_acc: {Accuracy:.2f}, Src_acc2: {Accuracy2:.2f}, Best_Test {Best_Test:.2f} ' \
     'Loss_ce: {Loss_ce:.6f} Loss_adent: {Loss_adent:.6f} Loss_cos: {Loss_cos:.6f}'.format(
        epoch=epoch, progress=niter%iter_per_epoch, iter_per_epoch=iter_per_epoch, \
            total_progress=100. * niter / (iter_per_epoch*opt.n_epochs), \
            Loss_ce=loss_ce.item(), Loss_adent=loss_adent, Loss_cos=loss_cos, \
                Accuracy=acc_src.item(), Accuracy2=acc_src2.item(),\
                Best_Test=best_test)
    print(prompt, end='\r')

    f = open('{0}/opt.txt'.format(run_dir), 'a')
    f.write(prompt)
    f.write('\n')
    f.close()
    writer.add_scalar('CE_adEnt/loss_ce', loss_ce.item(), niter)
    writer.add_scalar('CE_adEnt/loss_adent', loss_adent, niter)
    writer.add_scalar('CE_adEnt/loss_cos', loss_cos, niter)
    writer.add_scalar('CE_adEnt/src_accuracy', acc_src.item(), niter)
    writer.add_scalar('CE_adEnt/src_accuracy2', acc_src2.item(), niter)

    # ----------
    #  Each epoch
    # ----------
    if niter % iter_per_epoch == 0 and niter > 0:
        with torch.no_grad(): 
            # writer_tsne(writer, f_s, y_s, epoch, 'f_s_tsne')
            # writer_tsne(writer, f_t, tgt_y, epoch, 'f_t_tsne')
            # writer_tsne(writer, torch.cat((f_s, f_t), dim=0), \
            #     torch.cat((fake_label[0:int(num_feature/2)], real_label[0:int(num_feature/2)]), dim=0),\
            #         epoch, 'f_s_f_t_tsne')

            epoch = niter // iter_per_epoch
            
            # if epoch % opt.lr_decay == 0:
            #     for param_group in optimizer_model.param_groups:
            #         param_group["lr"] = param_group["lr"] * 0.3

            n = 0
            nagree = 0
            correct = 0
            correct2 = 0
            test_loss = 0 
            test_loss2 = 0 
            for X, Y in test_loader: 
                model.eval()
                classifier1.eval()
                classifier2.eval()
                n += X.size()[0]
                X_test = X.cuda() 
                Y_test = Y.cuda() 

                feature, _ = model(X_test) #
                output = classifier1(feature)
                output2 = classifier2(feature)
                test_loss += nn.CrossEntropyLoss()(output, Y_test).item()
                pred = output.data.cpu().max(1, keepdim=True)[1]
                pred2 = output2.data.cpu().max(1, keepdim=True)[1]

                correct += (Y.view_as(pred) == pred).sum()
                correct2 += (Y.view_as(pred2) == pred2).sum()

            test_loss /= len(test_loader.dataset)
            test_accuracy = 100. * correct / len(test_loader.dataset)
            test_accuracy2 = 100. * correct2 / len(test_loader.dataset)


            prompt = ' Test_acc: {test_accuracy:.1f} Test_acc2: {test_accuracy2:.1f} ' \
            'Test loss: {test_loss:.6f}\t{run_dir}'.format(
                test_loss=test_loss, test_accuracy=test_accuracy, \
                    test_accuracy2=test_accuracy2, run_dir=run_dir)

            print('')
            print(prompt)

            f = open('{0}/opt.txt'.format(run_dir), 'a')
            f.write(prompt)
            f.write('\n')
            f.close()

            writer.add_scalar('CE_adEnt/test_loss', test_loss, epoch)
            writer.add_scalar('CE_adEnt/test_accuracy', test_accuracy, epoch)
            
            if best_test < max(test_accuracy, test_accuracy2):
                best_test = max(test_accuracy, test_accuracy2)

                modelsave = '{run_dir}/{prefix}_{epoch}_{best_test:.1f}_.pth'.format(
                    run_dir=run_dir, prefix=opt.prefix, epoch=epoch, \
                        best_test=best_test)

                if best_test > 50:
                    torch.save({
                        'epoch': epoch,
                        'best_test': best_test,
                        'niter': niter,
                        'model': model.state_dict(),
                        'classifier1': classifier1.state_dict(),
                        'classifier2': classifier2.state_dict(),
                        'optimizer_model': optimizer_model.state_dict(),
                        'optimizer_classifier1': optimizer_classifier1.state_dict(),
                        'optimizer_classifier2': optimizer_classifier2.state_dict(),
                        }, modelsave)

            
    # ----------
    #  End of train
    # ----------
    if epoch >= opt.n_epochs:
        print('')
        print('train complete')
        
        modelsave = '{run_dir}/{epoch}_{best_test:.1f}.pth'.format(
            run_dir=run_dir, epoch=epoch, best_test=best_test)

        torch.save({
            'epoch': epoch,
            'best_test': best_test,
            'niter': niter,
            'model': model.state_dict(),
            'classifier1': classifier1.state_dict(),
            'classifier2': classifier2.state_dict(),
            'optimizer_model': optimizer_model.state_dict(),
            'optimizer_classifier1': optimizer_classifier1.state_dict(),
            'optimizer_classifier2': optimizer_classifier2.state_dict(),
            }, modelsave)

        run_dir_complete = '{run_dir}_{best_test:0.2f}'.format(
            run_dir=run_dir[:-8], best_test=best_test)
        os.rename(run_dir, run_dir_complete)

        break

