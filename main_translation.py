# -*- coding: utf-8 -*-
import random 
import matplotlib as plt
import pdb
import os
import argparse
from tensorboardX import SummaryWriter
import time

from Networks import * 

from itertools import chain
from util.image_pool import ImagePool
from util.loss import GANLoss
from util.net import weights_init, Discriminator, Generator, LenetClassifier
from util.sampler import InfiniteSampler
import utils

from torchvision.utils import make_grid
# from dropout import create_adversarial_dropout_mask, calculate_jacobians

_DIGIT_ROOT = '~/dataset/digits/'
_PREFIX = 'translation'

    ### For Tensorboard
        #   cur = time.time()
        #   run_dir = "runs/{0}".format(curtime[0:19])
        
        #   writer = SummaryWriter(run_dir)

        # writer.add_image('generated', sum_img.view(3, 256, 512), epoch)
        # writer.add_image('generated', sum_img.view(3, 256, 512), epoch)
    #             self.writer.add_scalar('PA', PA, self.current_epoch)
    #             self.writer.add_scalar('MPA', MPA, self.current_epoch)
    #             self.writer.add_scalar('MIoU', MIoU, self.current_epoch)
    #             self.writer.add_scalar('FWIoU', FWIoU, self.current_epoch)
    # writer.close()
    
def main(opt):
    beta = 10.0
    mu = 10.0
    gamma = 1.0
    alpha = 1.0
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{0}'.format(opt.gpu))
    exp = 'mulcls'
    # opt.model = 'svhn_mnist'
    # opt.model = 'mnist_usps'
    # opt.model = 'usps_mnist'
    # opt.model = 'cifar10_stl10'
    # opt.model = 'stl10_cifar10'

    # opt.model = 'svhn_svhn'
    # opt.model = 'mnist_mnist'
    # opt.model = 'usps_usps'
    # opt.model = 'svhn_usps'
    #########################
    #### DATASET 
    #########################

    # if norm==True:
    #     X_min = -1
    #     X_max = 1
    # else:
    #     X_min = src.train_X.min()
    #     X_max = src.train_X.max()


    modelsplit = opt.model.split('_')
    trainset, trainset2, testset = utils.load_data(prefix=opt.prefix, opt=opt)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, drop_last=True, sampler=InfiniteSampler(len(trainset))) # model
    train_loader2 = torch.utils.data.DataLoader(trainset2, batch_size=opt.batch_size, drop_last=True, sampler=InfiniteSampler(len(trainset2))) # model
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, drop_last=True) # model

    n_sample = max(len(trainset), len(trainset2))
    iter_per_epoch = n_sample // opt.batch_size + 1

    src_train_iter = iter(train_loader)
    tgt_train_iter = iter(train_loader2)
    # tgt_test_iter = iter(test_loader)

    
    #########################
    #### Model 
    #########################
    modelname = '{0}_{1}_{2:0.1f}_{3:0.1f}'.format(opt.prefix, opt.model, opt.dropout_probability, opt.loss4_KLD_dis_rate)
    resultname = './result/result_{0}_{1}.txt'.format(modelname, opt.num_epochs)

    if modelsplit[0] == 'svhn' or modelsplit[1] == 'svhn' or \
        modelsplit[0] == 'usps' or modelsplit[0] == 'cifar10' or \
            modelsplit[0] == 'stl10':

        for i in range(1, 5):
            globals()['model{0}'.format(i)] = conv9(p=opt.dropout_probability).cuda() 
        # model1 = conv9(p=opt.dropout_probability).cuda() # 3x32x32 -> 1x128x1x1 (before FC) 
    else:
        for i in range(1, 5):
            globals()['model{0}'.format(i)] = conv3(p=opt.dropout_probability).cuda() 
        # model1 = conv3(p=opt.dropout_probability).cuda() # 1x28x28 -> 1x128x4x4 (before FC) 


    dropout_mask1 = torch.randint(2, (1, 128, 1, 1), dtype=torch.float).cuda()
    # dropout_mask1 = torch.randint(2,(1,128,4,4), dtype=torch.float).cuda()
    
    weights_init_gaussian = weights_init('gaussian')

    n_ch = 64
    n_hidden = 5
    n_resblock = 4

    for X, Y in train_loader: 
        # pdb.set_trace()
        res_x = X.shape[-1]
        break
    
    for X, Y in train_loader2: 
        # pdb.set_trace()
        res_y = X.shape[-1]
        break


    if modelsplit[0] == 'mnist' or modelsplit[0] == 'usps':
        n_c_in = 1 # number of color channels
    else:
        n_c_in = 3 # number of color channels
        
    if modelsplit[1] == 'mnist' or modelsplit[1] == 'usps':
        n_c_out = 1 # number of color channels
    else:
        n_c_out = 3 # number of color channels

    gen_st = Generator(n_hidden=n_hidden, n_resblock=n_resblock, \
        n_ch=n_ch, res=res_x, n_c_in=n_c_in, n_c_out=n_c_out).cuda()
        
    gen_ts = Generator(n_hidden=n_hidden, n_resblock=n_resblock, \
        n_ch=n_ch, res=res_y, n_c_in=n_c_out, n_c_out=n_c_in).cuda()

    dis_s = Discriminator(n_ch=n_ch, res=res_x, n_c_in=n_c_in).cuda()
    dis_t = Discriminator(n_ch=n_ch, res=res_y, n_c_in=n_c_out).cuda()

    gen_st.apply(weights_init_gaussian)
    gen_ts.apply(weights_init_gaussian)
    dis_s.apply(weights_init_gaussian)
    dis_t.apply(weights_init_gaussian)

    pool_size = 50
    fake_src_x_pool = ImagePool(pool_size * opt.batch_size)
    fake_tgt_x_pool = ImagePool(pool_size * opt.batch_size)

    #########################
    #### Loss 
    #########################

    config2 = {'lr': opt.learning_rate, 'weight_decay': opt.weight_decay, 'betas': (0.5, 0.999)}

    for i in range(1, 5):
        globals()['optimizer{0}'.format(i)] = torch.optim.Adam(globals()['model{0}'.format(i)].parameters(), lr=opt.learning_rate)  
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=opt.learning_rate)  # 1x28x28 -> 1x128x4x4 (before FC) 
    opt_gen = torch.optim.Adam(
        chain(gen_st.parameters(), gen_ts.parameters(), model1.parameters(), model2.parameters()), **config2)  
    opt_dis = torch.optim.Adam(
        chain(dis_s.parameters(),dis_t.parameters()), **config2)  

    loss_CE = torch.nn.CrossEntropyLoss().cuda()
    loss_KLD = torch.nn.KLDivLoss(reduction='batchmean').cuda()

    loss_LS = GANLoss(device, use_lsgan=True)
    
    #########################
    #### argument print
    #########################

    modelpath = 'model/'+modelname+'.pth'
    prompt=''
    
    opt = arg_parser.parse_args()
    prompt=prompt+('====================================\n')
    for arg in vars(opt):
        prompt='{0}{1} : {2}\n'.format(prompt, arg, getattr(opt, arg))
    prompt=prompt+('====================================\n')
    
    print(prompt, end='')

    f = open(resultname, 'w')
    f.write(prompt)
    f.close()

    #########################
    #### Run 
    #########################


    if os.path.isfile(opt.pretrained):
        modelpath = opt.pretrained
        print("model load..", modelpath)
        checkpoint = torch.load(modelpath, map_location='cuda:{0}'.format(opt.gpu))
        dropout_mask1 = checkpoint['dropout_mask1']

    else:  
        modelpath='model/{0}'.format(modelname)
        os.makedirs(modelpath, exist_ok=True)

        print("model train..")
        print(modelname)


        niter = 0
        epoch = 0
        # writer = SummaryWriter(logdir='runs/mulcls', comment='_mulcls')
        writer = SummaryWriter(comment='_mulcls')


        for i in range(1, 5):
            globals()['model{0}'.format(i)].train()
        #model1.train() # dropout use

        while True:
            niter += 1
            src_x, src_y = next(src_train_iter)
            tgt_x, tgt_y = next(tgt_train_iter)

            src_x = src_x.cuda()
            src_y = src_y.cuda()
            tgt_x = tgt_x.cuda()


            fake_tgt_x = gen_st(src_x)
            fake_src_x = gen_ts(tgt_x)
            fake_back_src_x = gen_ts(fake_tgt_x)

##########################
            loss_gen1 = gamma * loss_LS(dis_s(fake_src_x), True)
            loss_gen2 = alpha * loss_LS(dis_t(fake_tgt_x), True)
            loss_gen3 = beta * loss_CE(model2(fake_tgt_x), src_y)
            loss_gen4 = mu * loss_CE(model1(src_x), src_y)
#######################

            loss_gen = beta * loss_CE(model2(fake_tgt_x), src_y)
            loss_gen += mu * loss_CE(model1(src_x), src_y)

            loss_gen += gamma * loss_LS(dis_s(fake_src_x), True)
            loss_gen += alpha * loss_LS(dis_t(fake_tgt_x), True)

            loss_dis_s = gamma * loss_LS(
                dis_s(fake_src_x_pool.query(fake_src_x)), False)
            loss_dis_s += gamma * loss_LS(dis_s(src_x), True)

            loss_dis_t = alpha * loss_LS(
                dis_t(fake_tgt_x_pool.query(fake_tgt_x)), False)
            loss_dis_t += alpha * loss_LS(dis_t(tgt_x), True)

            loss_dis = loss_dis_s + loss_dis_t

            if niter == 1:
                data = []
                for x in [src_x, fake_tgt_x, fake_back_src_x, tgt_x,
                            fake_src_x]:
                    x = x.to(torch.device('cpu'))
                    if x.size(1) == 1:
                        x = x.repeat(1, 3, 1, 1)  # grayscale2rgb
                    data.append(x)
                grid = make_grid(torch.cat(tuple(data), dim=0),
                                    normalize=True, range=(-1, 1))
                writer.add_image('generated_{0}'.format(exp), grid, epoch)

            for optim, loss in zip([opt_dis, opt_gen], [loss_dis, loss_gen]):
                optim.zero_grad()
                loss.backward(retain_graph=True)
                optim.step()

            if niter % 100 == 0 and niter > 0:
                writer.add_scalar('dis/src', loss_dis_s.item(), niter)
                writer.add_scalar('dis/tgt', loss_dis_t.item(), niter)
                writer.add_scalar('gen', loss_gen.item(), niter)
                

            if niter % iter_per_epoch == 0:
                epoch = niter // iter_per_epoch

                f = open(resultname, 'a')
                f.write('epoch : {0}\n'.format(epoch))
                f.write('\tloss_gen1 : {0:0.4f}\n'.format(loss_gen1.item()))
                f.write('\tloss_gen2 : {0:0.4f}\n'.format(loss_gen2.item()))
                f.write('\tloss_gen_CE_t : {0:0.4f}\n'.format(loss_gen3.item()))
                f.write('\tloss_gen_CE_s : {0:0.4f}\n'.format(loss_gen4.item()))                
                f.close()

                if epoch % 1 == 0:
                    data = []
                    for x in [src_x, fake_tgt_x, fake_back_src_x, tgt_x,
                            fake_src_x]:
                        x = x.to(torch.device('cpu'))
                        if x.size(1) == 1:
                            x = x.repeat(1, 3, 1, 1)  # grayscale2rgb
                        data.append(x)
                    grid = make_grid(torch.cat(tuple(data), dim=0))
                    writer.add_image('generated_{0}'.format(exp), grid, epoch)
            

            print('epoch {0} ({1}/{2}) '.format(epoch, niter % iter_per_epoch, iter_per_epoch) \
                + 'dis_s {0:02.4f}, dis_t {1:02.4f}, '.format(loss_dis_s.item(), loss_dis_t.item()) \
                    + 'loss_gen1 {0:02.4f}, loss_gen2 {1:02.4f} '.format(loss_gen1.item(), loss_gen2.item())
                    + 'loss_gen_CE_t {0:02.4f}, loss_gen_CE_s {1:02.4f}'.format(loss_gen3.item(), loss_gen4.item()), end='\r') 

            if epoch >= opt.num_epochs:
                break


            

    
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser = utils.add_args(arg_parser)
    opt = arg_parser.parse_args()
    
    
    opt.digitroot = _DIGIT_ROOT
    opt.prefix = _PREFIX
    main(opt)
