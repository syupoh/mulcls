# -*- coding: utf-8 -*-
import pdb
import os
import argparse
from tensorboardX import SummaryWriter

from Networks import *
from torchvision.utils import make_grid

from itertools import chain
from util.image_pool import ImagePool
from util.loss import GANLoss
from util.net import weights_init, Discriminator, Generator, LenetClassifier
from util.sampler import InfiniteSampler
import utils
from datetime import datetime

# from dropout import create_adversarial_dropout_mask, calculate_jacobians

_DIGIT_ROOT = '~/dataset/digits/'
_PREFIX = 'translation'
_MODEL = 'mnist_mnist'
_BETA = 10.0
_MU = 10.0
_GAMMA = 1.0
_ALPHA = 1.0
_NORM = True

def main(opt):
    opt.digitroot = _DIGIT_ROOT
    if opt.prefix=='':
        opt.prefix = _PREFIX
    if opt.model=='':
        opt.model = _MODEL
    if opt.beta=='':
        opt.beta = _BETA
    if opt.mu=='':
        opt.mu = _MU
    opt.gamma = _GAMMA
    opt.alpha = _ALPHA
    if opt.norm == None:
        opt.norm = _NORM

    modelname = '{0}_{1}_{2:0.1f}_{3:0.1f}'.format(opt.prefix, opt.model, opt.beta, opt.mu)
    modelpath = 'model/'+modelname+'.pth'

    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{0}'.format(opt.gpu))
    
    now = datetime.now()
    curtime = now.isoformat() 
    run_dir = "runs/{0}_{1}_ongoing".format(curtime[0:16], modelname)
    
    resultname = '{2}/result_{0}_{1}.txt'.format(modelname, opt.num_epochs, run_dir)
    n_ch = 64
    n_hidden = 5
    n_resblock = 4
    

    prompt = ''
    prompt += ('====================================\n')
    prompt += run_dir + '\n'
    for arg in vars(opt):
        prompt = '{0}{1} : {2}\n'.format(prompt, arg, getattr(opt, arg))
    prompt += ('====================================\n')
    print(prompt, end='')

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
    modelsplit = opt.model.split('_')
    if (modelsplit[0] == 'mnist' or modelsplit[0] == 'usps') and modelsplit[1] != 'svhn':
        n_c_in = 1 # number of color channels
    else:
        n_c_in = 3 # number of color channels
        
    if (modelsplit[1] == 'mnist' or modelsplit[1] == 'usps') and modelsplit[0] != 'svhn':
        n_c_out = 1 # number of color channels
    else:
        n_c_out = 3 # number of color channels


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
    
    # pdb.set_trace()

    #########################
    #### Model
    #########################
    if modelsplit[0] == 'svhn' or modelsplit[1] == 'svhn' or \
        modelsplit[0] == 'usps' or modelsplit[0] == 'cifar10' or \
            modelsplit[0] == 'stl10':
        model1 = conv9(p=opt.dropout_probability).cuda() # 3x32x32 -> 1x128x1x1 (before FC) 
        model2 = conv9(p=opt.dropout_probability).cuda() # 3x32x32 -> 1x128x1x1 (before FC) 
    else:
        model1 = conv3(p=opt.dropout_probability).cuda() # 1x28x28 -> 1x128x4x4 (before FC) 
        model2 = conv3(p=opt.dropout_probability).cuda() # 1x28x28 -> 1x128x4x4 (before FC) 


    dropout_mask1 = torch.randint(2, (1, 128, 1, 1), dtype=torch.float).cuda()
    # dropout_mask1 = torch.randint(2,(1,128,4,4), dtype=torch.float).cuda()
    
    weights_init_gaussian = weights_init('gaussian')

    for X, Y in train_loader: 
        res_x = X.shape[-1]
        break
    
    for X, Y in train_loader2: 
        res_y = X.shape[-1]
        break

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

    opt_gen = torch.optim.Adam(
        chain(gen_st.parameters(), gen_ts.parameters(), model1.parameters(), model2.parameters()), **config2)  
    opt_dis = torch.optim.Adam(
        chain(dis_s.parameters(), dis_t.parameters()), **config2)  

    loss_CE = torch.nn.CrossEntropyLoss().cuda()
    loss_KLD = torch.nn.KLDivLoss(reduction='batchmean').cuda()

    loss_LS = GANLoss(device, use_lsgan=True)
    
    #########################
    #### argument print
    #########################

    writer = SummaryWriter(run_dir)


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

        while True:
            model1.train()
            model2.train()

            niter += 1
            src_x, src_y = next(src_train_iter)
            tgt_x, tgt_y = next(tgt_train_iter)

            src_x = src_x.cuda()
            src_y = src_y.cuda()
            tgt_x = tgt_x.cuda()



            fake_tgt_x = gen_st(src_x)
            fake_src_x = gen_ts(tgt_x)
            fake_back_src_x = gen_ts(fake_tgt_x)

            if opt.prefix == 'tranlsation_noCE':
                loss_gen = opt.gamma * loss_LS(dis_s(fake_src_x), True)
                loss_gen += opt.alpha * loss_LS(dis_t(fake_tgt_x), True)
            else:
                loss_gen = opt.beta * loss_CE(model2(fake_tgt_x), src_y)
                loss_gen += opt.mu * loss_CE(model1(src_x), src_y)

                loss_gen += opt.gamma * loss_LS(dis_s(fake_src_x), True)
                loss_gen += opt.alpha * loss_LS(dis_t(fake_tgt_x), True)

            loss_dis_s = opt.gamma * loss_LS(
                dis_s(fake_src_x_pool.query(fake_src_x)), False)
            loss_dis_s += opt.gamma * loss_LS(dis_s(src_x), True)

            loss_dis_t = opt.alpha * loss_LS(
                dis_t(fake_tgt_x_pool.query(fake_tgt_x)), False)
            loss_dis_t += opt.alpha * loss_LS(dis_t(tgt_x), True)

            loss_dis = loss_dis_s + loss_dis_t


            for optim, loss in zip([opt_dis, opt_gen], [loss_dis, loss_gen]):
                optim.zero_grad()
                loss.backward(retain_graph=True)
                optim.step()

            if niter % opt.print_delay == 0 and niter > 0:
                with torch.no_grad(): 
        ##########################
                    loss_dis_s1 = opt.gamma * loss_LS(
                        dis_s(fake_src_x_pool.query(fake_src_x)), False)
                    loss_dis_s2 = opt.gamma * loss_LS(dis_s(src_x), True)
                    loss_dis_t1 = opt.alpha * loss_LS(
                        dis_t(fake_tgt_x_pool.query(fake_tgt_x)), False)
                    loss_dis_t2 = opt.alpha * loss_LS(dis_t(tgt_x), True)

                    loss_gen_s = opt.gamma * loss_LS(dis_s(fake_src_x), True)
                    loss_gen_t = opt.alpha * loss_LS(dis_t(fake_tgt_x), True)
                    loss_gen_CE_t = opt.beta * loss_CE(model2(fake_tgt_x), src_y)
                    loss_gen_CE_s = opt.mu * loss_CE(model1(src_x), src_y)
        ###########################
        
                    print('epoch {0} ({1}/{2}) '.format(epoch, (niter % iter_per_epoch), iter_per_epoch ) \
                    + 'dis_s1 {0:02.4f}, dis_s2 {1:02.4f}, '.format(loss_dis_s1.item(), loss_dis_s2.item()) \
                        + 'dis_t1 {0:02.4f}, dis_t2 {1:02.4f}, '.format(loss_dis_t1.item(), loss_dis_t2.item()) \
                            + 'loss_gen_s {0:02.4f}, loss_gen_t {1:02.4f} '.format(loss_gen_s.item(), loss_gen_t.item())
                                + 'loss_gen_CE_t {0:02.4f}, loss_gen_CE_s {1:02.4f}'.format(loss_gen_CE_t.item(), loss_gen_CE_s.item()), end='\r') 


                    writer.add_scalar('dis/src', loss_dis_s.item(), niter)
                    writer.add_scalar('dis/src1', loss_dis_s1.item(), niter)
                    writer.add_scalar('dis/src2', loss_dis_s2.item(), niter)
                    writer.add_scalar('dis/tgt', loss_dis_t.item(), niter)
                    writer.add_scalar('dis/tgt1', loss_dis_t1.item(), niter)
                    writer.add_scalar('dis/tgt2', loss_dis_t2.item(), niter)
                    writer.add_scalar('gen', loss_gen.item(), niter)
                    writer.add_scalar('gen/src', loss_gen_s.item(), niter)
                    writer.add_scalar('gen/tgt', loss_gen_t.item(), niter)
                    writer.add_scalar('CE/tgt', loss_CE(model2(fake_tgt_x), src_y).item(), niter)
                    writer.add_scalar('CE/src', loss_CE(model1(src_x), src_y).item(), niter)
                    
                    # pdb.set_trace()
                    if niter % (opt.print_delay*10) == 0 :
                        data_grid = []
                        for x in [src_x, fake_tgt_x, fake_back_src_x, tgt_x,
                                    fake_src_x]:
                            x = x.to(torch.device('cpu'))
                            if x.size(1) == 1:
                                x = x.repeat(1, 3, 1, 1)  # grayscale2rgb
                            data_grid.append(x)
                        grid = make_grid(torch.cat(tuple(data_grid), dim=0),
                                        normalize=True, range=(X_min, X_max), nrow=opt.batch_size) # for SVHN?
                        writer.add_image('generated_{0}'.format(opt.prefix), grid, niter)



            if niter % iter_per_epoch == 0 and niter > 0:
                with torch.no_grad(): 
                    epoch = niter // iter_per_epoch

                    model1.eval()
                    model2.eval()

                    avgaccuracy1 = 0
                    avgaccuracy2 = 0
                    n = 0
                    nagree = 0
                
                    for X, Y in test_loader: 
                        n += X.size()[0]
                        X_test = X.cuda() 
                        Y_test = Y.cuda() 

                        prediction1 = model1(X_test) #
                        predicted_classes1 = torch.argmax(prediction1, 1) 
                        correct_count1 = (predicted_classes1 == Y_test) 
                        testaccuracy1 = correct_count1.float().sum()
                        avgaccuracy1 += testaccuracy1
                        
                        prediction2 = model2(X_test) #
                        predicted_classes2 = torch.argmax(prediction2, 1) 
                        correct_count2 = (predicted_classes2 == Y_test) 
                        testaccuracy2 = correct_count2.float().sum()
                        avgaccuracy2 += testaccuracy2

                    avgaccuracy1 = (avgaccuracy1/n) *100
                    avgaccuracy2 = (avgaccuracy2/n) *100
                    agreement = (predicted_classes1 == predicted_classes2)
                    nagree = nagree + (agreement).int().sum()

                    writer.add_scalar('accuracy/tgt', avgaccuracy1, niter)
                    writer.add_scalar('accuracy/src', avgaccuracy2, niter)
                    writer.add_scalar('agreement', (nagree/n)*100, niter)
                    
                    f = open(resultname, 'a')
                    f.write('epoch : {0}\n'.format(epoch))
                    f.write('\tloss_gen_s : {0:0.4f}\n'.format(loss_gen_s.item()))
                    f.write('\tloss_gen_t : {0:0.4f}\n'.format(loss_gen_t.item()))
                    f.write('\tloss_gen_CE_t : {0:0.4f}\n'.format(loss_gen_CE_t.item()))
                    f.write('\tloss_gen_CE_s : {0:0.4f}\n'.format(loss_gen_CE_s.item())) 
                    f.write('\tloss_dis_s1 : {0:0.4f}\n'.format(loss_dis_s1.item()))
                    f.write('\tloss_dis_t1 : {0:0.4f}\n'.format(loss_dis_t1.item()))
                    f.write('\tloss_dis_s2 : {0:0.4f}\n'.format(loss_dis_s2.item()))
                    f.write('\tloss_dis_t2 : {0:0.4f}\n'.format(loss_dis_t2.item())) 
                    f.write('\tavgaccuracy_tgt : {0:0.2f}\n'.format(avgaccuracy1))
                    f.write('\tavgaccuracy_src : {0:0.2f}\n'.format(avgaccuracy2)) 
                    f.write('\tagreement : {0}\n'.format(nagree))  
                    f.close()
              
            if epoch >= opt.num_epochs:
                os.rename(run_dir, run_dir[:-8])
                break


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser = utils.add_args(arg_parser)
    opt_ = arg_parser.parse_args()
    
    main(opt_)
