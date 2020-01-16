# -*- coding: utf-8 -*-
import random 
import matplotlib as plt
import pdb
import os
import argparse
from tensorboardX import SummaryWriter
import time

from Networks import * 
import utils
# from dropout import create_adversarial_dropout_mask, calculate_jacobians

_DIGIT_ROOT = '~/dataset/digits/'
_PREFIX = ''

opt = argparse.ArgumentParser()
opt = utils.arg_parser(opt)
opt = opt.parse_args()

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
    # opt.model = 'svhn_mnist'
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
    resultname = './result/result_{0}_{1}_{2}.txt'.format(opt.prefix, opt.model, opt.num_epochs)
    #### DATASET 
    modelsplit = (opt.model).split('_')

    train_loader, test_loader = utils.load_data(prefix=opt.prefix, opt=opt)


    ###################### model
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{0}'.format(opt.gpu))

    if modelsplit[0]=='svhn' or modelsplit[0]=='svhn' :
        model = conv9(p=opt.dropout_probability).cuda() # 3x32x32 -> 1x128x1x1 (before FC) 
    else:
        model = conv3(p=opt.dropout_probability).cuda() # 1x28x28 -> 1x128x4x4 (before FC)


    dropout_mask1 = torch.randint(2,(1,128,1,1), dtype=torch.float).cuda()
    # dropout_mask1 = torch.randint(2,(1,128,4,4), dtype=torch.float).cuda()
    ###########################
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate) 
    loss_function = torch.nn.CrossEntropyLoss().cuda()

    f = open(resultname,'w')
    f.write('opt.model : {0}\n'.format(opt.model))
    f.write('opt.num_epochs : {0}\n'.format(opt.num_epochs))
    f.write('dropout_probability : {0}\n'.format(opt.dropout_probability))
    f.write('learning_rate : {0}\n'.format(opt.learning_rate))
    f.write('gpu : {0}\n'.format(opt.gpu))
    f.write('batch_size : {0}\n'.format(opt.batch_size))
    f.write('====================================\n')
    f.close()

    modelpath = 'model/'+opt.model+'.pth'
    if os.path.isfile(opt.pretrained):
        modelpath = opt.pretrained
        print("model load..", modelpath)
        checkpoint = torch.load(modelpath, map_location='cuda:{0}'.format(opt.gpu))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        dropout_mask1 = checkpoint['dropout_mask1']

    else:  
        modelpath='model/{0}'.format(opt.model)
        os.makedirs(modelpath, exist_ok=True)

        print("model train..")
        model.train() # dropout use
        for epoch in range(opt.num_epochs): 
            avg_loss = 0 
            batch_count = len(train_loader) 
            for X, Y in train_loader: 
                X = X.cuda() 
                Y = Y.cuda() 

                # prediction = model(X, dropout_mask1) 
                prediction = model(X) 
                loss = loss_function(prediction, Y) 
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step() 

                predicted_classes = torch.argmax(prediction, 1) 
                correct_count = (predicted_classes == Y) # average of correct count 
                trainaccuracy = correct_count.float().sum()
                trainaccuracy = trainaccuracy / len(Y.cpu()) *100
                # avg_loss += (loss / batch_count) 

                print('epoch : {0}, loss : {1:0.4f}, trainaccuracy : {2:0.2f}'.format(epoch,loss.item(), trainaccuracy.item()), end='\r') 
            #######################################################


            if epoch % opt.valid_epoch == 0:
                avgaccuracy = 0
                n=0
                for X, Y in test_loader: 
                    n += X.size()[0]
                    X_test = X.cuda() 
                    Y_test = Y.cuda() 

                    prediction = model(X_test) #

                    predicted_classes = torch.argmax(prediction, 1) 
                    correct_count = (predicted_classes == Y_test) 
                    testaccuracy = correct_count.float().sum()

                    avgaccuracy += testaccuracy

                avgaccuracy = (avgaccuracy/n) *100

                f = open(resultname,'a')
                f.write('epoch : {0}\n'.format(epoch))
                f.write('\tloss : {0:0.4f}\n'.format(loss.item()))
                f.write('\ttrainaccuracy : {0:0.2f}\n'.format(trainaccuracy.item()))
                f.write('\ttestaccuracy : {0:0.2f}\n'.format(avgaccuracy.item()))
                f.close()
                print('')
                
                modelsave = '{0}/{1}_{2}.pth'.format(modelpath,opt.model,epoch)
                print(' testaccuracy : {0:0.2f}'.format(avgaccuracy.item()))
                print(' -> model save : ', modelsave)

                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'trainaccuracy': trainaccuracy.item(),
                            'testaccuracy': avgaccuracy.item(),
                            'dropout_mask1': dropout_mask1,
                            'loss': loss,
                            }, modelsave)
            


    with torch.no_grad(): 
        model.eval() 
        n=0
        avgaccuracy = 0
        for X, Y in test_loader: 
            n += X.size()[0]
            X_test = X.cuda() 
            Y_test = Y.cuda() 

            prediction = model(X_test) 

            predicted_classes = torch.argmax(prediction, 1) 
            correct_count = (predicted_classes == Y_test) #
            accuracy = correct_count.float().sum()

            avgaccuracy += accuracy
        avgaccuracy = (avgaccuracy/n ) *100
        
        print('')
        print(' Final average : {0:0.2f}%'.format(avgaccuracy.item()))
            
        
        f = open(resultname,'a')
        f.write('Final average : {0:0.2f}'.format(avgaccuracy.item()))
        f.close()
        
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser = utils.add_args(arg_parser)
    opt = arg_parser.parse_args()
    
    opt.digitroot = _DIGIT_ROOT
    opt.prefix = _PREFIX
    main(opt)