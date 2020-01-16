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
_PREFIX = 'div'

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
    modelsplit = opt.model.split('_')
    train_loader, test_loader = utils.load_data(prefix=opt.prefix, opt=opt)

    modelname = '{0}_{1}'.format(opt.prefix, opt.model)
    
    resultname = './result/result_{0}_{1}.txt'.format(modelname, opt.num_epochs)
    
    ###################### model
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{0}'.format(opt.gpu))

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
    ###########################
    for i in range(1, 5):
            globals()['optimizer{0}'.format(i)] = torch.optim.Adam(globals()['model{0}'.format(i)].parameters(), lr=opt.learning_rate)  
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=opt.learning_rate)  # 1x28x28 -> 1x128x4x4 (before FC) 

    loss_CE = torch.nn.CrossEntropyLoss().cuda()
    loss_KLD = torch.nn.KLDivLoss(reduction='batchmean').cuda()

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

       
    modelpath = 'model/'+modelname+'.pth'
    if os.path.isfile(opt.pretrained):
        modelpath = opt.pretrained
        print("model load..", modelpath)
        checkpoint = torch.load(modelpath, map_location='cuda:{0}'.format(opt.gpu))
        model1.load_state_dict(checkpoint['model1_state_dict'])
        model2.load_state_dict(checkpoint['model2_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
        optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
        dropout_mask1 = checkpoint['dropout_mask1']

    else:  
        modelpath='model/{0}'.format(modelname)
        os.makedirs(modelpath, exist_ok=True)

        print("model train..")


        for epoch in range(opt.num_epochs):
            for i in range(1, 5):
                globals()['model{0}'.format(i)].train()
            #model1.train() # dropout use

            for X, Y in train_loader: 
                X = X.cuda() 
                Y = Y.cuda() 
                
                # for i in range(1, 4):
                #     globals()['prediction{0}'.format(i)] = globals()['model{0}'.format(i)](X)
                #     globals()['loss{0}'.format(i)] = loss_CE(globals()['prediction{0}'.format(i)], Y)
                #     globals()['optimizer{0}'.format(i)].zero_grad()
                #     globals()['loss{0}'.format(i)].backward()
                #     globals()['optimizer{0}'.format(i)].step()
                    
                #     globals()['predicted_classes{0}'.format(i)] \
                #         = torch.argmax(globals()['prediction{0}'.format(i)], 1) 
                #     globals()['correct_count{0}'.format(i)] \
                #         = (globals()['predicted_classes{0}'.format(i)] == Y)
                #     globals()['trainaccuracy{0}'.format(i)] \
                #         = (globals()['correct_count{0}'.format(i)].float()).sum()
                #     globals()['trainaccuracy{0}'.format(i)] /= len(Y.cpu()) * 100

                prediction1 = model1(X) 
                prediction2 = model2(X) 
                prediction3 = model3(X) 

                predicted_classes1 = torch.argmax(prediction1, 1) 
                correct_count1 = (predicted_classes1 == Y) # average of correct count 
                trainaccuracy1 = correct_count1.float().sum()
                trainaccuracy1 = trainaccuracy1 / len(Y.cpu()) *100

                predicted_classes2 = torch.argmax(prediction2, 1)
                correct_count2 = (predicted_classes2 == Y) # average of correct count 
                trainaccuracy2 = correct_count2.float().sum()
                trainaccuracy2 = trainaccuracy2 / len(Y.cpu()) *100
                
                predicted_classes3 = torch.argmax(prediction3, 1)
                correct_count3 = (predicted_classes3 == Y) # average of correct count 
                trainaccuracy3 = correct_count3.float().sum()
                trainaccuracy3 = trainaccuracy3 / len(Y.cpu()) *100

                #####                
                lossdiv2 = loss_KLD(F.log_softmax(prediction1, dim=1), F.softmax(prediction2, dim=1))
                lossdiv3 = loss_KLD(F.log_softmax(prediction2, dim=1), F.softmax(prediction1, dim=1))

                loss1 = loss_CE(prediction1, Y) 
                # loss1 = loss1+lossdiv2
                optimizer1.zero_grad()
                loss1.backward()  
                optimizer1.step()

                loss2 = loss_CE(prediction2, Y) 
                optimizer2.zero_grad() 
                loss2.backward() 
                optimizer2.step()  
                
                loss3 = loss_CE(prediction3, Y) 
                optimizer3.zero_grad() 
                loss3.backward() 
                optimizer3.step()  

                # optimizer1.zero_grad()
                # loss.backward()
                # optimizer1.step()

                agreement = (predicted_classes1 == predicted_classes2)
                disagreement = (predicted_classes1 != predicted_classes2)
                nagree = (agreement).int().sum()
                
                # pdb.set_trace()
                if epoch > 5:
                    prediction4 = model4(X[agreement])
                    loss4 = loss_CE(prediction4, Y[agreement]) 
                    optimizer4.zero_grad() 
                    loss4.backward() 
                    optimizer4.step()  

                    predicted_classes4 = torch.argmax(prediction4, 1)
                    correct_count4 = (predicted_classes4 == Y[agreement]) # average of correct count 
                    trainaccuracy4 = correct_count4.float().sum()
                    trainaccuracy4 = trainaccuracy4 / len(Y[agreement].cpu()) *100

                    ###############
                    
                    predicted_disagreement = model3(X[disagreement])
                    predicted_agreement = model3(X[agreement])
                    # pdb.set_trace()

                    minsize = min(predicted_disagreement.shape[0], \
                        predicted_agreement.shape[0])
                    maxsize = max(predicted_disagreement.shape[0], \
                        predicted_agreement.shape[0])

                    if predicted_disagreement.shape[0] > predicted_agreement.shape[0]:
                        Ltensor = predicted_disagreement
                        Stensor = predicted_agreement
                    else:
                        Ltensor = predicted_agreement
                        Stensor = predicted_disagreement
 
                    itern = int(maxsize/minsize)
                    lossdiv = 0
                    lossdiv_ = 0
                    for i in range(itern):
                        lossdiv += loss_KLD(F.log_softmax(Stensor, dim=1), F.softmax(Ltensor[i*minsize:(i+1)*minsize], dim=1))
                        lossdiv_ += loss_KLD(F.log_softmax(Ltensor[i*minsize:(i+1)*minsize], dim=1), F.softmax(Stensor, dim=1))
                    lossdiv += loss_KLD(F.log_softmax(Stensor[0:maxsize-itern*minsize], dim=1), F.softmax(Ltensor[itern*minsize-1:-1], dim=1))
                    lossdiv_ += loss_KLD(F.log_softmax(Ltensor[itern*minsize-1:-1], dim=1), F.softmax(Stensor[0:maxsize-itern*minsize], dim=1))
                    
                    lossdiv /= (itern+1)
                    lossdiv_ /= (itern+1)

                    
                    optimizer3.zero_grad() 
                    lossdiv.backward() 
                    optimizer3.step()  
                    # loss_KLD(predicted_disagreement, predicted_agreement)
                    
                    
                    
                    print('epoch : {0}, agreement : {1}/{2}, '.format(epoch, nagree, len(Y.cpu())) \
                        # + 'trainaccuracy1 : {0:0.2f}, trainaccuracy2 : {1:0.2f}, '.format(trainaccuracy1.item(), trainaccuracy2.item()) \
                            + 'trainaccuracy3 : {0:0.2f}, trainaccuracy4 : {1:0.2f}, '.format(trainaccuracy3.item(), trainaccuracy4.item()) \
                                + 'lossdiv_dis1 : {0:0.2f}, lossdiv_dis2 : {1:0.2f}, '.format(lossdiv.item(), lossdiv_.item()) \
                                    + 'lossdiv1 : {0:0.2f}, lossdiv2 : {1:0.2f} '.format(lossdiv2.item(), lossdiv3.item()), end='\r')                          
                else:

                    print('epoch : {0}, agreement : {1}/{2}, '.format(epoch, nagree, len(Y.cpu())) \
                        + 'trainaccuracy1 : {0:0.2f}, trainaccuracy2 : {1:0.2f}, '.format(trainaccuracy1.item(), trainaccuracy2.item()) \
                            + 'lossdiv1 : {0:0.2f}, lossdiv2 : {1:0.2f} '.format(lossdiv2.item(), lossdiv3.item()) , end='\r')
                            
            #######################################################


            if (epoch+1) % opt.valid_epoch == 0:
                for i in range(1, 5):
                    globals()['model{0}'.format(i)].eval()
                #model1.eval() # dropout use

                avgaccuracy1 = 0
                avgaccuracy2 = 0
                avgaccuracy3 = 0
                avgaccuracy4 = 0
                n = 0
                nagree = 0
                for X, Y in test_loader: 
                    n += X.size()[0]
                    X_test = X.cuda() 
                    Y_test = Y.cuda() 

                    prediction1 = model1(X_test) #
                    prediction2 = model2(X_test) #

                    predicted_classes1 = torch.argmax(prediction1, 1) 
                    correct_count1 = (predicted_classes1 == Y_test) 
                    testaccuracy1 = correct_count1.float().sum()
                    avgaccuracy1 += testaccuracy1
                    
                    predicted_classes2 = torch.argmax(prediction2, 1) 
                    correct_count2 = (predicted_classes2 == Y_test) 
                    testaccuracy2 = correct_count2.float().sum()
                    avgaccuracy2 += testaccuracy2
                    
                    prediction3 = model3(X_test) #
                    predicted_classes3 = torch.argmax(prediction3, 1) 
                    correct_count3 = (predicted_classes3 == Y_test) 
                    testaccuracy3 = correct_count3.float().sum()
                    avgaccuracy3 += testaccuracy3
                    
                    prediction4 = model4(X_test) #
                    predicted_classes4 = torch.argmax(prediction4, 1) 
                    correct_count4 = (predicted_classes4 == Y_test) 
                    testaccuracy4 = correct_count4.float().sum()
                    avgaccuracy4 += testaccuracy4

                    agreement = (predicted_classes1 == predicted_classes2)
                    nagree = nagree + (agreement).int().sum()


                avgaccuracy1 = (avgaccuracy1/n) *100
                avgaccuracy2 = (avgaccuracy2/n) *100
                avgaccuracy3 = (avgaccuracy3/n) *100
                avgaccuracy4 = (avgaccuracy4/n) *100


                # agreement = '{0}/{1}'.format(nagree, n)
                # data = ["epoch", "agreement", "trainaccuracy1", "trainaccuracy2", "trainaccuracy3", "testaccuracy1", \
                #     "testaccuracy2", "testaccuracy3", "lossdiv", "lossdiv2"]
                # prompt=''
                # for name in data:
                #     prompt = '{0}{1} : {2}\n'.format(prompt, name, globals()[name])   
                # f = open(resultname, 'a')
                # f.write(prompt)
                # f.close()
                
                f = open(resultname, 'a')
                f.write('epoch : {0}\n'.format(epoch))
                f.write('\tagreement : {0}/{1}\n'.format(nagree, n))
                f.write('\ttrainaccuracy1 : {0:0.2f}\n'.format(trainaccuracy1.item()))
                f.write('\ttrainaccuracy2 : {0:0.2f}\n'.format(trainaccuracy2.item()))
                f.write('\ttrainaccuracy3 : {0:0.2f}\n'.format(trainaccuracy3.item()))
                f.write('\ttrainaccuracy4 : {0:0.2f}\n'.format(trainaccuracy4.item()))
                f.write('\ttestaccuracy1 : {0:0.2f}\n'.format(avgaccuracy1.item()))
                f.write('\ttestaccuracy2 : {0:0.2f}\n'.format(avgaccuracy2.item()))
                f.write('\ttestaccuracy3 : {0:0.2f}\n'.format(avgaccuracy3.item()))
                f.write('\ttestaccuracy4 : {0:0.2f}\n'.format(avgaccuracy4.item()))
                f.write('\tloss_dis1 : {0:0.4f}\n'.format(lossdiv.item()))
                f.write('\tloss_dis2 : {0:0.4f}\n'.format(lossdiv_.item()))
                f.write('\tloss1 : {0:0.4f}\n'.format(lossdiv2.item()))
                f.write('\tloss2 : {0:0.4f}\n'.format(lossdiv3.item()))
                f.close()
                print('')
                
                modelsave = '{0}/{1}_{2}.pth'.format(modelpath, modelname, epoch)
                print(' testaccuracy1 : {0:0.2f}, '.format(avgaccuracy1.item()) \
                    + 'testaccuracy2 : {0:0.2f}, '.format(avgaccuracy2.item()) \
                        + 'testaccuracy3 : {0:0.2f}, '.format(avgaccuracy3.item()) \
                            + 'testaccuracy4 : {0:0.2f}'.format(avgaccuracy4.item()))
                print(' -> model save : ', modelsave)

                torch.save({
                            'epoch': epoch,
                            'model1_state_dict': model1.state_dict(),
                            'optimizer1_state_dict': optimizer1.state_dict(),
                            'trainaccuracy1': trainaccuracy1.item(),
                            'testaccuracy1': avgaccuracy1.item(),
                            'loss1': loss1.item(),
                            'model2_state_dict': model2.state_dict(),
                            'optimizer2_state_dict': optimizer2.state_dict(),
                            'trainaccuracy2': trainaccuracy2.item(),
                            'testaccuracy2': avgaccuracy2.item(),
                            'loss2': loss2.item(),
                            'model3_state_dict': model3.state_dict(),
                            'optimizer3_state_dict': optimizer3.state_dict(),
                            'trainaccuracy3': trainaccuracy3.item(),
                            'testaccuracy3': avgaccuracy3.item(),
                            'loss3': loss3.item(),
                            'dropout_mask1': dropout_mask1,
                            }, modelsave)



    with torch.no_grad(): 
        for i in range(1, 5):
            globals()['model{0}'.format(i)].eval()
        #model1.eval() # dropout use
        n=0
        avgaccuracy1 = 0
        avgaccuracy2 = 0
        avgaccuracy3 = 0 
        avgaccuracy4 = 0 
        for X, Y in test_loader: 
            n += X.size()[0]
            X_test = X.cuda() 
            Y_test = Y.cuda() 

            prediction1 = model1(X_test) 
            prediction2 = model2(X_test) 
            prediction3 = model3(X_test) 
            prediction4 = model4(X_test) 

            predicted_classes1 = torch.argmax(prediction1, 1) 
            predicted_classes2 = torch.argmax(prediction2, 1) 
            predicted_classes3 = torch.argmax(prediction3, 1) 
            predicted_classes4 = torch.argmax(prediction4, 1) 
            correct_count1 = (predicted_classes1 == Y_test) #
            correct_count2 = (predicted_classes2 == Y_test) #
            correct_count3 = (predicted_classes3 == Y_test) #
            correct_count4 = (predicted_classes4 == Y_test) #
            accuracy1 = correct_count1.float().sum()
            accuracy2 = correct_count2.float().sum()
            accuracy3 = correct_count3.float().sum()
            accuracy4 = correct_count4.float().sum()

            avgaccuracy1 += accuracy1
            avgaccuracy2 += accuracy2
            avgaccuracy3 += accuracy3
            avgaccuracy4 += accuracy4
        avgaccuracy1 = (avgaccuracy1/n ) *100
        avgaccuracy2 = (avgaccuracy2/n ) *100
        avgaccuracy3 = (avgaccuracy3/n ) *100
        avgaccuracy4 = (avgaccuracy4/n ) *100
        
        print('')
        
        prompt=''
        prompt=prompt+('====================================\n')
        prompt=prompt+(' Final average1 : {0:0.2f}%\n'.format(avgaccuracy1))
        prompt=prompt+(' Final average2 : {0:0.2f}%\n'.format(avgaccuracy2))
        prompt=prompt+(' Final average3 : {0:0.2f}%\n'.format(avgaccuracy3))
        prompt=prompt+(' Final average4 : {0:0.2f}%\n'.format(avgaccuracy4))
        prompt=prompt+('====================================\n')

        print(prompt)        
        f = open(resultname,'a')
        f.write(prompt)
        f.close()
    
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser = utils.add_args(arg_parser)
    opt = arg_parser.parse_args()
    
    
    opt.digitroot = _DIGIT_ROOT
    opt.prefix = _PREFIX
    main(opt)