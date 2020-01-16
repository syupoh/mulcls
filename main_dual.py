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


opt = argparse.ArgumentParser()
opt = utils.argparser(opt)


# opt.model = 'svhn_mnist'
# opt.model = 'mnist_usps'
# opt.model = 'usps_mnist'
# opt.model = 'cifar10_stl10'
# opt.model = 'stl10_cifar10'

# opt.model = 'svhn_svhn'
# opt.model = 'mnist_mnist'
# opt.model = 'usps_usps'
# opt.model = 'svhn_usps'

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

#########################
resultname = './result/result_{0}_{1}_{2}.txt'.format(opt.prefix, opt.model, opt.num_epochs)
#### DATASET 
modelsplit = opt.model.split('_')

train_loader, test_loader = utils.load_data(prefix=opt.prefix, opt=opt)


###################### model
torch.cuda.set_device(opt.gpu)
device = torch.device('cuda:{0}'.format(opt.gpu))

if modelsplit[0] == 'svhn' or modelsplit[1] == 'svhn' or \
    modelsplit[0] == 'usps' or modelsplit[0] == 'cifar10' or \
        modelsplit[0] == 'stl10':

    model1 = conv9(p=opt.dropout_probability).cuda() # 3x32x32 -> 1x128x1x1 (before FC) 
    model2 = conv9(p=opt.dropout_probability).cuda() # 3x32x32 -> 1x128x1x1 (before FC) 
    model3 = conv9(p=opt.dropout_probability).cuda() # 3x32x32 -> 1x128x1x1 (before FC) 
else:
    model1 = conv3(p=opt.dropout_probability).cuda() # 1x28x28 -> 1x128x4x4 (before FC)
    model2 = conv3(p=opt.dropout_probability).cuda()
    model3 = conv3(p=opt.dropout_probability).cuda()

dropout_mask1 = torch.randint(2, (1, 128, 1, 1), dtype=torch.float).cuda()
# dropout_mask1 = torch.randint(2,(1,128,4,4), dtype=torch.float).cuda()
###########################
optimizer1 = torch.optim.Adam(model1.parameters(), lr=opt.learning_rate) 
optimizer2 = torch.optim.Adam(model2.parameters(), lr=opt.learning_rate) 
optimizer3 = torch.optim.Adam(model3.parameters(), lr=opt.learning_rate) 
loss_function = torch.nn.CrossEntropyLoss().cuda()

prompt=''
prompt=prompt+('====================================\n')
prompt=prompt+('modelname : {0}\n'.format(opt.model))
prompt=prompt+('num_epochs : {0}\n'.format(opt.num_epochs))
prompt=prompt+('dropout_probability : {0}\n'.format(opt.dropout_probability))
prompt=prompt+('learning_rate : {0}\n'.format(opt.learning_rate))
prompt=prompt+('gpu : {0}\n'.format(opt.gpu))
prompt=prompt+('batch_size : {0}\n'.format(opt.batch_size))
prompt=prompt+('====================================\n')
print(prompt)

f = open(resultname,'w')
f.write(prompt)
# f.write('modelname : {0}\n'.format(modelname))
# f.write('num_epochs : {0}\n'.format(num_epochs))
# f.write('dropout_probability : {0}\n'.format(opt.dropout_probability))
# f.write('learning_rate : {0}\n'.format(learning_rate))
# f.write('gpu : {0}\n'.format(opt.gpu))
# f.write('batch_size : {0}\n'.format(batch_size))
# f.write('====================================\n')
f.close()

modelpath = 'model/'+opt.model+'.pth'
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
    modelpath='model/{0}'.format(opt.model)
    os.makedirs(modelpath, exist_ok=True)

    print("model train..")
    model1.train() # dropout use
    model2.train() # dropout use
    for epoch in range(opt.num_epochs): 
        avg_loss = 0 
        for X, Y in train_loader: 
            X = X.cuda() 
            Y = Y.cuda() 
            
            # prediction = model(X, dropout_mask1) 
            prediction1 = model1(X) 
            loss1 = loss_function(prediction1, Y) 
            optimizer1.zero_grad()
            loss1.backward()  
            optimizer1.step()

            prediction2 = model2(X) 
            loss2 = loss_function(prediction2, Y) 
            optimizer2.zero_grad() 
            loss2.backward() 
            optimizer2.step()  

            predicted_classes1 = torch.argmax(prediction1, 1) 
            correct_count1 = (predicted_classes1 == Y) # average of correct count 
            trainaccuracy1 = correct_count1.float().sum()
            trainaccuracy1 = trainaccuracy1 / len(Y.cpu()) *100

            predicted_classes2 = torch.argmax(prediction2, 1)
            correct_count2 = (predicted_classes2 == Y) # average of correct count 
            trainaccuracy2 = correct_count2.float().sum()
            trainaccuracy2 = trainaccuracy2 / len(Y.cpu()) *100
            
            agree = (predicted_classes1 == predicted_classes2).int().sum()
            # pdb.set_trace()
            if epoch > 5:
                
                prediction3 = model3(X[predicted_classes1 == predicted_classes2])
                loss3 = loss_function(prediction3, Y[predicted_classes1 == predicted_classes2]) 
                optimizer3.zero_grad() 
                loss3.backward() 
                optimizer3.step()  

                
                predicted_classes3 = torch.argmax(prediction3, 1)
                correct_count3 = (predicted_classes3 == Y[predicted_classes1 == predicted_classes2]) # average of correct count 
                trainaccuracy3 = correct_count3.float().sum()
                trainaccuracy3 = trainaccuracy3 / len(Y[predicted_classes1 == predicted_classes2].cpu()) *100
                
                print('epoch : {0}, agreement : {1}/{2}, trainaccuracy1 : {3:0.2f}, trainaccuracy2 : {4:0.2f},  trainaccuracy3 : {5:0.2f}'.format(
                    epoch,agree,len(Y.cpu()),trainaccuracy1.item(),trainaccuracy2.item(),trainaccuracy3.item()), end='\r') 
                    
            print('epoch : {0}, agreement : {1}/{2}, trainaccuracy1 : {3:0.2f}, trainaccuracy2 : {4:0.2f}'.format(
                epoch,agree,len(Y.cpu()),trainaccuracy1.item(),trainaccuracy2.item()), end='\r')  
        #######################################################


        if epoch % opt.valid_epoch == 0:
            avgaccuracy1 = 0
            avgaccuracy2 = 0
            avgaccuracy3 = 0
            n=0
            agree = 0
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
                
                agree = agree + (predicted_classes1 == predicted_classes2).int().sum()
                if epoch > 5:
                    prediction3 = model3(X_test) #
                    predicted_classes3 = torch.argmax(prediction3, 1) 
                    correct_count3 = (predicted_classes3 == Y_test) 
                    testaccuracy3 = correct_count3.float().sum()

                    avgaccuracy3 += testaccuracy3

            avgaccuracy1 = (avgaccuracy1/n) *100
            avgaccuracy2 = (avgaccuracy2/n) *100
            if epoch > 5:
                avgaccuracy3 = (avgaccuracy3/n) *100
            # avgaccuracy4 = (avgaccuracy4/n) *100

            f = open(resultname,'a')
            f.write('epoch : {0}\n'.format(epoch))
            f.write('\tagreement : {0}/{1}\n'.format(agree,n))
            f.write('\ttrainaccuracy1 : {0:0.2f}\n'.format(trainaccuracy1.item()))
            f.write('\ttrainaccuracy2 : {0:0.2f}\n'.format(trainaccuracy2.item()))
            
            if epoch > 5:
                f.write('\ttrainaccuracy3 : {0:0.2f}\n'.format(trainaccuracy3.item()))

            f.write('\ttestaccuracy1 : {0:0.2f}\n'.format(avgaccuracy1.item()))
            f.write('\ttestaccuracy2 : {0:0.2f}\n'.format(avgaccuracy2.item()))
            
            if epoch > 5:
                f.write('\ttestaccuracy3 : {0:0.2f}\n'.format(avgaccuracy3.item()))
            f.close()
            print('')
            
            modelsave = '{0}/{1}_{2}.pth'.format(modelpath,opt.model,epoch)
            print(' testaccuracy : {0:0.2f}'.format(avgaccuracy1.item()))
            print(' -> model save : ', modelsave)

            if epoch > 5:
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
            else:
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
                            'dropout_mask1': dropout_mask1,
                            }, modelsave)
        




with torch.no_grad(): 
    model1.eval()  
    model2.eval()  
    model3.eval() 
    n=0
    avgaccuracy1 = 0
    avgaccuracy2 = 0
    for X, Y in test_loader: 
        n += X.size()[0]
        X_test = X.cuda() 
        Y_test = Y.cuda() 

        prediction1 = model1(X_test) 
        prediction2 = model2(X_test) 
        prediction3 = model3(X_test) 

        predicted_classes1 = torch.argmax(prediction1, 1) 
        predicted_classes2 = torch.argmax(prediction2, 1) 
        predicted_classes3 = torch.argmax(prediction3, 1) 
        correct_count1 = (predicted_classes1 == Y_test) #
        correct_count2 = (predicted_classes2 == Y_test) #
        correct_count3 = (predicted_classes3 == Y_test) #
        accuracy1 = correct_count1.float().sum()
        accuracy2 = correct_count2.float().sum()
        accuracy3 = correct_count3.float().sum()

        avgaccuracy1 += accuracy1
        avgaccuracy2 += accuracy2
        avgaccuracy3 += accuracy3
    avgaccuracy1 = (avgaccuracy1/n ) *100
    avgaccuracy2 = (avgaccuracy2/n ) *100
    avgaccuracy3 = (avgaccuracy3/n ) *100
    
    print('')
    print(' Final average1 : {0:0.2f}%'.format(avgaccuracy1.item()))
    print(' Final average2 : {0:0.2f}%'.format(avgaccuracy2.item()))
    print(' Final average3 : {0:0.3f}%'.format(avgaccuracy2.item()))
        
    
    f = open(resultname,'a')
    f.write('Final average1 : {0:0.2f}\n'.format(avgaccuracy1.item()))
    f.write('Final average2 : {0:0.2f}\n'.format(avgaccuracy2.item()))
    f.write('Final average3 : {0:0.2f}\n'.format(avgaccuracy3.item()))
    f.close()
   