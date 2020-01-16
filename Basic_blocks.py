import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.autograd import Variable


def conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding_mode='same'),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn)
    )
    return model


# def conv_set9(act_fn):
#     model = nn.Sequential(
#         conv_block_3(3,128,act_fn),
#         maxpool(),
#         nn.Dropout2d(p=0.5),

#         conv_block_3(128,256,act_fn),
#         maxpool(),
#         nn.Dropout2d(p=0.5),

#         nn.Conv2d(256,512, kernel_size=3, stride=1, padding=0),
#         nn.BatchNorm2d(512),
#         act_fn,
#         nn.Conv2d(512,256, kernel_size=1, stride=1, padding=1),
#         nn.BatchNorm2d(256),
#         act_fn,
#         nn.Conv2d(256,128, kernel_size=1, stride=1, padding=1),
#         nn.BatchNorm2d(128),
#         act_fn,
#         nn.AdaptiveAvgPool2d((1, 1))
#         # nn.AvgPool2d(kernel_size=6)
#     )
#     return model

# def conv_set3(act_fn):
#     model = nn.Sequential(
#         nn.Conv2d(1,32, kernel_size=1, stride=1, padding=1),
#         nn.BatchNorm2d(32),
#         act_fn,
#         maxpool(),
#         nn.Dropout2d(p=0.5),

#         nn.Conv2d(32,64, kernel_size=1, stride=1, padding=1),
#         nn.BatchNorm2d(64),
#         act_fn,
#         maxpool(),
#         nn.Dropout2d(p=0.5),

#         nn.Conv2d(64,128, kernel_size=1, stride=1, padding=1),
#         nn.BatchNorm2d(128),
#         act_fn,
#         maxpool(),
        
#     )
#     return model
    