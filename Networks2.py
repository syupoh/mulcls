from Basic_blocks import * 
import numpy as np
import warnings
from torchvision import models

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

# For SVHN dataset
class DTN(nn.Module):
    def __init__(self):
        super(DTN, self).__init__()
        self.conv_params = nn.Sequential (
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
    
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout()
                )

        self.classifier = nn.Linear(512, 10)
        self.__in_features = 512

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def load(self, init_path):
        net_init_dict = torch.load(init_path)
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)


class Classifier(nn.Module):
    def __init__(self, input_size, class_num):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size,240),
            nn.ReLU(),
            nn.Linear(240,180),
            nn.ReLU(),
            nn.Linear(180,120),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(120,class_num)
        )
    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = self.fc1(x)
        return x


class Generator_digits(nn.Module):
    def __init__(self, input_size, num_feature):
        super(Generator_digits, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch,1024
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 32, 32
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 32, 32
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 2, 2, stride=2),  # batch, 2, 16, 16
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 32, 32)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = x.view(x.size(0),512)
        return x


class Discriminator_digits(nn.Module):
    def __init__(self):
        super(Discriminator_digits, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2),  # batch, 32, 16, 16
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 16, 16
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 8, 8
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x: batch, width, height, channel=1
        '''
        x = x.view(x.size(0), 2, 16, 16)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x_ = self.fc(x)
        x = self.fc2(x)
        return x_, x


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

        
"""Modified VGG16 to compute perceptual loss.
This class is mostly copied from pytorch/examples.
See, fast_neural_style in https://github.com/pytorch/examples.
"""


class VGG_OUTPUT(object):

    def __init__(self, relu1_2, relu2_2, relu3_3, relu4_3):
        self.__dict__ = locals()


class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return VGG_OUTPUT(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)