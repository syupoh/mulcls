from Networks import *

def add_args(parser):
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--valid_epoch', type=int, default=10)
    parser.add_argument('--print_delay', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--norm', type=bool, default=None)
    parser.add_argument('--digitroot', type=str, default='~/dataset/digits/')
    parser.add_argument('--pretrained', type=str, default='model/.pth')
    parser.add_argument('--dropout_probability', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--loss4_KLD_dis_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--mu', type=float, default=10)
    return parser


def load_data(opt):
    batch_size = opt.batch_size
    modelsplit = opt.model.split('_')

    tf_to32 = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    if opt.norm == True:
        tf_to32 = transforms.Compose([
            tf_to32,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -1 ~ 1
        ])

    
    if opt.norm == True:
        if modelsplit[0] == 'mnist' and modelsplit[1] == 'mnist':
            tf_toTensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)), # -1 ~ 1
            ])

        else:
            tf_toTensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # -1 ~ 1
            ])

    else:
        tf_toTensor = transforms.Compose([
            transforms.ToTensor()
        ])


    if modelsplit[0] == 'mnist' or modelsplit[1] == 'mnist':
        if modelsplit[0] =='svhn' or modelsplit[1] == 'svhn':
            tf_MNIST = tf_to32
        else:
            tf_MNIST = tf_toTensor

    

    digitroot = opt.digitroot
    if modelsplit[0] == 'mnist':
        trainset = dset.MNIST(root=digitroot+'MNIST_data/', train=True, transform=tf_MNIST, download=True)
    elif modelsplit[0] == 'usps':
        trainset = dset.USPS(root=digitroot+'USPS_data/', train=True, transform=tf_to32, download=True)
    elif modelsplit[0] == 'svhn':
        trainset = dset.SVHN(root=digitroot+'svhn_data/', split='train', transform=tf_toTensor, download=True)
    elif modelsplit[0] == 'cifar10':
        trainset = dset.CIFAR10(root=digitroot+'cifar10_data/', train=True, transform=tf_to32, download=True)
    elif modelsplit[0] == 'stl10':
        trainset = dset.STL10(root=digitroot+'STL10_data/', split='train', transform=tf_to32, download=True)
     
    # trainset.data.shape[-2] # height
    # trainset.data.shape[-1] # width

    if modelsplit[1] == 'mnist':
        trainset2 = dset.MNIST(root=digitroot+'MNIST_data/', train=True, transform=tf_MNIST, download=True) 
    elif modelsplit[1] == 'usps':
        trainset2 = dset.USPS(root=digitroot+'USPS_data/', train=True, transform=tf_to32, download=True) 
    elif modelsplit[1] == 'svhn':
        trainset2 = dset.SVHN(root=digitroot+'svhn_data/', split='train', transform=tf_toTensor, download=True) 
    elif modelsplit[1] == 'cifar10':
        trainset2 = dset.CIFAR10(root=digitroot+'cifar10_data/', train=True, transform=tf_to32, download=True)
    elif modelsplit[1] == 'stl10':
        trainset2 = dset.STL10(root=digitroot+'STL10_data/', split='train', transform=tf_to32, download=True)


    if modelsplit[1] == 'mnist':
        testset = dset.MNIST(root=digitroot+'MNIST_data/', train=False, transform=tf_MNIST, download=True) 
    elif modelsplit[1] == 'usps':
        testset = dset.USPS(root=digitroot+'USPS_data/', train=False, transform=tf_to32, download=True) 
    elif modelsplit[1] == 'svhn':
        testset = dset.SVHN(root=digitroot+'svhn_data/', split='test', transform=tf_toTensor, download=True) 
    elif modelsplit[1] == 'cifar10':
        testset = dset.CIFAR10(root=digitroot+'cifar10_data/', train=False, transform=tf_to32, download=True)
    elif modelsplit[1] == 'stl10':
        testset = dset.STL10(root=digitroot+'STL10_data/', split='test', transform=tf_to32, download=True)

    # testset.data.shape[-2] # height
    # testset.data.shape[-1] # width
    
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True) # model
    # train_loader2 = torch.utils.data.DataLoader(trainset2, batch_size=batch_size, shuffle=True, drop_last=True) # model
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True) # model

    return trainset, trainset2, testset
