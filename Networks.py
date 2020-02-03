from Basic_blocks import * 

class conv9(nn.Module):
    def __init__(self, in_dim=32*32, out_dim=10, p=0.5):
        super(conv9,self).__init__()
        act_fn = torch.nn.ReLU() 

        # self.conv_set9 = conv_set9(act_fn)
        self.conv_set9 = nn.Sequential(
            conv_block_3(3,128,act_fn),
            nn.MaxPool2d(kernel_size=2,ceil_mode=True),
            nn.Dropout2d(p=p),

            conv_block_3(128,256,act_fn),
            nn.MaxPool2d(kernel_size=2,ceil_mode=True),
            nn.Dropout2d(p=p),

            nn.Conv2d(256,512, kernel_size=3, stride=1, padding_mode='zeros'),
            nn.BatchNorm2d(512),
            act_fn,
            nn.Conv2d(512,256, kernel_size=1, stride=1, padding_mode='same'),
            nn.BatchNorm2d(256),
            act_fn,
            nn.Conv2d(256,128, kernel_size=1, stride=1, padding_mode='same'),
            nn.BatchNorm2d(128),
            act_fn,

            nn.AdaptiveAvgPool2d((1, 1))
            # nn.AvgPool2d(kernel_size=6)
        )
        self.linear = nn.Sequential(
            torch.nn.Linear(128, 10, bias=True),
            torch.nn.BatchNorm1d(10)
        )

    def forward(self, input_x, dropmask=None, mode=1):
        self.h1 = self.conv_set9(input_x)
        if dropmask is None:
            # Base dropout mask is 1 (Fully Connected)
            dropmask = torch.ones(self.h1.shape).cuda()

        self.h1_1 = dropmask*self.h1 # AdD dropout
        if mode == 1:
            # print(dropmask.shape)
            self.h2 = self.linear(self.h1_1.view(self.h1_1.shape[0],-1)) # FC 128->10
        elif mode == 2:
            self.h2 = self.h1_1

        return self.h2

class conv3(nn.Module):
    def __init__(self, in_dim=28*28, out_dim=10, p=0.5):
        super(conv3,self).__init__()
        act_fn = torch.nn.ReLU() 
        # self.conv_set3 = conv_set3(act_fn)

        self.conv_set3 = nn.Sequential(
            nn.Conv2d(1,32, kernel_size=1, stride=1, padding_mode='same'),
            nn.BatchNorm2d(32),
            act_fn,
            nn.MaxPool2d(kernel_size=2,ceil_mode=True),
            nn.Dropout2d(p=p),

            nn.Conv2d(32,64, kernel_size=1, stride=1, padding_mode='same'),
            nn.BatchNorm2d(64),
            act_fn,
            nn.MaxPool2d(kernel_size=2,ceil_mode=True),
            nn.Dropout2d(p=p),

            nn.Conv2d(64,128, kernel_size=1, stride=1, padding_mode='same'),
            nn.BatchNorm2d(128),
            act_fn,
            nn.MaxPool2d(kernel_size=2,ceil_mode=True)
            
        )
        self.linear = nn.Sequential(
            torch.nn.Linear(2048, 625, bias=True),
            torch.nn.BatchNorm1d(625),
            torch.nn.Linear(625, 10, bias=True),
            torch.nn.BatchNorm1d(10)
        )
        

    def forward(self, input_x, dropmask=None, mode=1):
        if mode == 1:
            self.h1 = self.conv_set3(input_x) # conv3
            if dropmask is None:
                # Base dropout mask is 1 (Fully Connected)
                dropmask = torch.ones(self.h1.shape).cuda()
            
            # print(dropmask.shape)
            self.h1_1 = dropmask*self.h1 # AdD dropout
            
            ## test mask
            # print(torch.all(torch.eq(self.h1_1, self.h1)))
            # pdb.set_trace()

            self.h2 = self.linear(self.h1_1.view(self.h1_1.shape[0], -1)) # FC 128->10

        elif mode == 2:
            self.h1 = self.conv_set3(input_x) # conv3
            if dropmask is None:
                # Base dropout mask is 1 (Fully Connected)
                dropmask = torch.ones(self.h1.shape).cuda()
            
            # print(dropmask.shape)
            self.h2 = dropmask*self.h1 # AdD dropout

        elif mode == 3:
            self.h2 = self.linear(input_x.view(input_x.shape[0], -1))

        return self.h2

