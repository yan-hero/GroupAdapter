from PIL import Image
import torch
import torch.nn as nn
import math
from torchvision import transforms

class Sampled_DCT2(nn.Module):
    def __init__(self,args, block_size=8,device='cpu'):

        super(Sampled_DCT2, self).__init__()
        ### forming the cosine transform matrix
        self.args = args
        self.block_size = block_size
        self.device = device
        self.Q = torch.zeros((self.block_size, self.block_size))
        self.Q[0,:] = math.sqrt(1.0 / float(self.block_size))
        for i in range(1, self.block_size, 1):
            for j in range(self.block_size):
                self.Q[i, j] = math.sqrt(2.0 / float(self.block_size)) * math.cos(
                    float((2 * j + 1) * math.pi * i) / float(2.0 * self.block_size))

        self.total = self.block_size * self.block_size
        self.truncate_line = 250
        self.loc = torch.zeros(self.total, 2)
        if self.block_size == 8:
            self.loc[0] = torch.tensor([0, 0])
            self.loc[1] = torch.tensor([0, 1])
            self.loc[2] = torch.tensor([1, 0])
            self.loc[3] = torch.tensor([2, 0])
            self.loc[4] = torch.tensor([1, 1])
            self.loc[5] = torch.tensor([0, 2])
            self.loc[6] = torch.tensor([0, 3])
            self.loc[7] = torch.tensor([1, 2])
            self.loc[8] = torch.tensor([2, 1])
            self.loc[9] = torch.tensor([3, 0])
            self.loc[10] = torch.tensor([4, 0])
            self.loc[11] = torch.tensor([3, 1])
            self.loc[12] = torch.tensor([2, 2])
            self.loc[13] = torch.tensor([1, 3])
            self.loc[14] = torch.tensor([0, 4])
            self.loc[15] = torch.tensor([0, 5])
            self.loc[16] = torch.tensor([1, 4])
            self.loc[17] = torch.tensor([2, 3])
            self.loc[18] = torch.tensor([3, 2])
            self.loc[19] = torch.tensor([4, 1])
            self.loc[20] = torch.tensor([5, 0])
            self.loc[21] = torch.tensor([6, 0])
            self.loc[22] = torch.tensor([5, 1])
            self.loc[23] = torch.tensor([4, 2])
            self.loc[24] = torch.tensor([3, 3])
            self.loc[25] = torch.tensor([2, 4])
            self.loc[26] = torch.tensor([1, 5])
            self.loc[27] = torch.tensor([0, 6])
            self.loc[28] = torch.tensor([0, 7])
            self.loc[29] = torch.tensor([1, 6])
            self.loc[30] = torch.tensor([2, 5])
            self.loc[31] = torch.tensor([3, 4])
            self.loc[32] = torch.tensor([4, 3])
            self.loc[33] = torch.tensor([5, 2])
            self.loc[34] = torch.tensor([6, 1])
            self.loc[35] = torch.tensor([7, 0])
            self.loc[36] = torch.tensor([7, 1])
            self.loc[37] = torch.tensor([6, 2])
            self.loc[38] = torch.tensor([5, 3])
            self.loc[39] = torch.tensor([4, 4])
            self.loc[40] = torch.tensor([3, 5])
            self.loc[41] = torch.tensor([2, 6])
            self.loc[42] = torch.tensor([1, 7])
            self.loc[43] = torch.tensor([2, 7])
            self.loc[44] = torch.tensor([3, 6])
            self.loc[45] = torch.tensor([4, 5])
            self.loc[46] = torch.tensor([5, 4])
            self.loc[47] = torch.tensor([6, 3])
            self.loc[48] = torch.tensor([7, 2])
            self.loc[49] = torch.tensor([7, 3])
            self.loc[50] = torch.tensor([6, 4])
            self.loc[51] = torch.tensor([5, 5])
            self.loc[52] = torch.tensor([4, 6])
            self.loc[53] = torch.tensor([3, 7])
            self.loc[54] = torch.tensor([4, 7])
            self.loc[55] = torch.tensor([5, 6])
            self.loc[56] = torch.tensor([6, 5])
            self.loc[57] = torch.tensor([7, 4])
            self.loc[58] = torch.tensor([7, 5])
            self.loc[59] = torch.tensor([6, 6])
            self.loc[60] = torch.tensor([5, 7])
            self.loc[61] = torch.tensor([6, 7])
            self.loc[62] = torch.tensor([7, 6])
            self.loc[63] = torch.tensor([7, 7])

    def rgb_to_ycbcr(self, input):

        # input is mini-batch N x 3 x H x W of an RGB image
        # output = Variable(input.data.new(*input.size())).to(self.device)
        output = torch.zeros_like(input).cuda()
        input = (input * 255.0)
        output[:, 0, :, :] = input[:, 0, :, :] * 0.299 + input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.114
        output[:, 1, :, :] = input[:, 0, :, :] * -0.168736 - input[:, 1, :, :] * 0.331264 + input[:, 2, :,:] * 0.5 + 128
        output[:, 2, :, :] = input[:, 0, :, :] * 0.5 - input[:, 1, :, :] * 0.418688 - input[:, 2, :, :] * 0.081312 + 128

        return output / 255.0

    def ycbcr_to_freq(self, input):

        input = input[:,0,:,:].unsqueeze(1)  #use Y channel,do not use Cb,Cr channel
        a = int(input.shape[2] / self.block_size)
        b = int(input.shape[3] / self.block_size)
        image_dct_result = torch.zeros_like(input).to(input.device)
        self.Q = self.Q.to(input.device)
        dct_coeff_distribute = torch.zeros(size=(input.shape[0],self.total,a*b)).to(input.device)
        # Compute DCT in block_size x block_size blocks
        cnt = 0
        for i in range(a):
            for j in range(b):
                #[batch_size,channel,8,8]
                dctcoeff = torch.matmul(
                    torch.matmul(self.Q, input[:, :, i * self.block_size: (i + 1) * self.block_size,j * self.block_size: (j + 1) * self.block_size]),
                    self.Q.permute(1, 0).contiguous())

                image_dct_result[:,:,i * self.block_size: (i + 1) * self.block_size,j * self.block_size: (j + 1) * self.block_size]=dctcoeff
                dctcoeff = dctcoeff.squeeze(1)
                for k in range(len(self.loc)):
                    m,n = self.loc[k]
                    dct_coeff_distribute[:,k,cnt] = dctcoeff[:,int(m),int(n)]
                cnt += 1

        #process every frequency dct_coefficient distribution
        # max = torch.zeros(dct_coeff_distribute.shape[0],dct_coeff_distribute.shape[1])
        # for k in range(len(self.loc)):
        #     max[:,k],_ = torch.max(torch.abs(dct_coeff_distribute[:,k,:]),dim=-1)
        # print('max:\n',max)
        # print(max.shape)
        torch.manual_seed(self.args.seed)
        dct_coeff_sample = torch.zeros(size=(dct_coeff_distribute.shape[0],dct_coeff_distribute.shape[1],250)).to(input.device)
        for i in range(dct_coeff_sample.shape[0]):
            for j in range(dct_coeff_sample.shape[1]):
                dct_coeff_sample[i,j] = dct_coeff_distribute[i,j][torch.randperm(dct_coeff_distribute.shape[2])[:250]]

        return image_dct_result,dct_coeff_sample

    def forward(self, x):
        # return self.ycbcr_to_freq(self.rgb_to_ycbcr(x))

        if (x.shape[1] == 3):
            return self.ycbcr_to_freq(self.rgb_to_ycbcr(x))
        else:
            return self.ycbcr_to_freq(x)

class ConvBlock(nn.Sequential):
    def __init__(self,in_channel,out_channel,kernel_size=(3,1)):
        padding = (1,0) if kernel_size==(3,1) else (0,0)

        super(ConvBlock, self).__init__(
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,padding=padding,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

class Inception(nn.Module):
    def __init__(self,in_channel):
        super(Inception, self).__init__()
        self.branch1 = ConvBlock(in_channel,64,kernel_size=(1,1))
        self.branch2 = nn.Sequential(ConvBlock(in_channel,48,kernel_size=(1,1)),
                                     ConvBlock(48,64))

        self.branch3 = nn.Sequential(ConvBlock(in_channel,64,kernel_size=(1,1)),
                                     ConvBlock(64,96),
                                     ConvBlock(96,96))

        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=(3,1), stride=(1,1), padding=(1,0)),
                                     ConvBlock(in_channel,32,kernel_size=1))

    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1,branch2,branch3,branch4],dim=1)


class CNN_Frequency(nn.Module):
    def __init__(self,in_channel):
        super(CNN_Frequency, self).__init__()
        layers = []
        layers.extend([ConvBlock(in_channel,32),ConvBlock(32,64),ConvBlock(64,128)])
        self.conv = nn.Sequential(*layers)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2,1),stride=(2,1))
        self.Inception_V3 = Inception(128)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(122,1),stride=(122,1))
        self.last_conv = ConvBlock(256,64,kernel_size=(1,1))


    def forward(self,freq_input):
        #[B,64,250]->[B,64,250,1]
        input = freq_input.unsqueeze(3)
        x = self.conv(input)
        x = self.maxpool_1(x)
        x = self.Inception_V3(x)
        x = self.maxpool_2(x)
        # [B,64]
        x = self.last_conv(x).flatten(1)
        return x

class Frequency_extractor(nn.Module):
    def __init__(self,args,init_weight=False):
        super(Frequency_extractor, self).__init__()
        self.dct_net = Sampled_DCT2(args)
        self.cnn_freq = CNN_Frequency(64)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,input):
        _,dct_coef_distribute = self.dct_net(input)
        out = self.cnn_freq(dct_coef_distribute)

        return out


# if __name__ == '__main__':
#
#     image_1 = Image.open('F:\pytorch_project\plane.jpg')
#     image_2 = Image.open('F:\pytorch_project\plane.jpg')
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor()
#     ])
#     image_1 = transform(image_1)
#     image_2 = transform(image_2)
#     image = torch.stack([image_1,image_2],dim=0)
#
#     print(image.shape)
#     net = Sampled_DCT2()
#     _,dct_distribute = net(image)
#     print(dct_distribute.shape)
#     cnn_net = CNN_Frequency(64)
#     output = cnn_net(dct_distribute)
#     print(output.shape)

    # def DFT(x):
    #     N = len(x)
    #     X = np.zeros(N,dtype=np.complex_)
    #     for k in range(N):
    #         for n in range(N):
    #             X[k] += x[n]*np.exp(-2j*np.pi*k*n/N)
    #     return X
    # dct_distribute_numpy = np.zeros(shape=dct_distribute.shape,dtype=complex)
    # for i in range(dct_distribute.shape[0]):
    #     for j in range(dct_distribute.shape[1]):
    #         dct_distribute_numpy[i,j] = DFT(dct_distribute[i,j].numpy())

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(4,4)
    # for i in range(ax.shape[0]):
    #     for j in range(ax.shape[1]):
    #         ax[i][j].plot(dct_distribute[1,i*8+j])
    # plt.show()