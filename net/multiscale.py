import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
# import torch_dct as DCT
from thop import profile


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.channel=channel
        self.dropout = nn.Dropout(p=0.5)
        self.layers = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                    )
    def forward(self, x):
        out = self.layers(x)
        out=self.dropout(out)
        out = out + x
        return out




class Upsample(nn.Module):
    def __init__(self, n_feat,scale):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*scale*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(scale))

    def forward(self, x):
        return self.body(x)


class Upsamplein(nn.Module):
    def __init__(self, in_feat, out_feat,scale_f):
        super(Upsamplein, self).__init__()

        self.up = nn.Upsample(scale_factor=scale_f, mode='bilinear', align_corners=True)
        self.body =nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.up(self.body(x))

class allnet(nn.Module):
    def __init__(self, channel):
        super(allnet, self).__init__()
        in_stage=31
        dim_stage=48
        # self.conv=nn.Conv2d(34,31,3,1,1)
        self.conv1 = nn.Conv2d(3, dim_stage, 3, 1, 1)
        self.up8 = Upsamplein(in_feat=in_stage, out_feat=dim_stage,scale_f=8)
        self.up4 = Upsamplein(in_feat=in_stage, out_feat=dim_stage,scale_f=4)
        self.up2 = Upsamplein(in_feat=in_stage, out_feat=dim_stage,scale_f=2)
        self.up0 = nn.Conv2d(in_stage, dim_stage, kernel_size=3, stride=1, padding=1, bias=False)

        self.convdown1 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resb1 = ResBlock(dim_stage)
        self.down1 = nn.Conv2d(dim_stage, dim_stage, 4, 2, 1, bias=False)
        self.convdown2 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resb2 = ResBlock(dim_stage)
        self.down2 = nn.Conv2d(dim_stage, dim_stage, 4, 2, 1, bias=False)
        self.convdown3 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resb3 = ResBlock(dim_stage)
        self.down3 = nn.Conv2d(dim_stage, dim_stage, 4, 2, 1, bias=False)
        self.convdown4 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resb4 = ResBlock(dim_stage)

        self.upde42 = Upsample(dim_stage,2)
        self.convdedown4 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resbde4 = ResBlock(dim_stage)
        self.upde4 = Upsample(dim_stage,2)

        self.upde34 = Upsample(dim_stage,2)
        self.upde32 = Upsample(dim_stage,2)
        self.convdedown3 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resbde3 = ResBlock(dim_stage)
        self.convdedown33 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resbde33 = ResBlock(dim_stage)
        self.upde3 = Upsample(dim_stage,2)

        self.upde28 = Upsample(dim_stage,2)
        self.upde24 = Upsample(dim_stage,2)
        self.upde22 = Upsample(dim_stage,2)
        self.convdedown2 = nn.Conv2d(dim_stage*3, dim_stage, 1, 1, 0, bias=False)
        self.resbde2 = ResBlock(dim_stage)
        self.convdedown22 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        self.resbde22 = ResBlock(dim_stage)

        self.convout=nn.Conv2d(dim_stage, 31, 1, 1, 0, bias=False)

        # self.convdedown2 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        # self.resbde2 = ResBlock(dim_stage)
        # self.convdedown1 = nn.Conv2d(dim_stage*2, dim_stage, 1, 1, 0, bias=False)
        # self.resbde1 = ResBlock(dim_stage)

        
        # self.up = nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),

    def forward(self, x,y): # x:train_lrhs  y:train_hrms
        xu8=self.up8(x)
        xu4=self.up4(x)
        xu2=self.up2(x)
        xu0=self.up0(x)
        y0 =self.conv1(y)

        en0=torch.cat((xu8,y0),dim=1)
        en0=self.convdown1(en0)
        enres = en0
        en0=self.resb1(en0)
        en1=self.down1(en0)
        en1=torch.cat((xu4,en1),dim=1)
        en1=self.resb2(self.convdown2(en1))
        en2=self.down2(en1)
        en2=torch.cat((xu2,en2),dim=1)
        en2=self.resb3(self.convdown3(en2))
        en3=self.down3(en2)
        en3=torch.cat((xu0,en3),dim=1)
        en3=self.resb4(self.convdown4(en3))

        upde43=self.upde42(en3)
        de3=torch.cat((en2,upde43),dim=1)
        de3=self.resbde4(self.convdedown4(de3))
        de3=self.upde4(de3)

        upde34=self.upde34(upde43)
        upde32=self.upde32(en2)
        de2=torch.cat((upde32,upde34),dim=1)
        de2=self.resbde3(self.convdedown3(de2))
        de2=torch.cat((de2,de3),dim=1)
        de2=self.resbde33(self.convdedown33(de2))
        de2=self.upde3(de2)

        upde28=self.upde28(upde34)
        upde24=self.upde24(upde32)
        upde22=self.upde22(en1)
        de1=torch.cat((upde22,upde24,upde28),dim=1)
        de1=self.resbde2(self.convdedown2(de1))
        de1=torch.cat((de1,de2),dim=1)
        de1=self.resbde22(self.convdedown22(de1))

        out=self.convout(enres+de1)

        return out,out,out






a=torch.rand(1,31,8,8)
b=torch.rand(1,3,64,64)
cnn=allnet(48)
flops, params = profile(cnn, inputs=(a,b ))
print('flops:{}'.format(flops))
print('params:{}'.format(params))
c,c1,c2=cnn(a,b)
print(c.shape)