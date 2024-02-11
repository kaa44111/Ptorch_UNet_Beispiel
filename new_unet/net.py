import torch
from torch import nn
from torch.nn import functional as F


# 卷积类
class Conv_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1, padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        #     第一个卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        #     第二个卷积同理
        )
    def forward(self, x):
        return self.layer(x)


#下采样类
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        # 序列构造器
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),#可以用也可以不用
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


#上采样  得到特征图，和同层的下采样图进行拼接，然后再进行卷积
class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        #转置卷积与插值法
        #用1+1的卷积降通道
        self.layer = nn.Conv2d(channel, channel//2, kernel_size=1, stride=1)

    def forward(self, x,feature_map): #feature_map 是同层之前下采样的特征图
        up = F.interpolate(x,scale_factor=2,mode='nearest')#scale_factor=2 变成原来的2倍  插值法
        out = self.layer(up)
        return torch.cat([out,feature_map],dim=1)  #n c h w 0 1 2 3


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1=Conv_Block(3, 64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64, 128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128, 256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256, 512)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512, 1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024, 512)
        self.u2=UpSample(512)
        self.c7=Conv_Block(512, 256)
        self.u3=UpSample(256)
        self.c8=Conv_Block(256, 128)
        self.u4=UpSample(128)
        self.c9=Conv_Block(128, 64)
        self.out=nn.Conv2d(64, 3,kernel_size=1, stride=1, padding=1)
        self.Th=nn.Sigmoid()#激活函数，虽然是彩色图像，只需要对图像进行二分类即可，也就是讲一个像素点分为黑色或白色

    def forward(self, x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))#先对R1进行下采样，然后再进行第二次的一个卷积
        R3=self.c3(self.d2(R2))
        R4=self.c4(self.d3(R3))
        R5=self.c5(self.d4(R4))
        O1=self.c6(self.u1(R5,R4))  #先将R5上采样，并且和R4进行拼接，然后再进行卷积
        O2=self.c7(self.u2(O1,R3))
        O3=self.c8(self.u3(O2,R2))
        O4=self.c9(self.u4(O3,R1))

        return self.Th(self.out(O4))

#测试，输出模型的形状，看看和我们设计的样子是否一样
if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    net = UNet()
    print(net(x).shape)