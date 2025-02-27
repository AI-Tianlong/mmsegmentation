import torch
import torch.nn as nn
import math

class Se(nn.Module):
    def __init__(self,in_channel, reduction=16):
        super().__init__()
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.fc=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        import pdb; pdb.set_trace()
        out=self.pool(x) # [2,1024,1,1] 每个通道一个注意力
        out=self.fc(out.view(out.size(0),-1)) # [2,1024]
        out=out.view(x.size(0),x.size(1),1,1) # [2,1024,1,1]
        return out*x # [2,1024,128,128]*[2,1024,1,1] 用了Python的广播机制

class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        import pdb; pdb.set_trace()
        out=self.pool(x) # [2,1024,1,1]
        out=out.view(x.size(0),1,x.size(1)) # [2,1,1024]
        out=self.conv(out) # [2,1,1024]
        out=out.view(x.size(0),x.size(1),1,1) # # [2,1024,1,1]
        return out*x

class CBAM(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        #通道注意力机制
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #空间注意力机制
        self.conv=nn.Conv2d(in_channels=2, out_channels=1, 
                            kernel_size=kernel_size ,
                            stride=1,
                            padding=kernel_size//2,bias=False)

    def forward(self,x):
        #通道注意力机制
        import pdb; pdb.set_trace()
        maxout=self.max_pool(x) # [2, 1024, 128, 128] -->  [2, 1024, 1, 1]    
        maxout=self.mlp(maxout.view(maxout.size(0),-1))  # [2, 1024, 1, 1]-->[2, 1024]
        avgout=self.avg_pool(x) # [2, 1024, 128, 128] -->  [2, 1024, 1, 1]
        avgout=self.mlp(avgout.view(avgout.size(0),-1))  # [2, 1024, 1, 1]-->[2, 1024]
        channel_out=self.sigmoid(maxout+avgout) # [2, 1024]
        channel_out=channel_out.view(x.size(0),x.size(1),1,1) # [2, 1024,1,1] 
        channel_out=channel_out*x  #广播机制
        #空间注意力机制
        max_out,_=torch.max(channel_out,dim=1,keepdim=True) # [2,1024,128,128] --> [2,1,128,128]
        mean_out=torch.mean(channel_out,dim=1,keepdim=True) # [2,1024,128,128] --> [2,1,128,128]
        out=torch.cat((max_out,mean_out),dim=1) # [[2,1,128,128],[2,1,128,128]] --> [2,2,128,128]
        out=self.sigmoid(self.conv(out)) # [2,2,128,128], 两个通道变成一个通道了。
        out=out*channel_out # 然后再乘上系数。
        return out

if __name__ == '__main__':

    mode = 'CBAM'

    if mode == 'SE':
        channel_attentaion=Se(in_channel=1024)
        outputs = torch.randn(2, 1024, 128, 128)
        outputs = channel_attentaion(outputs)
        print(outputs.shape)

    elif mode == 'ECA':
        channel_attentaion=ECA(in_channel=1024)
        outputs = torch.randn(2, 1024, 128, 128)
        outputs = channel_attentaion(outputs)
        print(outputs.shape)
    
    elif mode == 'CBAM':
        channel_attentaion=CBAM(in_channel=1024)
        outputs = torch.randn(2, 1024, 128, 128)
        outputs = channel_attentaion(outputs)
        print(outputs.shape)