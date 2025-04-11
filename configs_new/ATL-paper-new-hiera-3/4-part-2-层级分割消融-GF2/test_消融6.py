import torch
from torch import nn

class merge_block(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()
        
        self.sigmoid=nn.Sigmoid()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1) # [2,4,128,128]-->[2,4,1,1]  
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1) # [2,4,128,128]-->[2,4,1,1]  
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=out_channels,bias=False), # [2,4,1,1]-->[2,1024,1,1]
            nn.ReLU())
        # L1 spatial_attentation
        self.spatial_conv = nn.Conv2d(in_channels=2, 
                                        out_channels=1, 
                                        kernel_size=7 ,
                                        stride=1,
                                        padding=7//2,
                                        bias=False)
        self.channel_conv_1x1 = nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=1, 
                                            stride=1, 
                                            padding=0, 
                                            bias=False)
        

    def forward(self, convseg_outputs): # 这里传入这个 会不会把他改变了啊。还是稳妥一点，传一个copy进来吧
        # channel
        max_out_channel_att = self.max_pool(convseg_outputs)
        max_out_channel_att = self.mlp(max_out_channel_att.view(max_out_channel_att.size(0),-1))  # [2,4,1,1]-->[2,9]-->[2,18] or 反过来
        avg_out_channel_att = self.avg_pool(convseg_outputs)
        avg_out_channel_att = self.mlp(avg_out_channel_att.view(avg_out_channel_att.size(0),-1)) # [2,4,1,1]-->[2,9]-->[2,18] or 反过来
        channel_att_out = self.sigmoid(max_out_channel_att+avg_out_channel_att) # [2,9]-->[2,18] or 反过来
        import pdb;pdb.set_trace()
        channel_att_out = channel_att_out.view(channel_att_out.size(0), channel_att_out.size(1),1,1) #[2,9]-->[2,9,1,1]
        # spatial
        max_out_spatial_att, _ = torch.max(convseg_outputs, dim=1, keepdim=True) # [2,4,128,128]-->[2,1,128,128]
        mean_out_spatial_att = torch.mean(convseg_outputs, dim=1, keepdim=True) # [2,4,128,128]-->[2,1,128,128]
        spatial_att_out = torch.cat((max_out_spatial_att, mean_out_spatial_att), dim=1) #[2,2,128,128]
        spatial_att_out = self.sigmoid(self.spatial_conv(spatial_att_out)) #[2,2,128,128]-->[2,1,128,128]
                        # [2,9,1,1] * [2,9,128,128] * [2,1,128,128] #用了广播机制
        # 原始特征  1x1 # 这里有个问题，没用原始特征了啊？
        convseg_outputs_att = self.channel_conv_1x1(convseg_outputs) # [2,4,128,128]-->[2,9,128,128]
        convseg_outputs_att = channel_att_out * convseg_outputs_att * spatial_att_out

        return convseg_outputs_att
    


num_classes_level_list = [4,9,18]
stage1_merge_L1_to_L2 = merge_block(num_classes_level_list[0], num_classes_level_list[1])
conv_seg_out = torch.randn(2, 4, 160, 160) # [2,4,128,128]
conv_seg_out.shape
a = stage1_merge_L1_to_L2(conv_seg_out)