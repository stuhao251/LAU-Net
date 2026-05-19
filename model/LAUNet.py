import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer
def Conv1x1(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, padding=0, bias=bias)
    return layer
class Conv3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out
class Conv1Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(Conv1Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class IRM_D(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(IRM_D, self).__init__()
        self.up_branch = nn.Sequential(
            Conv1Block(in_channels, in_channels),
            Conv3Block(in_channels, in_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )
        self.down_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )
    def forward(self, x):
        up = self.up_branch(x)
        down = self.down_branch(x)
        out = up+down
        return out
# x = torch.randn(1, 64, 128, 128)
# irm=IRM_D(64,128)
# print(irm(x).shape)
class IRM_U(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(IRM_U, self).__init__()
        self.up_branch = nn.Sequential(
            Conv1Block(in_channels, in_channels),
            Conv3Block(in_channels, in_channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )
        self.down_branch = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )
    def forward(self, x):
        up = self.up_branch(x)
        down = self.down_branch(x)
        out = up+down
        return out

class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )
    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)
        return x * y
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class PAU(nn.Module):
    def __init__(self, n_feat, o_feat,  reduction=16, bias=False):
        super(PAU, self).__init__()

        modules_body = \
            [
                nn.Conv2d(n_feat, o_feat, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(n_feat, o_feat, kernel_size=3, stride=1, padding=1)
            ]
        self.body = nn.Sequential(*modules_body)

        self.SA = SALayer()
        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.conv1x1 = nn.Conv2d(n_feat*2, o_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        branch_sa = self.SA(res)
        branch_ca = self.CA(res)
        res = torch.cat([branch_sa, branch_ca], dim=1)
        res = self.conv1x1(res)

        return res+x

class External_Conv_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(External_Conv_Layer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),    #3 h w - 32 h w

            nn.MaxPool2d(kernel_size=2, stride=2),                                         #32 h w - 32 h/2 w/2

            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),   #32 h/2 w/2 - 8 h/2 w/2
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),                                        #8 h/2 w/2 -  8 h/4 w/4

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # 8 h/4 w/4 - 8 h/4 w/4
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),                                      # 8 h/4 w/4 - 8 h/8 w/8

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # 8 h/8 w/8 - 8 h/8 w/8
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),          # 8 h/8 w/8 - 8 h/8 w/4
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),          # 8 h/4 w/4 - 8 h/2 w/2
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1), # 8 h/2 w/2 - 32 h/2 w/2

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),          # 32 h/2 w/2 - 32 h w
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # 32 h w  - 3 h w
        )
    def forward(self, x):

        out = self.encoder(x)
        out = self.decoder(out)
        return out+x



class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        # Encoder
        self.en1 = nn.Sequential(
            Conv3Block(64, 64),
            Conv3Block(64, 64),
            PAU(64, 64)
        )
        self.en1_max = nn.MaxPool2d(2,2)

        self.en2 = nn.Sequential(
            Conv3Block(64, 128),
            Conv3Block(128, 128),
            PAU(128, 128)
        )
        self.u2 = IRM_U(128,64)
        self.d2 = IRM_D(128, 256)
        self.en2_max = nn.MaxPool2d(2, 2)

        self.en3 = nn.Sequential(
            Conv3Block(128, 256),
            Conv3Block(256, 256),
            PAU(256, 256)
        )
        self.u3 = IRM_U(256, 128)
        self.d3 = IRM_D(256, 512)
        self.en3_max = nn.MaxPool2d(2, 2)

        self.en4 = nn.Sequential(
            Conv3Block(256, 512),
            Conv3Block(512, 512),
            PAU(512, 512)
        )
        self.en4_max = nn.MaxPool2d(2, 2)

        #bottom
        self.bottom = nn.Sequential(
            Conv3Block(512, 1024),
            Conv3Block(1024, 1024)
        )

        #decoder
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024,512,3,1,1)
        )
        self.de4 = nn.Sequential(
            Conv3Block(1536, 512),
            Conv3Block(512, 512)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, 3, 1, 1)
        )
        self.de3 = nn.Sequential(
            Conv3Block(768, 256),
            Conv3Block(256, 256)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, 1, 1)
        )
        self.de2 = nn.Sequential(
            Conv3Block(384, 128),
            Conv3Block(128, 128)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, 1, 1)
        )
        self.de1 = nn.Sequential(
            Conv3Block(192, 64),
            Conv3Block(64, 64)
        )


    def forward(self, x):
        skip_connections = []

        # Encoder
        out = self.en1(x)
        skip_connections.append(out)
        out = self.en1_max(out)

        out = self.en2(out)
        skip_connections.append(out)
        u2 = self.u2(out)
        d2 = self.d2(out)
        out = self.en2_max(out)

        out = self.en3(out)
        skip_connections.append(out)
        u3 = self.u3(out)
        d3 = self.d3(out)
        out = self.en3_max(out)

        out = self.en4(out)
        skip_connections.append(out)
        out = self.en4_max(out)

        # bottom
        bottom = self.bottom(out)

        #decoder
        out = self.up4(bottom)
        out = self.de4( torch.cat([out,skip_connections[3],d3],dim=1) )

        out = self.up3(out)
        out = self.de3(torch.cat([out, skip_connections[2], d2], dim=1))

        out = self.up2(out)
        out = self.de2(torch.cat([out, skip_connections[1], u3], dim=1))

        out = self.up1(out)
        out = self.de1(torch.cat([out, skip_connections[0], u2], dim=1))

        return out


class LAUNet(nn.Module):
    def __init__(self):
        super(LAUNet, self).__init__()
        self.first = nn.Sequential(
            Conv3x3(3, 64),
            PAU(64,64)
        )

        self.middle = U_Net()
        self.externl = External_Conv_Layer(3,3)

        self.second = nn.Sequential(
            Conv3x3(64, 3)

        )

    def forward(self, x):
        u_in = self.first(x)
        u_out = self.middle(u_in)
        main_out = self.second(u_out)
        branch_out = self.externl(x)
        out = main_out+branch_out

        return out


if __name__ =="__main__":
    from thop import profile
    x = torch.randn(1,3,128,128)
    #
    # ca = CALayer(32)
    # sa = SALayer()
    # print(ca(x).shape,sa(x).shape)
    #
    # p = PAU(32,32)
    # p(x)
    # print(p(x).shape)

    # x = torch.randn(1, 64, 128, 128)
    # unet = U_Net()
    # unet(x)
    # print(unet(x).shape)

    lau = LAUNet()
    lau(x)

    flops, params = profile(lau, inputs=(x,))

    print('parameters:', params / 1e6)
    print('flops', flops / 1e9)

    print(lau(x).shape)


