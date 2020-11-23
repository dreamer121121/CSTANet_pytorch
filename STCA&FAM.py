import torch
import torch.nn as nn

class STCA(nn.Module):
    """
    This is our Spatial-Temporal Channel Attention Module,which is used to
    fuse the spatial and temporal information in a learnable way.
    """
    def __init__(self,in_planes,ratio=16):
        super(STCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.Sfc1   = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.Sfc2   = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.Tfc1 = nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False)
        self.Tfc2 = nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        self.afc = nn.Linear(2,2)
        self.softmax = nn.Softmax()

    def forward(self,x1,x2):
        N,C,T,W,H = x1.size()

        S_out = self.Sfc2(self.Sfc1(self.avg_pool(x1)))
        T_out = self.Tfc2(self.Tfc1(self.avg_pool(x2)))

        Attention = torch.cat((S_out, T_out), dim=2).view(N * C, 2)
        Attention = self.softmax(self.afc(Attention)).view((N, C, 2, 1, 1, 1))
        return Attention

class Fam(nn.Module):
    def __init__(self):
        super(Fam, self).__init__()
    def forward(self,input):
        T, C, W, H = input.size()
        o1 = input[:T-1]
        o2 = input[1:]
        map1 = torch.sum(o1,dim=1)/C
        map2 = torch.sum(o2,dim=1)/C
        res = abs(map2-map1)
        att = torch.sigmoid(res).unsqueeze(dim=1)
        out = torch.cat((input[:T-1] * att,input[T-1].unsqueeze(0)),dim=0)
        return out

class FAM(nn.Module):
    """
    This is our Foreground Attention Module(FAM),which is used to enhance
    the Foreground of a video
    """
    def __init__(self):
        super(FAM, self).__init__()
        self.fam = Fam()
    def forward(self,input):
        input = input.transpose(1, 2).contiguous()
        out = []
        for clip in input:
            out.append(self.fam(clip))
        res = torch.stack(out).transpose(1,2).contiguous()
        return res
