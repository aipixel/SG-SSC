import torch
import torch.nn as nn
import torch.nn.functional as F


class DDRUnit3D(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True,
                 batch_norm=False, inst_norm=False):
        super(DDRUnit3D, self).__init__()
        s = stride
        k = kernel
        d = dilation
        p = k // 2 * d
        self.batch_norm = batch_norm
        self.conv_in = nn.Conv3d(c_in, c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(c) if batch_norm else None
        self.conv1x1x3 = nn.Conv3d(c, c, (1, 1, k), stride=s, padding=(0, 0, p), bias=True, dilation=(1, 1, d))
        self.conv1x3x1 = nn.Conv3d(c, c, (1, k, 1), stride=s, padding=(0, p, 0), bias=True, dilation=(1, d, 1))
        self.conv3x1x1 = nn.Conv3d(c, c, (k, 1, 1), stride=s, padding=(p, 0, 0), bias=True, dilation=(d, 1, 1))
        self.bn4 = nn.BatchNorm3d(c) if batch_norm else None
        self.conv_out = nn.Conv3d(c, c_out, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm3d(c_out) if batch_norm else None
        self.residual = residual
        self.conv_resid = nn.Conv3d(c_in, c_out, kernel_size=1, bias=False) if residual and c_in != c_out else None
        self.inst_norm = nn.InstanceNorm3d(c_out) if inst_norm else None

    def forward(self, x):
        y0 = self.conv_in(x)
        if self.batch_norm:
            y0 = self.bn1(y0)
        y0 = F.relu(y0, inplace=True)

        y1 = self.conv1x1x3(y0)
        y1 = F.relu(y1, inplace=True)

        y2 = self.conv1x3x1(y1) + y1
        y2 = F.relu(y2, inplace=True)

        y3 = self.conv3x1x1(y2) + y2 + y1
        if self.batch_norm:
            y3 = self.bn4(y3)
        y3 = F.relu(y3, inplace=True)

        y = self.conv_out(y3)
        if self.batch_norm:
            y = self.bn5(y)

        x_squip = x if self.conv_resid is None else self.conv_resid(x)

        y = y + x_squip if self.residual else y

        y = self.inst_norm(y) if self.inst_norm else y

        y = F.relu(y, inplace=True)

        return y


class DDRBlock3D(nn.Module):
    def __init__(self, c_in, c, c_out, units=2, kernel=3, stride=1, dilation=1,
                 pool=True, residual=True, batch_norm=False, inst_norm=False):
        super(DDRBlock3D, self).__init__()
        self.pool = nn.MaxPool3d(2, stride=2) if pool else None
        self.units = nn.ModuleList()
        for i in range(units):
            if i == 0:
                self.units.append(DDRUnit3D(c_in, c, c_out, kernel, stride, dilation, residual, batch_norm, inst_norm))
            else:
                self.units.append(DDRUnit3D(c_out, c, c_out, kernel, stride, dilation, residual, batch_norm, inst_norm))

    def forward(self, x):
        y = self.pool(x) if self.pool is not None else x
        for ddr_unit in self.units:
            y = ddr_unit(y)
        return y


class DDRBlock3DUP(nn.Module):
    def __init__(self, c_in, c, c_out, units=2, kernel=3, stride=1, dilation=1, residual=True,
                 batch_norm=False, inst_norm=False):
        super(DDRBlock3DUP, self).__init__()
        self.transp = nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2)
        self.units = nn.ModuleList()
        for _ in range(units):
            self.units.append(DDRUnit3D(c_out, c, c_out, kernel, stride, dilation, residual, batch_norm, inst_norm))

    def forward(self, x):
        y = self.transp(x)
        for ddr_unit in self.units:
            y = ddr_unit(y)
        return y

class DDR_ASPP3d(nn.Module):
    def __init__(self, c_in, c, c_out, residual=False, batch_norm=False,):
        super(DDR_ASPP3d, self).__init__()
        print('DDR_ASPP3d: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))

        self.aspp0 = nn.Sequential(nn.Conv3d(c_in, c_out, 1, 1, 0),
                                   nn.ReLU())

        self.aspp1 = DDRUnit3D(c_in, c, c_out, dilation=6, residual=residual, batch_norm=batch_norm)

        self.aspp2 = DDRUnit3D(c_in, c, c_out, dilation=12, residual=residual, batch_norm=batch_norm)

        self.aspp3 = DDRUnit3D(c_in, c, c_out, dilation=18, residual=residual, batch_norm=batch_norm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                             nn.Conv3d(c_in, c_out, 1, 1, 0),
                                             nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv3d(c_out*5, 32, 1, 1, 0),
                                   nn.ReLU())
    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x_ = self.global_avg_pool(x)

        # x_ = F.upsample(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        x_ = F.interpolate(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        #print(x0.shape, x1.shape, x2.shape, x3.shape, x_.shape)
        
        x = torch.cat((x0, x1, x2, x3, x_), dim=1)
        x = self.conv1(x)
        return x
             
class CMCA(nn.Module):
    def __init__(self, dim, residual=True, batch_norm=False, inst_norm=False, reduction=2):
        super(CMCA, self).__init__()
        self.dim = dim
        
        self.avg_pool_l = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avg_pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        
        self.max_pool_l = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.max_pool_h = nn.AdaptiveMaxPool3d((1, None, 1))
        self.max_pool_w = nn.AdaptiveMaxPool3d((1, 1, None))
        
        self.conv_l = nn.Sequential(nn.Conv3d(self.dim, self.dim, (3, 1, 1), stride=1, padding=(1, 0, 0)),
                                    nn.ReLU(),
                                    nn.Conv3d(self.dim, self.dim, (3, 1, 1), stride=1, padding=(1, 0, 0)))     
        
        self.conv_h = nn.Sequential(nn.Conv3d(self.dim, self.dim, (1, 3, 1), stride=1, padding=(0, 1, 0)),
                                    nn.ReLU(),
                                    nn.Conv3d(self.dim, self.dim, (1, 3, 1), stride=1, padding=(0, 1, 0)))     

        self.conv_w = nn.Sequential(nn.Conv3d(self.dim, self.dim, (1, 1, 3), stride=1, padding=(0, 0, 1)),
                                    nn.ReLU(),
                                    nn.Conv3d(self.dim, self.dim, (1, 1, 3), stride=1, padding=(0, 0, 1)))   
                                 
        self.mlp1 = nn.Sequential(nn.Conv3d(self.dim * 2, self.dim, kernel_size=1),
                                  nn.ReLU(),
                                  nn.Conv3d(self.dim, self.dim, kernel_size=1),
                                  nn.Sigmoid())                                                     
        
    def forward(self, input1, input2):
        #input1 is the main branch 
        B, C, L, H, W = input1.shape
        corr = input1 - input2
        
        corr_l = self.conv_l(corr)
        avg_l = self.avg_pool_l(corr_l)
        max_l = self.max_pool_l(corr_l)
        corr_l = torch.cat((avg_l, max_l), dim=1)   # B 2C L 1 1
        #print('corr_l.shape:', corr_l.shape)
        
        corr_h = self.conv_h(corr)
        avg_h = self.avg_pool_h(corr_h)
        max_h = self.max_pool_h(corr_h)
        corr_h = torch.cat((avg_h, max_h), dim=1)   # B 2C l H 1
        corr_h = corr_h.permute(0, 1, 3, 2, 4)
        #print('corr_h.shape:', corr_h.shape)
        
        corr_w = self.conv_w(corr)
        avg_w = self.avg_pool_w(corr_w)
        max_w = self.max_pool_w(corr_w)
        corr_w = torch.cat((avg_w, max_w), dim=1)   # B 2C 1 1 W
        corr_w = corr_w.permute(0, 1, 4, 2, 3)
        #print('corr_w.shape:', corr_w.shape)
        
        x = torch.cat([corr_l, corr_h, corr_w], dim=2)
        x = self.mlp1(x)
        corr_l, corr_h, corr_w = torch.split(x, [L, H, W], dim=2)
        corr_h = corr_h.permute(0, 1, 3, 2, 4)
        corr_w = corr_w.permute(0, 1, 3, 4, 2)
        
        out = input1 + corr_l * input2 + corr_h * input2 + corr_w * input2
        
        return out

class SGNet(nn.Module):
    def __init__(self, residual=True, batch_norm=True, inst_norm=False, priors=True):
        super(SGNet, self).__init__()
        self.priors = priors

        # depth branch
        self.d1 = nn.Conv3d(1, 8, 3, stride=1, bias=True, padding=1)
        self.d2 = DDRBlock3D(8, 16, 16, units=1, pool=True, residual=residual, batch_norm=batch_norm,
                             inst_norm=inst_norm)
        self.d_out = DDRBlock3D(16, 32, 32, units=1, pool=True, residual=residual, batch_norm=batch_norm,
                                inst_norm=inst_norm)
        if self.priors:
            # 2D priors branch
            self.p1 = nn.Conv3d(12, 16, 3, stride=1, bias=True, padding=1)
            self.p2 = DDRBlock3D(16, 16, 16, units=1, pool=False, residual=residual, batch_norm=batch_norm,
                                 inst_norm=inst_norm)
            self.p_out = DDRBlock3D(16, 16, 32, units=1, pool=False, residual=residual, batch_norm=batch_norm,
                                    inst_norm=inst_norm)

        # encoder
        self.enc1 = DDRBlock3D(32, 64, 64, units=2, dilation=2, pool=True, residual=residual, batch_norm=batch_norm,
                               inst_norm=inst_norm)
        self.enc2 = DDRBlock3D(64, 128, 128, units=3, dilation=3, pool=True, residual=residual, batch_norm=batch_norm,
                               inst_norm=inst_norm)

        # decoder
        self.dec2 = DDRBlock3DUP(128, 64, 64, units=2, dilation=2, residual=residual, batch_norm=batch_norm,
                                 inst_norm=inst_norm)
        self.dec1 = DDRBlock3DUP(64, 32, 32, units=2, dilation=1, residual=residual, batch_norm=batch_norm,
                                 inst_norm=inst_norm)

        # final
        self.fd1 = DDR_ASPP3d(32, 32, 32, residual=residual, batch_norm=batch_norm)                
        
        self.fs1 = DDRBlock3D(32, 16, 16, units=2, pool=False, residual=residual, batch_norm=batch_norm,
                              inst_norm=inst_norm)
        self.fs2 = DDRBlock3D(32, 16, 16, units=2, pool=False, residual=residual, batch_norm=batch_norm,
                              inst_norm=inst_norm)
                              
        self.sc1 = nn.Conv3d(16, 16, 1, bias=False)   
        self.sc2 = nn.Sequential(DDR_ASPP3d(16, 16, 16, residual=residual, batch_norm=batch_norm),
                                 nn.Conv3d(32, 16, 1),
                                 nn.ReLU())
        self.sc3 = nn.Conv3d(16, 2, 1)    
                                 
        self.fssc1 = DDR_ASPP3d(16, 16, 16, residual=residual, batch_norm=batch_norm)
        self.fssc2 = nn.Sequential(nn.Conv3d(32, 16, 1),
                                   nn.ReLU(),
                                   nn.Conv3d(16, 12, 3, 1, 1))
                                   
        self.cmca = CMCA(16, residual=residual, batch_norm=batch_norm, inst_norm=inst_norm)
        
    def forward(self, depth, priors):
        # get outputs from encoder
        # print("depth")
        d = self.d1(depth)
        # print(d.shape)
        d = self.d2(d)
        # print(d.shape)
        d_out = self.d_out(d)

        # print("d_out", d_out.shape)

        if self.priors:
            # print("priors")
            # priors = self.softmax_priors(priors)
            p = self.p1(priors)
            # print(p.shape)
            p = self.p2(p)
            # print(p.shape)
            p_out = self.p_out(p)

        # print("p_out", p_out.shape)
        # print("encoder")
        e1 = self.enc1(d_out+p_out) if self.priors else self.enc1(d_out)
        # print("e1", e1.shape)
        e2 = self.enc2(e1)
        # print("e2", e2.shape)

        # print("decoder")
        d2 = self.dec2(e2)
        # print("d2", d2.shape)
        d1 = self.dec1(e1 + d2)

        # final
        f = self.fd1(d1 + d_out + p_out) if self.priors else self.fd1(d1 + d_out)
        
        # interaction_module
        fsc = self.fs1(f)
        fssc = self.fs2(f) 
        
        # sc
        fssc_cache = self.sc1(fssc) 
        fsc1 = fsc + fssc_cache
        fsc2 = self.sc2(fsc1)         
        sc = self.sc3(fsc2)         
        
        # ssc
        fssc1 = self.cmca(fssc, fsc2) 
        fssc1 = self.fssc1(fssc1)
        
        ssc = self.fssc2(fssc1)

        #print(sc.shape,ssc1.shape,ssc2.shape)
        return sc, ssc

def get_model(input_type, batch_norm=True, inst_norm=False):
    assert input_type in ['rgb+normals', 'rgb+depth', 'depth'], "Not supported network type: " + input_type

    return SGNet(residual=True, batch_norm=batch_norm, inst_norm=inst_norm, priors=input_type != "depth")
