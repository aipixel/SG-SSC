import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.models as models
import models.transformer_utils.res2net as models
from models.transformer_utils.transformer import TransformerEncoder, TransformerDecoder
from timm.models.layers.weight_init import trunc_normal_
from collections import OrderedDict

def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
    

def convrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    # "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups),
            nn.ReLU())
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups)


stages_suffixes = {0: '_conv',
                   1: '_conv_relu_varout_dimred'}


class RCUBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_blocks, n_stages):
        super(RCUBlock, self).__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
                        conv3x3(in_planes if (i == 0) and (j == 0) else out_planes,
                                out_planes, stride=1,
                                bias=(j == 0)))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = F.relu(x)
                x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
            x += residual
        return F.relu(x)

def get_same_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,
                               padding=get_same_padding(kernel_size, dilation), dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

        self.res_translate = None
        if not inplanes == planes or not stride == 1:
            self.res_translate = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
  
    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.res_translate is not None:
            residual = self.res_translate(residual)
        out += residual

        return out
        
        
class TransformerPredictor(nn.Module):
    def __init__(self, in_channels_2d, num_tokens=300, hidden_dim=256, num_classes=11, layers=3, nheads=8):
        super(TransformerPredictor, self).__init__()

        self.num_tokens = num_tokens     #15*20
        self.hidden_dim = hidden_dim     #128  
        self.num_classes = num_classes   #11        

        self.pos_embed_2d = nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_dim))
        trunc_normal_(self.pos_embed_2d, std=.02)

        self.enc_share = TransformerEncoder(
            embed_dim=self.hidden_dim, #128
            depth=layers, #3
            num_heads=nheads, #8
            drop_rate=0.1, 
            attn_drop_rate=0.1, 
            drop_path=0.1 
        )

        self.dec_2d = TransformerDecoder(
            embed_dim=self.hidden_dim,
            depth=layers,
            num_heads=nheads,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path=0.1
        )

        self.query_embed_2d = nn.Embedding(self.num_classes, self.hidden_dim)

        self.input_proj_c = nn.Conv1d(in_channels_2d, self.hidden_dim, kernel_size=1)
        #c2_xavier_fill(self.input_proj_c)
        
        self.input_proj_d = nn.Conv1d(in_channels_2d, self.hidden_dim, kernel_size=1)
        #c2_xavier_fill(self.input_proj_d)

        #self.mask_embed_2d = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        self.embed_fusion = nn.Sequential(nn.Conv2d(2*self.hidden_dim, self.hidden_dim, 3, 1, 1),
                                          nn.ReLU(),
                                          nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1))
                                          
    def forward(self, x_c, x_d):
        ## 2d pre-processing ##
        B, C, H, W = x_c.size()

        trans_c = x_c.view(B, C, H*W)
        trans_c = self.input_proj_c(trans_c).transpose(1, 2).contiguous() # b hw 128
        
        trans_d = x_d.view(B, C, H*W)
        trans_d = self.input_proj_d(trans_d).transpose(1, 2).contiguous() # b hw 128

        trans_input = torch.cat([trans_c, trans_d], dim=1)
        pos_input = torch.cat([self.pos_embed_2d.expand(B, -1, -1),self.pos_embed_2d.expand(B, -1, -1)], dim=1)
        
        ## TransformerEncoder ##        
        trans_feat = self.enc_share(trans_input, pos_input)
        mask_embed = self.dec_2d(trans_feat, self.query_embed_2d.weight.unsqueeze(dim=0).expand(B, -1, -1))
        
        trans_feat_c = trans_feat[:, :H*W, :].transpose(1, 2).contiguous().view(B, self.hidden_dim, H, W)
        trans_feat_d = trans_feat[:, H*W:, :].transpose(1, 2).contiguous().view(B, self.hidden_dim, H, W)
        
        fusion_embed = torch.cat([trans_feat_c, trans_feat_d], dim=1)
        fusion_embed = self.embed_fusion(fusion_embed)
        #mask_embed = self.mask_embed_2d(mask_embed)
        #print('mask_embed.shape:',mask_embed.shape)
        
        return fusion_embed, mask_embed


class BiModalFusion(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(BiModalFusion, self).__init__()
        self.out_planes = out_planes
        self.mode1_branch = nn.Sequential(OrderedDict([
                                              ('b1_dim_red', conv3x3(out_planes*2, out_planes+1, stride=1, bias=True)),
                                              ('b1_rcu', RCUBlock(out_planes+1, out_planes+1, 1, 2)),
                                              ('b1_conv', conv3x3(out_planes+1, out_planes+1, stride=1, bias=True))
                                            ]))
        self.mode2_branch = nn.Sequential(OrderedDict([
                                              ('b2_dim_red', conv3x3(out_planes*2, out_planes+1, stride=1, bias=True)),
                                              ('b2_rcu', RCUBlock(out_planes+1, out_planes+1, 1, 2)),
                                              ('b2_conv', conv3x3(out_planes+1, out_planes+1, stride=1, bias=True))
                                            ]))
        self.conv_c = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_d = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x1, x2, x3):
        x2 = self.conv_c(x2)
        x2 = torch.cat([x1, x2], dim=1)
        
        x3 = self.conv_d(x3)
        x3 = torch.cat([x1, x3], dim=1)
    
        mc = self.mode1_branch(x2)
        md = self.mode2_branch(x3)
        
        feat_c = mc[:,:self.out_planes,:,:]
        conf_c = mc[:,self.out_planes:,:,:]
        #print('conf_c.shape:',conf_c.shape)
        feat_d = md[:,:self.out_planes,:,:]
        conf_d = md[:,self.out_planes:,:,:]
        
        conf_c, conf_d = torch.chunk(self.softmax(torch.cat([conf_c, conf_d], 1)), 2, dim=1)
        out = conf_c * feat_c + conf_d * feat_d
        out = F.relu(x1+out)
        
        return out
        

class BiModalRDFNetLW(nn.Module):
    def __init__(self, num_classes=11):
        super(BiModalRDFNetLW, self).__init__()

        # get a pretrained encoder
        self.encoder1 = models.res2net50(pretrained=True)
        del self.encoder1.fc
        del self.encoder1.avgpool

        self.encoder2 = models.res2net50(pretrained=True)
        del self.encoder2.fc
        del self.encoder2.avgpool

        self.predictor = TransformerPredictor(in_channels_2d=2048, hidden_dim=256, num_classes=11, layers=3)
        
        self.mmf3 = BiModalFusion(1024, 256)
        self.conv_g1_pool = self._make_crp(256, 256, 1)
        
        self.mmf2 = BiModalFusion(512, 256)
        self.conv_g2_pool = self._make_crp(256, 256, 1)
        
        self.mmf1 = BiModalFusion(256, 256)
        self.conv_g3_pool = self._make_crp(256, 256, 1)

        self.conv_g4_pool = self._make_crp(256, 256, 1)
        #self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        self.clf_conv = nn.Sequential(nn.Conv2d(11, 11, 3, 1, 1),
                                      nn.ReLU(),
                                      nn.Conv2d(11, num_classes, kernel_size=3, stride=1, padding=1, bias=True))

    def _make_crp(self, in_planes, out_planes, stages):
            layers = [ResBlock(in_planes, out_planes) for _ in range(stages)]
            return nn.Sequential(*layers)
    
    def forward(self, x1, x2):
        # get outputs from encoder1
        x_1 = self.encoder1.conv1(x1)
        x_1 = self.encoder1.bn1(x_1)
        x_1 = self.encoder1.relu(x_1)
        x_1 = self.encoder1.maxpool(x_1)
        l1_1 = self.encoder1.layer1(x_1)
        l2_1 = self.encoder1.layer2(l1_1)
        l3_1 = self.encoder1.layer3(l2_1)
        l4_1 = self.encoder1.layer4(l3_1)
        #print('x_1.shape:',x_1.shape)
        #print('l1_1.shape:',l1_1.shape)
        #print('l2_1.shape:',l2_1.shape)
        #print('l3_1.shape:',l3_1.shape)
        #print('l4_1.shape:',l4_1.shape)

        # get outputs from encoder2
        x_2 = self.encoder2.conv1(x2)
        x_2 = self.encoder2.bn1(x_2)
        x_2 = self.encoder2.relu(x_2)
        x_2 = self.encoder2.maxpool(x_2)
        l1_2 = self.encoder2.layer1(x_2)
        l2_2 = self.encoder2.layer2(l1_2)
        l3_2 = self.encoder2.layer3(l2_2)
        l4_2 = self.encoder2.layer4(l3_2)

        seg_list = []

        fusion_p4, mask_embed = self.predictor(l4_1, l4_2)
        seg_p4 = torch.einsum('bqc,bchw->bqhw', mask_embed, fusion_p4)
        #seg_list.append(seg_p4)
        
        fusion_p3 = F.interpolate(fusion_p4, size=l3_1.size()[2:], mode='bilinear', align_corners=False)
        fusion_p3 = self.mmf3(fusion_p3, l3_1, l3_2)
        fusion_p3 = self.conv_g1_pool(fusion_p3)
        seg_p3 = torch.einsum('bqc,bchw->bqhw', mask_embed, fusion_p3)
        seg_list.append(seg_p3)

        fusion_p2 = F.interpolate(fusion_p3, size=l2_1.size()[2:], mode='bilinear', align_corners=False)
        fusion_p2 = self.mmf2(fusion_p2, l2_1, l2_2)
        fusion_p2 = self.conv_g2_pool(fusion_p2)
        seg_p2 = torch.einsum('bqc,bchw->bqhw', mask_embed, fusion_p2)
        seg_list.append(seg_p2)

        fusion_p1 = F.interpolate(fusion_p2, size=l1_1.size()[2:], mode='bilinear', align_corners=False)
        fusion_p1 = self.mmf1(fusion_p1, l1_1, l1_2)
        fusion_p1 = self.conv_g3_pool(fusion_p1)
        seg_p1 = torch.einsum('bqc,bchw->bqhw', mask_embed, fusion_p1)
        seg_list.append(seg_p1)
        #print('seg_p1.shape:',seg_p1.shape)
        fusion_p0 = F.interpolate(seg_p1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        #fusion_p0 = self.conv_g4_pool(fusion_p0)
        #fusion_p0 = F.pixel_shuffle(fusion_p1,4)
        #out = torch.einsum('bqc,bchw->bqhw', mask_embed, fusion_p0)
        out = self.clf_conv(fusion_p0)
        #print('out.shape:',out.shape)
        seg_list.append(out)

        return seg_list
