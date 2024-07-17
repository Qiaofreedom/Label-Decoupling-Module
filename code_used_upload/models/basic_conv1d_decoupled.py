import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fastai.layers import *
from fastai.core import *

##############################################################################################################################################
# utility functions


def _conv1d(in_planes,out_planes,kernel_size=3, stride=1, dilation=1, act="relu", bn=True, drop_p=0):
    lst=[]
    if(drop_p>0):
        lst.append(nn.Dropout(drop_p))
    lst.append(nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, dilation=dilation, bias=not(bn)))
    if(bn):
        lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    return nn.Sequential(*lst)

def _fc(in_planes,out_planes, act="relu", bn=True):
    lst = [nn.Linear(in_planes, out_planes, bias=not(bn))]
    if(bn):
        lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    return nn.Sequential(*lst)

def cd_adaptiveconcatpool(relevant, irrelevant, module):
    mpr, mpi = module.mp.attrib(relevant,irrelevant)
    apr, api = module.ap.attrib(relevant,irrelevant)
    return torch.cat([mpr, apr], 1), torch.cat([mpi, api], 1)
def attrib_adaptiveconcatpool(self,relevant,irrelevant):
    return cd_adaptiveconcatpool(relevant,irrelevant,self)

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    def attrib(self,relevant,irrelevant):
        return attrib_adaptiveconcatpool(self,relevant,irrelevant)
    
class SqueezeExcite1d(nn.Module):
    '''squeeze excite block as used for example in LSTM FCN'''
    def __init__(self,channels,reduction=16):
        super().__init__()
        channels_reduced = channels//reduction
        self.w1 = torch.nn.Parameter(torch.randn(channels_reduced,channels).unsqueeze(0))
        self.w2 = torch.nn.Parameter(torch.randn(channels, channels_reduced).unsqueeze(0))

    def forward(self, x):
        #input is bs,ch,seq
        z=torch.mean(x,dim=2,keepdim=True)#bs,ch
        intermed = F.relu(torch.matmul(self.w1,z))#(1,ch_red,ch * bs,ch,1) = (bs, ch_red, 1)
        s=F.sigmoid(torch.matmul(self.w2,intermed))#(1,ch,ch_red * bs, ch_red, 1=bs, ch, 1
        return s*x #bs,ch,seq * bs, ch,1 = bs,ch,seq

def weight_init(m):
    '''call weight initialization for model n via n.appy(weight_init)'''
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    if isinstance(m,SqueezeExcite1d):
        stdv1=math.sqrt(2./m.w1.size[0])
        nn.init.normal_(m.w1,0.,stdv1)
        stdv2=math.sqrt(1./m.w2.size[1])
        nn.init.normal_(m.w2,0.,stdv2)

class catAdaptiveConcatPool1d(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.op = AdaptiveConcatPool1d()

    def forward(self, x):
        cnn, gru = x
        pool = self.op(cnn)
        return [pool, gru]


class catMaxPool1d(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.op = nn.MaxPool1d(2)

    def forward(self, x):
        cnn, gru = x
        pool = self.op(cnn)
        return [pool, gru]

class catFlatten(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.op = Flatten()

    def forward(self, x):
        cnn, gru = x
        flatten = self.op(cnn)
        dense = torch.cat([flatten, gru], dim=-1)
        return dense

class rSE(nn.Module):
    def __init__(self, nin, reduce=16):
        super(rSE, self).__init__()
        self.nin = nin
        self.se = nn.Sequential(nn.Linear(self.nin, self.nin // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.nin // reduce, self.nin),
                                nn.Sigmoid())

        self.rse = nn.Sequential(nn.Linear(self.nin, self.nin),
                                nn.Sigmoid())
    def forward(self, x):

        em = x[:, :, 0]   #从输入 x 中提取 em 和 div 特征，分别对应 x 的第3维度的   第0和第1个通道。（第0个通道是经过LDM处理后的特征集合（Module Output），第1个通道是没有经过LDM处理后的特征集合(Raw Output)，即原始的输入分类器之前的特征集合）
        div = x[:, :, 1]

        #diff = torch.abs(em - div)

        prob_em = self.rse(torch.ones_like(em))  # rse里面有Sigmoid函数

        #prob_em = self.se(diff)

        prob_div = 1 - prob_em

        out = em * prob_em + div * prob_div   # em是第0个通道（Module Output），div是第1个通道(Raw Output)

        return out

class DivOutLayer(nn.Module): #这个是LDM结构。输入x是LDM结构中的Module Input部分

    def __init__(self, em_structure, div_structure, bn, drop_rate, em_actns, div_actns, cls_num, metric_out_dim, if_train, **kwargs):
        super().__init__()
        self.em_stru = em_structure # 基本的卷积模型
        self.div_stru = div_structure # 每个embedding space的结构
        self.bn = bn
        self.drop_rate = drop_rate
        self.em_actns = em_actns  # 激活函数
        self.div_actns = div_actns # 激活函数
        self.cls_num = cls_num #类型的数量
        self.metric_out_dim = metric_out_dim  # metric_out_dim 指的是 每个embedding space的结构的 separation layer的 特征数量。
        self.if_train = if_train
        self.baskets = nn.ModuleList()
        self.em_basket = nn.ModuleList()
        self.aggre = rSE(nin=cls_num, reduce=cls_num // 2)

        for ni, no, p, actn in zip(self.em_stru[:-1], self.em_stru[1:], self.drop_rate, self.em_actns):
            bag = []
            if self.bn:
                bag.append(nn.BatchNorm1d(ni).cuda())
            if p != 0:
                bag.append(nn.Dropout(p).cuda())

            bag.append(nn.Linear(ni, no).cuda())

            if actn != None:
                bag.append(actn)  # 激活函数
            bag = nn.Sequential(*bag)
            self.em_basket.append(bag)


        for div_num in range(self.cls_num): # 有多少类就有多少个cls_num。  div_num代表每个类，也就是每个embedding space。
            sub_basket = nn.ModuleList()
            for ni, no, p, actn in zip(self.div_stru[:-1], self.div_stru[1:], self.drop_rate, self.div_actns):
                bag = []

                if self.bn:
                    bag.append(nn.BatchNorm1d(ni).cuda())
                if p != 0:
                    bag.append(nn.Dropout(p).cuda())

                bag.append(nn.Linear(ni, no).cuda())

                if actn != None:
                    bag.append(actn)
                bag = nn.Sequential(*bag)
                sub_basket.append(bag) # 也就是每个embedding space

            self.baskets.append(sub_basket) # 也就是所有的 embedding space




    def forward(self, x):
        cat_out = []
        feats = []
        count = 0

        x_em_deal = x

        for layer in self.em_basket: # 基本的卷积模型
            x_em_deal = layer(x_em_deal)

        x_em_deal = torch.unsqueeze(x_em_deal, dim=-1) # 基本的卷积模型的输出，也就是论文Fig.3.(b)里面的Raw Output（LDM结构里面的Module Input）

        for layers in self.baskets: # 也就是所有的 embedding space
            count += 1
            x_deal = x
            for layer in layers:
                x_deal = layer(x_deal)
                if x_deal.shape[-1] == self.metric_out_dim: # metric_out_dim 指的是 每个embedding space的结构的 separation layer的 特征数量。    对于一个形状为 (2, 3, 4) 的张量，x_deal.shape 将返回 (2, 3, 4)。
                    x_deal_feat = F.normalize(x_deal, p=2, dim=-1)
                    
                    # normalize 是这个模块中的一个函数，用于对输入张量进行归一化。
                    # p=2 指定了范数的类型，这里使用的是 L2 范数（也称为欧几里得范数）。L2 范数是所有元素的平方和的平方根。
                    # dim=-1 指定了进行归一化的维度。-1 表示最后一个维度。
                    
                    feats.append(x_deal_feat)

                if x_deal.shape[-1] == 1 and count == 1:  # 检查 x_deal 的最后一个维度是否为 1。检查 count 是否等于 1，即这是第一个处理的 x_deal 张量。
                    cat_out = x_deal  
                if x_deal.shape[-1] == 1 and count != 1:
                    cat_out = torch.cat((cat_out, x_deal), dim=-1)

        cat_out = torch.unsqueeze(cat_out, dim=-1)  #这里是LDM的结构的输出也就是Module Output
        # torch.unsqueeze 函数将 cat_out 的指定维度 dim 增加一个维度。在这里，dim=-1 表示在 cat_out 的最后一个维度上增加一个新的维度。
        # 如果 cat_out 的形状是 (3, 2)，那么 torch.unsqueeze(cat_out, dim=-1) 的输出形状将是 (3, 2, 1)。

        out = self.aggre(torch.cat((x_em_deal, cat_out), dim=-1)) # 在最后一维上cat，然后做： self.aggre = rSE(nin=cls_num, reduce=cls_num // 2)。见123行
        if self.if_train == True:
            return [out, feats]
        else:
            return out   # 是论文Fig.3.(b)里面的 Final Output.已经融合了Raw Output和Module Output两种Output。

def create_head1d(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    # 深度学习模型的头部（head），主要用于处理1维特征。这个头部将输入特征通过一系列的全连接层和其他层，最终输出指定数量的类别（nc）
    # nf：输入特征的数量。nc：输出类别的数量。lin_ftrs：线性层的特征数量列表。ps：dropout 的概率。bn_final：是否在最后一层添加 BatchNorm。bn：是否在每个全连接层后添加 BatchNorm。act：激活函数的类型，可以是 'relu' 或 'elu'。concat_pooling：是否使用 AdaptiveConcatPool1d 作为池化层。
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    # lin_ftrs 确定了线性层的特征数量。如果没有提供，默认将其设置为 [2*nf, nc]（使用 concat_pooling）或 [nf, nc]
    ps = listify(ps)  # ps = listify(ps) 这行代码的作用是将 ps 转换为列表形式 
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps  # ps 是 dropout 的概率，如果是单个值，将其扩展为与 lin_ftrs 匹配的列表。ps[0] 是 ps 列表中的唯一元素。ps[0]/2 是将这个元素的值减半。[ps[0]/2] * (len(lin_ftrs)-2) 创建一个包含 (len(lin_ftrs)-2) 个 ps[0]/2 的列表。这一步是为了给中间的线性层设置较小的 dropout 概率。
    # + ps 是将原来的 ps 列表（包含一个元素）追加到新列表的末尾。dropout 概率 ps 只需要为每两个相邻层之间的连接设置，因此 ps 的长度应该是 len(lin_ftrs) - 1。也就是说，如果len(lin_ftrs)的元素为4，则ps列表的长度为3.
    actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [None]  # 这里列表的元素个数和ps这个列表的元素个数一致。
    # actns 是激活函数的列表。[nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] 创建一个包含一个激活函数的列表。乘以 (len(lin_ftrs) - 2)，生成一个包含合适数量激活函数的列表。+ [None] 添加一个 None，表示最后一层线性层不需要激活函数。
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)  # bn_drop_lin(ni, no, bn, p, actn) 创建一个线性层，并可能添加批归一化和 dropout 层，然后将这些层添加到 layers
        # 举例说明：假设 lin_ftrs 为 [128, 256, 512, 10]，ps 为 [0.25, 0.25, 0.5]，actns 为 [ReLU(), ReLU(), None]，则 zip 后得到：
        # (128, 256, 0.25, ReLU())
        # (256, 512, 0.25, ReLU())
        # (512, 10, 0.5, None)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01)) # 如果 bn_final 为 True，在最后添加一个 BatchNorm1d 层。lin_ftrs[-1] 是最后一个线性层的输出特征数。
    return nn.Sequential(*layers)  # 使用 nn.Sequential 将 layers 中的所有层组合成一个 序列模型，并返回。

def create_head1d_decoupled(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, div_lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True, if_train=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc]  # was [nf, 512,nc]   # lin_ftrs 确定了 传统的head的 倒数第二层的 特征数量（论文里面的P值）。
    div_lin_ftrs = [2*nf if concat_pooling else nf, nc] if div_lin_ftrs is None else [2*nf if concat_pooling else nf] + div_lin_ftrs + [1] #was [nf, 512, 1]   # div_lin_ftrs是每个embedding space的结构的 separation layer的 特征数量
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    em_actns = [nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)] * (len(lin_ftrs) - 2) + [None] # 基本模型里面的 激活函数
    div_actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(div_lin_ftrs)-3) + [None, None] # LDM结构里面的 激活函数
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten(),
              DivOutLayer(em_structure=lin_ftrs, div_structure=div_lin_ftrs, bn=bn, drop_rate=ps, em_actns=em_actns, div_actns=div_actns, cls_num=nc, metric_out_dim=div_lin_ftrs[-2], if_train=if_train)]
        # DivOutLayer是LDM结构。输入x是LDM结构中的Module Input部分        
        # lin_ftrs 确定了 基本模型的 线性层的 特征数量。# div_lin_ftrs是每个embedding space的结构的 线性层的 特征数量

    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)
##############################################################################################################################################
# basic convolutional architecture

class basic_conv1d(nn.Sequential):
    '''basic conv1d'''
    def __init__(self, filters=[128,128,128,128],kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1, squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,split_first_layer=False,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        layers = []
        if(isinstance(kernel_size,int)):
            kernel_size = [kernel_size]*len(filters)
        for i in range(len(filters)):
            layers_tmp = []
            
            layers_tmp.append(_conv1d(input_channels if i==0 else filters[i-1],filters[i],kernel_size=kernel_size[i],stride=(1 if (split_first_layer is True and i==0) else stride),dilation=dilation,act="none" if ((headless is True and i==len(filters)-1) or (split_first_layer is True and i==0)) else act, bn=False if (headless is True and i==len(filters)-1) else bn,drop_p=(0. if i==0 else drop_p)))
            if((split_first_layer is True and i==0)):
                layers_tmp.append(_conv1d(filters[0],filters[0],kernel_size=1,stride=1,act=act, bn=bn,drop_p=0.))
                #layers_tmp.append(nn.Linear(filters[0],filters[0],bias=not(bn)))
                #layers_tmp.append(_fc(filters[0],filters[0],act=act,bn=bn))
            if(pool>0 and i<len(filters)-1):
                layers_tmp.append(nn.MaxPool1d(pool,stride=pool_stride,padding=(pool-1)//2))
            if(squeeze_excite_reduction>0):
                layers_tmp.append(SqueezeExcite1d(filters[i],squeeze_excite_reduction))
            layers.append(nn.Sequential(*layers_tmp))  # 将临时层包装在 nn.Sequential 中，并添加到主 layers 列表中。

        #head
        #layers.append(nn.AdaptiveAvgPool1d(1))    
        #layers.append(nn.Linear(filters[-1],num_classes))
        #head #inplace=True leads to a runtime error see ReLU+ dropout https://discuss.pytorch.org/t/relu-dropout-inplace/13467/5
        self.headless = headless
        if(headless is True):
            head = nn.Sequential(nn.AdaptiveAvgPool1d(1),Flatten())
        else:
            head=create_head1d(filters[-1], nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)
        
        super().__init__(*layers)
    
    def get_layer_groups(self): # 这个方法返回一个元组，包含两个层组。
        return (self[2],self[-1])  # 返回网络中的第三层，网络中的最后一层

    def get_output_layer(self): # 这个方法返回模型的输出层。
        if self.headless is False: 
            return self[-1][-1]   # self[-1][-1]：表示网络中的最后一层的最后一个元素。这通常是输出层。
        else:
            return None   # 如果 self.headless 为 True，表示模型没有head部分，返回 None
    
    def set_output_layer(self,x): # 这个方法设置模型的输出层。
        if self.headless is False:
            self[-1][-1] = x  # 表示网络中的最后一层的最后一个元素。这通常是输出层。


class basic_conv1d_decoupled(nn.Sequential):
    '''basic conv1d'''

    def __init__(self, filters=[128, 128, 128, 128], kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1,
                 squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,
                 split_first_layer=False, drop_p=0., lin_ftrs_head=None, div_lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True,
                 act_head="relu", concat_pooling=True, if_train=True):
        layers = []
        if (isinstance(kernel_size, int)):
            kernel_size = [kernel_size] * len(filters)
        for i in range(len(filters)):
            layers_tmp = []

            layers_tmp.append(
                _conv1d(input_channels if i == 0 else filters[i - 1], filters[i], kernel_size=kernel_size[i],
                        stride=(1 if (split_first_layer is True and i == 0) else stride), dilation=dilation,
                        act="none" if ((headless is True and i == len(filters) - 1) or (
                                    split_first_layer is True and i == 0)) else act,
                        bn=False if (headless is True and i == len(filters) - 1) else bn,
                        drop_p=(0. if i == 0 else drop_p)))
            if ((split_first_layer is True and i == 0)):
                layers_tmp.append(_conv1d(filters[0], filters[0], kernel_size=1, stride=1, act=act, bn=bn, drop_p=0.))
                # layers_tmp.append(nn.Linear(filters[0],filters[0],bias=not(bn)))
                # layers_tmp.append(_fc(filters[0],filters[0],act=act,bn=bn))
            if (pool > 0 and i < len(filters) - 1):
                layers_tmp.append(nn.MaxPool1d(pool, stride=pool_stride, padding=(pool - 1) // 2))
            if (squeeze_excite_reduction > 0):
                layers_tmp.append(SqueezeExcite1d(filters[i], squeeze_excite_reduction))
            layers.append(nn.Sequential(*layers_tmp))

        # head
        # layers.append(nn.AdaptiveAvgPool1d(1))
        # layers.append(nn.Linear(filters[-1],num_classes))
        # head #inplace=True leads to a runtime error see ReLU+ dropout https://discuss.pytorch.org/t/relu-dropout-inplace/13467/5
        self.headless = headless
        if (headless is True):
            head = nn.Sequential(nn.AdaptiveAvgPool1d(1), Flatten())
        else:
            head = create_head1d_decoupled(filters[-1], nc=num_classes, lin_ftrs=lin_ftrs_head, div_lin_ftrs=div_lin_ftrs_head, ps=ps_head,
                                 bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling, if_train=if_train)
        layers.append(head)

        super().__init__(*layers)

    def get_layer_groups(self):
        return (self[2], self[-1])

    def get_output_layer(self):
        if self.headless is False:
            return self[-1][-1]
        else:
            return None

    def set_output_layer(self, x):
        if self.headless is False:
            self[-1][-1] = x
 
############################################################################################
# convenience functions for basic convolutional architectures

def fcn(filters=[128]*5,num_classes=2,input_channels=8):
    filters_in = filters + [num_classes]
    return basic_conv1d(filters=filters_in,kernel_size=3,stride=1,pool=2,pool_stride=2,input_channels=input_channels,act="relu",bn=True,headless=True)

def fcn_wang(num_classes=2,input_channels=8,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=[128,256,128],kernel_size=[8,5,3],stride=1,pool=0,pool_stride=2, num_classes=num_classes,input_channels=input_channels,act="relu",bn=True,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def fcn_wang_decoupled(num_classes=2,input_channels=8,lin_ftrs_head=None, div_lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True, if_train=True):
    return basic_conv1d_decoupled(filters=[128,256,128],kernel_size=[8,5,3],stride=1,pool=0,pool_stride=2, num_classes=num_classes,input_channels=input_channels,act="relu",bn=True,lin_ftrs_head=lin_ftrs_head, div_lin_ftrs_head=div_lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling, if_train=if_train)

def schirrmeister(num_classes=2,input_channels=8,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=[25,50,100,200],kernel_size=10, stride=3, pool=3, pool_stride=1, num_classes=num_classes, input_channels=input_channels, act="relu", bn=True, headless=False,split_first_layer=True,drop_p=0.5,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def sen(filters=[128]*5,num_classes=2,input_channels=8,squeeze_excite_reduction=16,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=filters,kernel_size=3,stride=2,pool=0,pool_stride=0,input_channels=input_channels,act="relu",bn=True,num_classes=num_classes,squeeze_excite_reduction=squeeze_excite_reduction,drop_p=drop_p,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

def basic1d(filters=[128]*5,kernel_size=3, stride=2, dilation=1, pool=0, pool_stride=1, squeeze_excite_reduction=0, num_classes=2, input_channels=8, act="relu", bn=True, headless=False,drop_p=0.,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
    return basic_conv1d(filters=filters,kernel_size=kernel_size, stride=stride, dilation=dilation, pool=pool, pool_stride=pool_stride, squeeze_excite_reduction=squeeze_excite_reduction, num_classes=num_classes, input_channels=input_channels, act=act, bn=bn, headless=headless,drop_p=drop_p,lin_ftrs_head=lin_ftrs_head, ps_head=ps_head, bn_final_head=bn_final_head, bn_head=bn_head, act_head=act_head, concat_pooling=concat_pooling)

