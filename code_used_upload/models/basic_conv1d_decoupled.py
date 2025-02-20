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
        self.se = nn.Sequential(nn.Linear(self.nin, self.nin // reduce),   # self.nin = P (也就是 P 维度)。 
                                nn.ReLU(inplace=True),
                                nn.Linear(self.nin // reduce, self.nin),
                                nn.Sigmoid())

        self.rse = nn.Sequential(nn.Linear(self.nin, self.nin),  # 计算 P → P 的全连接层，得到一个 shape (batch_size, P) 的 prob_em，表示 em 的权重
                                nn.Sigmoid())
    def forward(self, x):

        em = x[:, :, 0]   #从输入 x 中提取 em 和 div 特征，分别对应 x 的第3维度的   第0和第1个通道。（第0个通道是经过LDM处理后的特征集合（Module Output），第1个通道是没有经过LDM处理后的特征集合(Raw Output)，即原始的输入分类器之前的特征集合）
        div = x[:, :, 1]

        #diff = torch.abs(em - div)

        prob_em = self.rse(torch.ones_like(em))  # rse里面有Sigmoid函数，经过 Sigmoid()，确保 prob_em 取值范围在 (0,1)。 计算 P → P 的全连接层，得到一个 shape (batch_size, P) 的 prob_em，表示 em 的权重

        #prob_em = self.se(diff)

        prob_div = 1 - prob_em  # `div` 的权重 = 1 - `prob_em`。 (batch_size, P)

        out = em * prob_em + div * prob_div   # em是第0个通道（Module Output），div是第1个通道(Raw Output)。 out.shape = (batch_size, P)

        return out

class DivOutLayer(nn.Module): #这个是LDM结构。输入x是LDM结构中的Module Input部分

    def __init__(self, em_structure, div_structure, bn, drop_rate, em_actns, div_actns, cls_num, metric_out_dim, if_train, **kwargs):
        super().__init__()
        self.em_stru = em_structure  # lin_ftrs 确定了 传统的head的 倒数第二层的 特征数量（论文里面的P值）。
        self.div_stru = div_structure   # div_lin_ftrs定义了 head部分 加入了LDM结构的 separation layer层。包括输入维度、隐藏层结构和最终输出维度。 如果 div_lin_ftrs=[512, 256]，那么 separation layer 结构就是：  输入维度 → 512 → 256 → 1（最终输出）
        self.bn = bn
        self.drop_rate = drop_rate
        self.em_actns = em_actns  # 激活函数
        self.div_actns = div_actns # 激活函数
        self.cls_num = cls_num #类型的数量
        self.metric_out_dim = metric_out_dim  # metric_out_dim 指的是 每个embedding space的结构的 separation layer的 输出向量的 最后一个维度 sequence_length（特征维度）（论文里面的j.超参数）如果是没有加LDM的结构的模型里面，这个在自己的工作里面大概是4820。
        self.if_train = if_train
        self.baskets = nn.ModuleList()
        self.em_basket = nn.ModuleList()
        self.aggre = rSE(nin=cls_num, reduce=cls_num // 2)

        for ni, no, p, actn in zip(self.em_stru[:-1], self.em_stru[1:], self.drop_rate, self.em_actns):  # 这段代码的作用是构建 em_basket 这个神经网络模块，它是 LDM 结构中处理 Raw Output的部分。它处理 Raw Output（即 Module Input），输出 Raw Features，用于后续 Final Output 计算。
            bag = []
            if self.bn:
                bag.append(nn.BatchNorm1d(ni).cuda())  # 添加 BatchNorm 归一化层
            if p != 0:
                bag.append(nn.Dropout(p).cuda())   # 添加 Dropout 防止过拟合

            bag.append(nn.Linear(ni, no).cuda()) # 关键的全连接层

            if actn != None:
                bag.append(actn)  # 激活函数
            bag = nn.Sequential(*bag)  # 将所有层组合成一个 Sequential 模块
            self.em_basket.append(bag)  # 加入 `em_basket` 列表


        
        for div_num in range(self.cls_num):  # 它是 LDM 结构中处理 Module Output的部分。 有多少类就有多少个cls_num。  div_num代表每个类，也就是每个separation layer
            sub_basket = nn.ModuleList()
            for ni, no, p, actn in zip(self.div_stru[:-1], self.div_stru[1:], self.drop_rate, self.div_actns):
                bag = []

                if self.bn:
                    bag.append(nn.BatchNorm1d(ni).cuda())
                if p != 0:
                    bag.append(nn.Dropout(p).cuda())

                bag.append(nn.Linear(ni, no).cuda())   # 关键：最后一层 no = 1. 每个 separation layer 只给出 1 个 Cik。这里no = 1，意味着 这个 Linear 层的这个 1 并不代表 Cik 的维度，而是 torch 计算时默认的通道维度。所以 div_lin_ftrs[-1] = 1 是正确的，而不是 K。
                # separation layer 不直接输出 K 维向量. 而是有 K 个 separation layer，每个输出 j 维向量（Cik）
                # Separation Layer 并不是一个大网络输出 K 维 Cik，而是 K 个小网络，每个输出 1 个 Cik。

                if actn != None:
                    bag.append(actn)    # 激活函数 (ReLU/ELU/PReLU)
                bag = nn.Sequential(*bag)  # 将所有层组合成一个 Sequential 模块
                sub_basket.append(bag)  # 每个类别 k 有一个 separation layer

            self.baskets.append(sub_basket) # 所有类别的 separation layer. 说明 K 个类别各有 1 个 separation layer，每个 separation layer 的最终输出是 j 维向量（Cik）




    def forward(self, x):  # x是每个 separation layer 处理相同的输入（Module Input）
        cat_out = []  # 存放 LDM 计算得到的 `Cik`
        feats = []  # 存放 LDM 归一化后的 每一个`Cik`（用于后续可视化）
        count = 0

        # ========== 计算 Raw Output（传统的倒数第二层输出）==========
        
        x_em_deal = x # 这里的 x 是 Module Input，也就是 Fig.3(b) 里的输入

        for layer in self.em_basket: 
            x_em_deal = layer(x_em_deal) # 逐层通过 `em_basket`（传统 MLP ）

        x_em_deal = torch.unsqueeze(x_em_deal, dim=-1)   # 变成 (batch_size, P, 1)，其中 P 是传统倒数第二层的维度（Fig.3(b) 里的 Raw Output）

        # ========== 计算 Module Output（LDM 结构的 Cik）==========
        for layers in self.baskets:  # 遍历所有 embedding space，有K个类别，每个类别 K 都有自己的 `separation layer`。这里的每个layers就是对应`separation layer`。
            count += 1
            x_deal = x  # x是每个 separation layer 处理相同的输入（Module Input）
            for layer in layers: # 遍历当前嵌入空间中的每一层（layer），layers 是一个层的列表
                x_deal = layer(x_deal) 
                if x_deal.shape[-1] == self.metric_out_dim:   # 说明这层是 `separation layer` 的倒数第一层（输出 j 维度的 Cik 向量，Cik 向量是Xi向量被解耦之后的向量）
                    # x_deal 是一个三维张量， 其形状为 (batch_size, lead, sequence_length)。 
                    # metric_out_dim 指的是最后一个维度 sequence_length（特征维度）（论文里面的j.超参数）（它影响到如何设计和连接后续的网络层）。x_deal.shape[-1]表示最后一个维度（特征维度）的大小。 
                    
                    x_deal_feat = F.normalize(x_deal, p=2, dim=-1)
                    
                    # L2 归一化，用于计算度量学习损失（Triplet Loss）
                    # normalize 是这个模块中的一个函数，用于对输入张量进行归一化。归一化可以防止数值爆炸或消失，改善梯度传播，提升模型的训练效果。归一化后的特征通常会使得模型在优化和泛化上表现更好。归一化后的向量在计算余弦相似度等度量时非常有用，因为归一化将所有向量的长度标准化到相同的尺度。
                    # p=2 指定了范数的类型，这里使用的是 L2 范数（也称为欧几里得范数）。L2 范数是所有元素的平方和的平方根。
                    # dim=-1 指定了进行归一化的维度。-1 表示最后一个维度。
                    
                    feats.append(x_deal_feat)  # 存放 LDM 归一化后的 每一个`Cik`（用于后续可视化）

                if x_deal.shape[-1] == 1 and count == 1:  # 检查 x_deal 的最后一个维度是否为 1。检查 count 是否等于 1，也就是看是否是第一个Cik。说明是最后一层，输出的是 `Cik` 值.此时 cat_out 的形状是 (batch_size, 1)
                    cat_out = x_deal  
                if x_deal.shape[-1] == 1 and count != 1: # 说明 x_deal 是 Cik 值，最终拼接成 (batch_size, K)
                    cat_out = torch.cat((cat_out, x_deal), dim=-1)  # 多个类别的 `Cik` 值拼接在一起

        cat_out = torch.unsqueeze(cat_out, dim=-1)  #这里是LDM的结构的输出也就是Module Output。 # 变成 (batch_size, K, 1)，其中 K 是类别数（每个类别对应一个 `Cik`）
        # torch.unsqueeze 函数将 cat_out 的指定维度 dim 增加一个维度。在这里，dim=-1 表示在 cat_out 的最后一个维度上增加一个新的维度。
        # 如果 cat_out 的形状是 (3, 2)，那么 torch.unsqueeze(cat_out, dim=-1) 的输出形状将是 (3, 2, 1)。

        # ========== 融合 Raw Output 和 Module Output ==========
        out = self.aggre(torch.cat((x_em_deal, cat_out), dim=-1))  #拼接后的维度是（batch_size, P, 2）。P和K的值一样。 `rSE`（r-SqueezeExcite）用于自适应加权 `Raw Output` (batch_size, P, 1) 和 `Module Output` (batch_size, K, 1)，得到 Final Output
        
        # 在最后一维上cat，然后做： self.aggre = rSE(nin=cls_num, reduce=cls_num // 2)。见123行。 这里 P == K 是必要条件，否则 rSE 无法进行逐元素加权计算。 最好的方法是设计 lin_ftrs 和 div_lin_ftrs 时，直接让 P == K，避免额外的变换。

        # ========== 选择是否返回度量学习特征 ==========
        if self.if_train == True:
            return [out, feats] # 训练时返回 Final Output 和 归一化 `Cik`。 feats存放 LDM 归一化后的 每一个`Cik`（用于后续可视化）
        else:
            return out   # 推理时只返回 Final Output。 是论文Fig.3.(b)里面的 Final Output.已经融合了Raw Output和Module Output两种Output。

def create_head1d(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    # 深度学习模型的头部（head），主要用于处理1维特征。这个头部将输入特征通过一系列的全连接层和其他层，最终输出指定数量的类别（nc）
    # nf：输入特征的数量。nc：输出类别的数量。lin_ftrs：线性层的特征数量列表。ps：dropout 的概率。bn_final：是否在最后一层添加 BatchNorm。bn：是否在每个全连接层后添加 BatchNorm。act：激活函数的类型，可以是 'relu' 或 'elu'。concat_pooling：是否使用 AdaptiveConcatPool1d 作为池化层。
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    # lin_ftrs 确定了 传统的head的 倒数第二层的 特征数量（论文里面的P值）。。如果没有提供，默认将其设置为 [2*nf, nc]（使用 concat_pooling）或 [nf, nc]
    ps = listify(ps)  # 主要是确保传入的 ps（dropout 概率）总是以列表形式存在，无论用户最初是传入一个单一浮点数还是已经是列表。 
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
    div_lin_ftrs = [2*nf if concat_pooling else nf, nc] if div_lin_ftrs is None else [2*nf if concat_pooling else nf] + div_lin_ftrs + [1] #was [nf, 512, 1]   # div_lin_ftrs定义了 head部分 加入了LDM结构的 separation layer层。包括输入维度、隐藏层结构和最终输出维度。
    # 如果 div_lin_ftrs=[512, 256]，那么 separation layer 结构就是：  输入维度 → 512 → 256 → 1（最终输出）
    ps = listify(ps)
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    em_actns = [nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)] * (len(lin_ftrs) - 2) + [None] # 基本模型里面的 激活函数
    div_actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(div_lin_ftrs)-3) + [None, None] # LDM结构里面的 激活函数
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten(),
              DivOutLayer(em_structure=lin_ftrs, div_structure=div_lin_ftrs, bn=bn, drop_rate=ps, em_actns=em_actns, div_actns=div_actns, cls_num=nc, metric_out_dim=div_lin_ftrs[-2], if_train=if_train)]
        # DivOutLayer是LDM结构。输入x是LDM结构中的Module Input部分        
        # lin_ftrs 确定了 传统的head的 倒数第二层的 特征数量（论文里面的P值）。# div_lin_ftrs定义了 head部分 加入了LDM结构的 separation layer层。包括输入维度、隐藏层结构和最终输出维度。 如果 div_lin_ftrs=[512, 256]，那么 separation layer 结构就是：  输入维度 → 512 → 256 → 1（最终输出）


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
        # 如果 headless=True，则不加分类头部，只输出特征向量。
        # 如果 headless=False，则加上分类头部，输出分类结果。
        # 作用：headless=True → 适用于提取特征，不做分类（例如迁移学习）。headless=False → 适用于完整的分类任务
        if(headless is True):
            head = nn.Sequential(nn.AdaptiveAvgPool1d(1),Flatten())
            # 先进行 全局平均池化 (AdaptiveAvgPool1d(1))，将 seq_len 维度压缩成 1（相当于去掉时间维度）。
            # 然后 展平 (Flatten())，变成 [batch, features] 的形状，但不添加额外的全连接分类层。
            # 这样，模型只输出一个压缩后的特征向量，而不会进行最终的分类预测
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
        # 如果 headless=True，则不加分类头部，只输出特征向量。
        # 如果 headless=False，则加上分类头部，输出分类结果。
        # 作用：headless=True → 适用于提取特征，不做分类（例如迁移学习）。headless=False → 适用于完整的分类任务
        if (headless is True):
            head = nn.Sequential(nn.AdaptiveAvgPool1d(1), Flatten())
            # 先进行 全局平均池化 (AdaptiveAvgPool1d(1))，将 seq_len 维度压缩成 1（相当于去掉时间维度）。
            # 然后 展平 (Flatten())，变成 [batch, features] 的形状，但不添加额外的全连接分类层。
            # 这样，模型只输出一个压缩后的特征向量，而不会进行最终的分类预测
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

