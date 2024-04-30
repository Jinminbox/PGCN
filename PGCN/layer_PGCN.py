import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import graphpool

from einops import rearrange


class LocalLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LocalLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.lrelu = nn.LeakyReLU(0.1)
        self.bias = bias
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input, lap, is_weight=True):
        if is_weight:
            weighted_feature = torch.einsum('b i j, j d -> b i d', input, self.weight)
            output = torch.einsum('i j, b j d -> b i d', lap, weighted_feature)+self.bias
        else:
            output = torch.einsum('i j, b j d -> b i d', lap, input)
        return output # (batch_size, 62, out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self.in_features)} -> {str(self.out_features) }"



class GlobalLayer(nn.Module):
    """
    Simple GAT layer
    """
    def __init__(self, in_features, out_feature):
        super(GlobalLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_feature
        self.num_heads = 6
        self.lrelu = nn.LeakyReLU(0.1)

        self.embed = nn.Linear(3, 30) # location embeding
        self.get_qk = nn.Linear(self.in_features, self.in_features * 2)

        self.equ_weights = Parameter(torch.FloatTensor(self.num_heads)) # 对邻接矩阵进行head维的相加
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.bias = Parameter(torch.FloatTensor(self.out_features))
        self.reset_parameters()

    def forward(self, h, res_coor):
        h_with_embed = h + self.lrelu(self.embed(res_coor)) # position encoding

        attention_value = self.cal_att_matrix(h, h_with_embed)
        output = torch.matmul(attention_value, self.weight) + self.bias

        return output

    def cal_att_matrix(self, feature, feature_with_embed):  # feature: [n=71, hd=30]
        out_feature = []
        batch_size, N = feature.size(0), feature.size(1)

        qk = rearrange(self.get_qk(feature_with_embed), "b n (h d qk) -> (qk) b h n d", h=self.num_heads, qk=2)
        queries, keys= qk[0], qk[1]
        values = feature

        dim_scale = (queries.size(-1)) ** -0.5 # temperature
        dots = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) * dim_scale  # batch, num_heads, query_len, key_len

        attn = torch.einsum("b g i j -> b i j", dots) # 降维
        adj_matrix = self.dropout_80_percent(attn)
        # attn = self.attend(adj_matrix)
        attn = F.softmax(adj_matrix/0.3, dim=2)

        out_feature = torch.einsum('b i j, b j d -> b i d', attn, values)

        return out_feature


    def dropout_80_percent(self, attn):
        att_subview_, _ = attn.sort(2, descending=True)

        att_threshold = att_subview_[:, :, att_subview_.size(2) // 6]
        att_threshold = rearrange(att_threshold, 'b i -> b i 1')
        att_threshold = att_threshold.repeat(1, 1, attn.size()[2])
        attn[attn<att_threshold] = -1e-7
        return attn


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.equ_weights.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self.in_features)} -> {str(self.out_features)}"



class MesoLayer(nn.Module):
    def __init__(self, subgraph_num, num_heads, coordinate, trainable_vector):
        super(MesoLayer, self).__init__()
        self.subgraph_num = subgraph_num
        self.coordinate = coordinate

        self.lrelu = nn.LeakyReLU(0.1)
        self.graph_list = self.sort_subgraph(subgraph_num)
        self.emb_size = 30
        # self.num_heads = num_heads

        self.softmax = nn.Softmax(dim=0)
        self.att_softmax = nn.Softmax(dim=1)

        # 用于meso区域的节点自适应权重
        self.trainable_vec = Parameter(torch.FloatTensor(trainable_vector))
        # self.trainable_coor_vec = Parameter(torch.FloatTensor(trainable_vector))
        self.weight = Parameter(torch.FloatTensor(self.emb_size, 10))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.trainable_vec.size(0))
        self.trainable_vec.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        # self.trainable_coor_vec.data.uniform_(-stdv, stdv)

    def forward(self, x):
        coarsen_x, coarsen_coor = self.att_coarsen(x)
        return coarsen_x, coarsen_coor

    """使用att进行聚合"""
    def att_coarsen(self, features):
        # feature.shape: (batch_size, N, out_features)
        features = graphpool.feature_trans(self.subgraph_num, features)
        coordinates = graphpool.location_trans(self.subgraph_num, self.coordinate)
        coarsen_feature, coarsen_coordinate = [], []

        idx_head = 0
        for index_length in self.graph_list:
            idx_tail = idx_head + index_length
            sub_feature = features[:, idx_head:idx_tail]
            sub_coordinate = coordinates[idx_head:idx_tail]

            # 计算注意力权重
            feature_with_weight = torch.einsum('b j g, g h -> b j h', sub_feature, self.weight)
            feature_T = rearrange(feature_with_weight, 'b j h -> b h j')
            att_weight_matrix = torch.einsum('b j h, b h i -> b j i', feature_with_weight, feature_T)

            att_weight_vector = torch.sum(att_weight_matrix, dim=2)

            att_vec = self.att_softmax(att_weight_vector)

            sub_feature_ = torch.einsum('b j, b j g -> b g', att_vec, sub_feature)  # talking heads, pre-softmax
            sub_coordinate_ = torch.einsum('b j, j g -> b g', att_vec,
                                           sub_coordinate)  # talking heads, pre-softmax
            sub_coordinate_ = torch.mean(sub_coordinate_, dim=0)

            coarsen_feature.append(rearrange(sub_feature_, "b g -> b 1 g"))
            coarsen_coordinate.append(rearrange(sub_coordinate_, "g -> 1 g"))
            idx_head = idx_tail

        coarsen_features = torch.cat(tuple(coarsen_feature), 1)
        coarsen_coordinates = torch.cat(tuple(coarsen_coordinate), 0)
        return coarsen_features, coarsen_coordinates

    def sort_subgraph(self, subgraph_num):
        """
        根据子图数量确定子图的划分细节
        :param subgraph_num:
        :return:
        """
        subgraph_7 = [5, 9, 9, 25, 9, 9, 12]
        subgraph_4 = [6, 6, 4, 6]
        subgraph_2 = [27, 27]

        graph_list = None
        if subgraph_num == 7:
            graph_list = subgraph_7
        elif subgraph_num == 4:
            graph_list = subgraph_4
        elif subgraph_num == 2:
            graph_list = subgraph_2

        return graph_list




