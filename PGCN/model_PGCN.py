import torch.nn as nn
from layer_PGCN import LocalLayer, MesoLayer, GlobalLayer
import torch
from utils import normalize_adj


# torch.set_printoptions(profile="full")


class PGCN(nn.Module):
    """
    GCN 62*5 --> 62*(2+4+6+8)
    """
    def __init__(self, args, local_adj, coor):
        super(PGCN, self).__init__()

        self.args = args
        self.nclass = args.n_class
        self.dropout = args.dropout
        self.l_relu = args.lr
        self.adj = local_adj
        self.coordinate = coor

        # self.local_embed = nn.Linear(1, 5)  # 将原始的特征作为meso层的输入

        # Code for PGCN
        # Local GCN
        self.local_gcn_1 = LocalLayer(args.in_feature, 10, True)
        self.local_gcn_2 = LocalLayer(10, 15, True)
        # self.local_gcn_3 = LocalLayer(15, 15, True)
        # self.local_gcn_4 = LocalLayer(15, 15, True)
        # self.local_gcn_5 = LocalLayer(15, 10, True)
        # self.local_gcn_6 = LocalLayer(10, 20, True)


        # Meso
        self.meso_embed = nn.Linear(5, 30) # 将原始的特征拓展后作为meso层的输入
        self.meso_layer_1 = MesoLayer(subgraph_num=7, num_heads=6, coordinate = self.coordinate, trainable_vector=78)
        self.meso_layer_2 = MesoLayer(subgraph_num=2, num_heads=6, coordinate = self.coordinate, trainable_vector=54)
        self.meso_dropout = nn.Dropout(0.2)


        # Global GCN
        # self.embed = nn.Linear(5, 30)
        self.global_layer_1 = GlobalLayer(30, 40)
        # self.global_embed = nn.Linear(40, 30)
        # self.global_layer_2 = GlobalLayer(self.args, 30, 40)
        # self.global_dropout = nn.Dropout(0.5)


        # # Code for ResGCN
        # self.local_gcn_1 = LocalLayer(self.in_feature, 10, True)
        # self.local_gcn_2 = LocalLayer(10, 15, True)
        # self.local_emb = nn.Linear(5, 15)
        # self.local_gcn_3 = LocalLayer(15, 20, True)
        # self.local_gcn_4 = LocalLayer(20, 30, True)
        # self.meso_emb = nn.Linear(15, 30)
        # self.local_gcn_5 = LocalLayer(30, 40, True)
        # self.local_gcn_6 = LocalLayer(40, 70, True)
        # self.global_emb = nn.Linear(30, 70)

        # mlp
        self.mlp0 = nn.Linear(71*70, 2048)
        self.mlp1 = nn.Linear(2048, 1024)
        self.mlp2 = nn.Linear(1024, self.nclass)

        # common
        # self.layer_norm = nn.LayerNorm([30])
        self.bn = nn.BatchNorm1d(1024)
        self.lrelu = nn.LeakyReLU(self.l_relu)
        self.dropout = nn.Dropout(self.dropout)
        # self.att_dropout = nn.Dropout(0.9)


    def forward(self, x):

        # #############################################
        # # step1:Local GCN
        # #############################################
        lap_matrix = normalize_adj(self.adj)
        laplacian = lap_matrix

        local_x1 = self.lrelu(self.local_gcn_1(x, laplacian, True))
        local_x2 = self.lrelu(self.local_gcn_2(local_x1, laplacian, True))
        res_local = torch.cat((x, local_x1, local_x2), 2)

        if "local" not in self.args.module:
            self.args.module += "local "

        ##########################################
        #### step2:mesoscopic scale
        ##########################################
        meso_input = self.meso_embed(x) # 使用原始的feature 作为meso层的输入
        coarsen_x1, coarsen_coor1 = self.meso_layer_1(meso_input)
        coarsen_x1 = self.lrelu(coarsen_x1)

        coarsen_x2, coarsen_coor2 = self.meso_layer_2(meso_input)
        coarsen_x2 = self.lrelu(coarsen_x2)

        # current_batch_size = coarsen_coor1.size()[0]
        # initial_coordinate = self.coordinate.repeat(current_batch_size, 1, 1)

        res_meso = torch.cat((res_local, coarsen_x1, coarsen_x2), 1)
        res_coor = torch.cat((self.coordinate, coarsen_coor1, coarsen_coor2), 0)

        if "meso" not in self.args.module:
            self.args.module += "meso "

        #############################################
        # step3:global scale
        #############################################

        # current_batch_size = res_local.size()[0]
        # initial_coordinate = self.coordinate.repeat(current_batch_size, 1, 1)

        global_x1 = self.lrelu(self.global_layer_1(res_meso, res_coor))

        # global_x2 = self.lrelu(self.global_layer_2(self.global_embed(global_x1), res_coor))

        res_global = torch.cat((res_meso, global_x1), 2)

        if "global" not in self.args.module:
            self.args.module += "global"

        # ############################################
        # step4:emotion recognition
        # ############################################

        x = res_global.view(res_global.size(0), -1)

        x = self.lrelu(self.mlp0(x))
        x = self.dropout(x)
        # x = self.bn(x)
        x = self.lrelu(self.mlp1(x))
        x = self.bn(x)
        # x = self.dropout(x)
        x = self.mlp2(x)

        return x, lap_matrix, ""










