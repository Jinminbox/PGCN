import torch
import numpy as np



def feature_trans(subgraph_num, feature):

    if subgraph_num==7:                  # 脑区分割
        return feature_trans_7(feature)
    elif subgraph_num==4:                # 跨区域分割
        return feature_trans_4(feature)
    elif subgraph_num==2:                # 半脑分割
        return feature_trans_2(feature)
    pass


def location_trans(subgraph_num, location):

    if subgraph_num==7:                  # 脑区分割
        return location_trans_7(location)
    elif subgraph_num==2:                # 半脑分割
        return location_trans_2(location)
    pass


##############################################################
########################### 半脑分割
##############################################################

def feature_trans_2(feature):
    """
    对于原始的特征进行变换，使其符合att_pooling的形状
    :param feature:
    :return:
    """
    reassigned_feature = torch.cat((
        feature[:, 0:1], feature[:, 3:4], feature[:, 5:9],
        feature[:, 14:18], feature[:, 23:27], feature[:, 32:36],
        feature[:,41:45], feature[:, 50:53], feature[:, 57:59],

        feature[:, 2:3], feature[:, 4:5], feature[:, 10:14],
        feature[:, 19:23],feature[:, 28:32], feature[:, 37:41],
        feature[:, 46:50], feature[:, 54:57], feature[:, 60:62],
    ), dim=1)

    return reassigned_feature


def location_trans_2(location):
    """
    对于原始的坐标进行变换，使其符合att_pooling的形状
    :param feature:
    :return:
    """
    reassigned_location = torch.cat((
        location[0:1], location[3:4], location[5:9],
        location[14:18], location[23:27], location[32:36],
        location[41:45], location[50:53], location[57:59],

        location[2:3], location[4:5], location[10:14],
        location[19:23],location[28:32], location[37:41],
        location[46:50], location[54:57], location[60:62],
    ), dim=0)

    return reassigned_location


##############################################################
########################### 跨区域分割
##############################################################

def feature_trans_4(feature):
    reassigned_feature = torch.cat((
        feature[:, 5:6], feature[:, 14:15], feature[:, 23:24],
        feature[:, 13:14], feature[:, 22:23], feature[:, 31:32],

        feature[:, 6:7], feature[:, 15:16], feature[:, 24:25],
        feature[:, 12:13], feature[:, 21:22], feature[:, 30:31],

        feature[:, 32:33], feature[:, 41:42],
        feature[:, 40:41], feature[:, 49:50],

        feature[:, 33:34], feature[:, 42:43], feature[:, 50:51],
        feature[:, 39:40], feature[:, 48:49], feature[:, 56:57],
    ), dim=1)

    return reassigned_feature


##############################################################
########################### 脑区分割
##############################################################

def feature_trans_7(feature):
    """
    对于原始的特征进行变换，使其符合att_pooling的形状
    :param feature:
    :return:
    """
    reassigned_feature = torch.cat((
        feature[:, 0:5],

        feature[:, 5:8], feature[:, 14:17], feature[:, 23:26],

        feature[:, 23:26], feature[:, 32:35], feature[:, 41:44],

        feature[:, 7:12], feature[:, 16:21],feature[:, 25:30],
        feature[:, 34:39], feature[:, 43:48],

        feature[:, 11:14],feature[:, 20:23],feature[:, 29:32],

        feature[:, 29:32],feature[:, 38:41],feature[:, 47:50],

        feature[:, 50:62]), dim=1)

    return reassigned_feature


def location_trans_7(location):
    """
    对于原始的坐标进行变换，使其符合att_pooling的形状
    :param feature:
    :return:
    """
    reassigned_location = torch.cat((
        location[0:5],

        location[5:8], location[14:17], location[23:26],

        location[23:26], location[32:35], location[41:44],

        location[7:12], location[16:21],location[25:30],
        location[34:39], location[43:48],

        location[11:14],location[20:23],location[29:32],

        location[29:32],location[38:41],location[47:50],

        location[50:62]), dim=0)

    return reassigned_location





