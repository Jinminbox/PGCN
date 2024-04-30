#-------------------------------------
# SEED4数据的预处理代码 并将数据打包到npy文件中
# Date: 2024.4.24
# Author: Ming Jin
# All Rights Reserved
#-------------------------------------


import os
import numpy as np
from scipy.io import loadmat
import einops
import torch
import random
import pickle
from sklearn import svm
# from einops import rearrange, reduce, repeat



########################################################
####### 取随机数
########################################################
def random_1D_seed(num):
    rand_lists = []
    for index in range(num):
        grand_list = [i for i in range(62)]
        random.shuffle(grand_list)
        rand_tensor = torch.tensor(grand_list).view(1, 62)
        rand_lists.append(rand_tensor)

    rand_torch = torch.cat(tuple(rand_lists), 0)
    return rand_torch

def extend_normal(sample):
    for i in range(len(sample)):

        features_min = np.min(sample[i])
        features_max = np.max(sample[i])
        sample[i] = (sample[i] - features_min) / (features_max - features_min)
    return sample

def return_coordinates():
    m1 = [(-2.285379, 10.372299, 4.564709),
          (0.687462, 10.931931, 4.452579),
          (3.874373, 9.896583, 4.368097),
          (-2.82271, 9.895013, 6.833403),
          (4.143959, 9.607678, 7.067061),

          (-6.417786, 6.362997, 4.476012),
          (-5.745505, 7.282387, 6.764246),
          (-4.248579, 7.990933, 8.73188),
          (-2.046628, 8.049909, 10.162745),
          (0.716282, 7.836015, 10.88362),
          (3.193455, 7.889754, 10.312743),
          (5.337832, 7.691511, 8.678795),
          (6.842302, 6.643506, 6.300108),
          (7.197982, 5.671902, 4.245699),

          (-7.326021, 3.749974, 4.734323),
          (-6.882368, 4.211114, 7.939393),
          (-4.837038, 4.672796, 10.955297),
          (-2.677567, 4.478631, 12.365311),
          (0.455027, 4.186858, 13.104445),
          (3.654295, 4.254963, 12.205945),
          (5.863695, 4.275586, 10.714709),
          (7.610693, 3.851083, 7.604854),
          (7.821661, 3.18878, 4.400032),

          (-7.640498, 0.756314, 4.967095),
          (-7.230136, 0.725585, 8.331517),
          (-5.748005, 0.480691, 11.193904),
          (-3.009834, 0.621885, 13.441012),
          (0.341982, 0.449246, 13.839247),
          (3.62126, 0.31676, 13.082255),
          (6.418348, 0.200262, 11.178412),
          (7.743287, 0.254288, 8.143276),
          (8.214926, 0.533799, 4.980188),

          (-7.794727, -1.924366, 4.686678),
          (-7.103159, -2.735806, 7.908936),
          (-5.549734, -3.131109, 10.995642),
          (-3.111164, -3.281632, 12.904391),
          (-0.072857, -3.405421, 13.509398),
          (3.044321, -3.820854, 12.781214),
          (5.712892, -3.643826, 10.907982),
          (7.304755, -3.111501, 7.913397),
          (7.92715, -2.443219, 4.673271),

          (-7.161848, -4.799244, 4.411572),
          (-6.375708, -5.683398, 7.142764),
          (-5.117089, -6.324777, 9.046002),
          (-2.8246, -6.605847, 10.717917),
          (-0.19569, -6.696784, 11.505725),
          (2.396374, -7.077637, 10.585553),
          (4.802065, -6.824497, 8.991351),
          (6.172683, -6.209247, 7.028114),
          (7.187716, -4.954237, 4.477674),

          (-5.894369, -6.974203, 4.318362),
          (-5.037746, -7.566237, 6.585544),
          (-2.544662, -8.415612, 7.820205),
          (-0.339835, -8.716856, 8.249729),
          (2.201964, -8.66148, 7.796194),
          (4.491326, -8.16103, 6.387415),
          (5.766648, -7.498684, 4.546538),

          (-6.387065, -5.755497, 1.886141),
          (-3.542601, -8.904578, 4.214279),
          (-0.080624, -9.660508, 4.670766),
          (3.050584, -9.25965, 4.194428),
          (6.192229, -6.797348, 2.355135),
          ]

    m1 = (m1 - np.min(m1)) / (np.max(m1) - np.min(m1))
    m1 = np.float32(np.array(m1))
    return m1


def eeg_data(label_list, trial_list, raw_path, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for subject in os.listdir(raw_path):
        subject_name = str(subject).strip('.mat')

        data = np.array([])  # 存储每一个subject的数组
        label = np.array([])
        clip = None
        frequency = "de_LDS"
        count = 0

        for i in trial_list:
            dataKey = frequency + str(i+1)
            metaData = np.array((loadmat(os.path.join(raw_path, subject_name), verify_compressed_data_integrity=False)[dataKey])).astype('float')  # 读取到原始的三维元数据
            trMetaData = einops.rearrange(metaData, 'w h c -> h w c')  # (42,62,5)

            count += 1
            x = np.array(trMetaData)
            y = np.array([label_list[i],] * x.shape[0])

            if data.shape[0] == 0:
                data = x
                label = y
            else:
                data = np.append(data, x, axis=0)
                label = np.append(label, y, axis=0)

            if count == 16:
                clip = data.shape[0]

        # 对于data进行min-max归一化
        data = extend_normal(data)

        dict = {'sample': data, 'label': label, 'clip': clip}
        np.save(os.path.join(path, (str(subject).strip('.mat') + ".npy")), dict)



if __name__ == "__main__":
    """Resave all data to .npy"""
    raw_data_path = "/data2/EEG_data/SEED4/eeg_feature_smooth/"
    data_path = "../npy_data/seed4/" # 存储合并的两种数据
    # 对于seed4数据集，由于类别的排布不均匀，所以对于每个session的trial需要进行调整
    session = 1
    label_list = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
    # 选取每种类型的后2个trial用于测试
    trial_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 18, 19, 12, 13, 16, 17, 20, 21, 22, 23]
    print(f"Load session {session}...")
    eeg_data(label_list, trial_list, os.path.join(raw_data_path, str(session)), os.path.join(data_path, str(session)))

    session = 2
    label_list = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
    # 选取每种类型的后2个trial用于测试
    trial_list = [0, 1, 2, 3,  4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 12, 16, 18, 19, 20, 21, 22, 23]
    print(f"Load session {session}...")
    eeg_data(label_list, trial_list, os.path.join(raw_data_path, str(session)), os.path.join(data_path, str(session)))

    session = 3
    label_list = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    # 选取每种类型的后2个trial用于测试
    trial_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 18, 19, 10, 14, 16, 17, 20, 21, 22, 23]
    print(f"Load session {session}...")
    eeg_data(label_list, trial_list, os.path.join(raw_data_path, str(session)), os.path.join(data_path, str(session)))

    print(f"Resave data succeed.")
