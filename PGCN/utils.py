import warnings
import os
import torch.nn as nn
from torch.nn import functional as F
import torch
import datetime
import logging
import shutil
import numpy as np
import einops



class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=4, epsilon=0.14, ):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss



def set_logging_config(logdir):
    """
    logging configuration
    :param logdir:
    :return:
    """
    def beijing(sec, what):
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()


    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logging.Formatter.converter = beijing

    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, ("log.txt"))),
                                  logging.StreamHandler(os.sys.stdout)])


def save_checkpoint(state, is_best, dir, subject_name):
    """
    save the checkpoint during training stage
    :param state: content to be saved
    :param is_best: if model's performance is the best at current step
    :param exp_name: experiment name
    :return: None
    """
    torch.save(state, os.path.join(f'{dir}', f'{subject_name}_checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(f'{dir}', f'{subject_name}_checkpoint.pth.tar'),
                        os.path.join(f'{dir}', f'{subject_name}_model_best.pth.tar'))


def normalize_adj(adj):

    D = torch.diag(torch.sum(adj, dim=1))
    D_ = torch.diag(torch.diag(1 / torch.sqrt(D))) # D^(-1/2)
    lap_matrix = torch.matmul(D_, torch.matmul(adj, D_))

    return lap_matrix


# def de_train_test_split_3fold(data, label, index1, index2, config):

#     x_train = np.array([])
#     x_test = np.array([])
#     y_train = np.array([])
#     y_test = np.array([])

#     # if str(config["dataset_name"]) == "SEED5":
#     #     new_data = split_eye_data(data, config["sup_node_num"])

#     new_data = data[:,:310]
#     new_data = einops.rearrange(new_data, "w (h c) -> w h c", c=5)  # (235,62,5)



#     x1 = new_data[:index1]
#     x2 = new_data[index1:index2]
#     x3 = new_data[index2:]

#     y1 = label[:index1]
#     y2 = label[index1:index2]
#     y3 = label[index2:]

#     if config["cfold"] == 1:
#         x_train = np.append(x2, x3, axis=0)
#         x_test = x1
#         y_train = np.append(y2, y3, axis=0)
#         y_test = y1

#     elif config["cfold"] == 2:
#         x_train = np.append(x1, x3, axis=0)
#         x_test = x2
#         y_train = np.append(y1, y3, axis=0)
#         y_test = y2

#     else:
#         x_train = np.append(x1, x2, axis=0)
#         x_test = x3
#         y_train = np.append(y1, y2, axis=0)
#         y_test = y3


#     data_and_label = {"x_train": x_train,
#                       "x_test": x_test,
#                       "y_train": y_train,
#                       "y_test": y_test}

#     return data_and_label


# class Data_split(object):

#     def __init__(self, args):
#         super(Data_split, self).__init__()
#         self.args = args
#         self.subject_index = self.args.subject_index
#         self.sample_list = self.args.dataloader.sampleList
#         self.label_list = self.args.dataloader.labelList
#         self.split_index = self.args.dataloader.split_index[self.subject_index] # for dependent only


#         self.data_and_label = None
#         if self.args.mode.upper() == "DEPENDENT":
#             self.data_and_label = self.de_train_test_split()

#         if self.args.mode.upper() == "INDEPENDENT":
#             self.data_and_label = self.inde_train_test_split()


#     ## for dependent experiment
#     def de_train_test_split(self):

#         x_train = self.sample_list[self.subject_index][:self.split_index]
#         x_test = self.sample_list[self.subject_index][self.split_index:]
#         y_train = self.label_list[self.subject_index][:self.split_index]
#         y_test = self.label_list[self.subject_index][self.split_index:]

#         data_and_label = {"x_train": x_train,
#                           "x_test": x_test,
#                           "y_train": y_train,
#                           "y_test": y_test}

#         return data_and_label

#     ## for independent experiment
#     def inde_train_test_split(self):
#         x_train, x_test, y_train, y_test = np.array([]), np.array([]), \
#                                            np.array([]), np.array([])

#         for j in range(len(self.sample_list)):
#             if j == self.subject_index:
#                 x_test = self.sample_list[j]
#                 y_test = self.label_list[j]
#             else:
#                 if x_train.shape[0] == 0:
#                     x_train = self.sample_list[j]
#                     y_train = self.label_list[j]
#                 else:
#                     x_train = np.append(x_train, self.sample_list[j], axis=0)
#                     y_train = np.append(y_train, self.label_list[j], axis=0)

#         data_and_label = {"x_train": x_train,
#                           "x_test": x_test,
#                           "y_train": y_train,
#                           "y_test": y_test}

#         return data_and_label


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, subject_name, val_acc, model, epoch):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(subject_name, val_acc, model, epoch)
        elif score <= self.best_score + self.delta:
            pass
            # self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # if self.counter >= self.patience:
            #     self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(subject_name, val_acc, model, epoch)
            self.counter = 0

    def save_checkpoint(self, subject_name, val_acc, model, epoch):
        '''Saves model when validation acc increase.'''
        if self.verbose:
            self.trace_func(f'Validation acc increased ({self.val_acc_max:.6f} --> {val_acc:.6f}) in epoch ({epoch}).  Saving model ...')
        model_save_path = self.path + subject_name + ".pt"
        torch.save(model.state_dict(), model_save_path)
        self.val_acc_max = val_acc