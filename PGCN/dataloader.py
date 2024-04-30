import os
import numpy as np


def load_data_de(path, subject):

    dict_load = np.load(os.path.join(path, (str(subject))), allow_pickle=True)
    data = dict_load[()]['sample']
    label = dict_load[()]['label']
    split_index = dict_load[()]["clip"]

    x_tr = data[:split_index]
    x_ts = data[split_index:]
    y_tr = label[:split_index]
    y_ts = label[split_index:]

    data_and_label = {
        "x_tr": x_tr,
        "x_ts": x_ts,
        "y_tr": y_tr,
        "y_ts": y_ts
    }

    return data_and_label


def load_data_inde(path, subject):
    x_tr = np.array([])
    y_tr = np.array([])
    x_ts = np.array([])
    y_ts = np.array([])
    for i_subject in os.listdir(path):
        dict_load = np.load(os.path.join(path, (str(i_subject))), allow_pickle=True)
        data = dict_load[()]['sample']
        label = dict_load[()]['label']

        if i_subject == subject:
            x_ts = data
            y_ts = label
        else:
            if x_tr.shape[0] == 0:
                x_tr = data
                y_tr = label
            else:
                x_tr = np.append(x_tr, data, axis=0)
                y_tr = np.append(y_tr, label, axis=0)

    data_and_label = {
        "x_tr": x_tr,
        "x_ts": x_ts,
        "y_tr": y_tr,
        "y_ts": y_ts
    }
    return data_and_label

