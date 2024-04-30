import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:3", help="gpu device")
    # parser.add_argument('--augment_feature_attention', type=int, default=20,
    #                     help='Feature augmentation matrix for attention.')
    parser.add_argument("--log_dir", type=str, default=os.path.join("..", "logs", ), help="log file dir")
    parser.add_argument('--out_feature', type=int, default=20, help='Output feature for GCN.')
    parser.add_argument('--seed', type=int, default=222, help='Random seed.')  # 42
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=20, help='early stopping param')
    # hyperparameter
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    # parser.add_argument('--alpha', type=float, default=0.005, help='Attention reconciliation hyperparameters')  # 5e-4
    parser.add_argument('--beta', type=float, default=5e-5, help='update laplacian matrix')  # 5e-4
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')

    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate .')
    # parser.add_argument('--leakyrelu', type=float, default=0.1, help='leaky relu.')
    # pri-defined dataset
    parser.add_argument("--dataset", type=str, default="SEED4", help="dataset: SEED, SEED4, SEED5, MPED, bcic ")
    parser.add_argument("--session", type=str, default="2", help="")
    parser.add_argument("--mode", type=str, default="dependent", help="dependent, independent or transfer")
    parser.add_argument("--checkpoint", type=str, default=None, help="store current subject's checkpoint")
    parser.add_argument("--module", type=str, default="", help="Store which modules are used in this run")

    # 定义数据集相关的部分参数
    args = parser.parse_args()
    if args.dataset == 'SEED':
        pass
    elif args.dataset == 'SEED4':
        parser.add_argument("--in_feature", type=int, default=5, help="")
        parser.add_argument("--n_class", type=int, default=4, help="")
        parser.add_argument("--epsilon", type=float, default=0.05, help="")
        parser.add_argument("--datapath", type=str, default="../npy_data/seed4/", help="")
    elif args.dataset == 'SEED5':
        pass
    elif args.dataset == 'MPED':
        pass
    else:
        raise ValueError("Wrong dataset!")

    return parser.parse_args()

