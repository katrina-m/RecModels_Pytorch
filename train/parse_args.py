import argparse
import torch


def common_args(parser):
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--corpus_name', nargs='?', default='ml-1m', help='Choose a dataset from {ml_1m}')
    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='../trained_model/model.pth',
                        help='Path of stored model.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=20, help='Number of epoch for early stopping')
    parser.add_argument('--print_every', type=int, default=10, help='Iter interval of printing CF loss.')
    parser.add_argument('--evaluate_every', type=int, default=1, help='Epoch interval of evaluating CF.')
    parser.add_argument('--K', type=int, default=10, help='Calculate metric@K when evaluating.')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str)
    parser.add_argument('--verbose', type=bool, default=True, help='Verbose.')

    return parser


def parse_tgat_args():
    parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
    parser.add_argument('--data_name', type=str, help='data sources to use, try wikipedia or reddit', default='ml-1m')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')
    parser.add_argument('--valid_batch_size', type=int, default=1024, help='valid_batch_size')
    parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
    parser.add_argument('--num_degree', type=int, default=20, help='number of neighbors to sample')
    parser.add_argument('--num_heads', type=int, default=1, help='number of heads used in attention layer')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--num_layers', type=int, default=2, help='number of network layers')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimentions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimentions of the time embedding')
    parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
    parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
    parser.add_argument('--use_time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), type=str)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--print_every', default=50, type=int)
    args = parser.parse_args()

    save_dir = '../trained_model/TGAT/{}/'.format(
        args.data_name)
    args.save_dir = save_dir

    return args


def parse_SASGFRec_args(args_dict=None):
    parser = argparse.ArgumentParser(description="Run SASGFRec.")
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--kg_batch_size', default=512, type=int, help='Batch size')
    parser.add_argument('--valid_batch_size', default=64, type=int, help='Valid batch size')
    parser.add_argument('--maxlen', default=50, type=int, help='Max sequence lengths')
    parser.add_argument('--graph_maxlen', default=20, type=int, help='Max sequence lengths for graph seeds')
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float, help="Dropout rate.")
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--fan_outs', type=list, default=[15, 15], help='Fan outs')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors')
    parser.add_argument('--use_time', type=str, default="pos", choices=['time', 'pos', 'empty'], help='number of neighbors')

    parser = common_args(parser)
    args = parser.parse_args()

    if args_dict is not None:
        for key, value in args_dict.items():
            setattr(args, key, value)

    save_dir = '../trained_model/SASGFRec/{}/hiddendim{}_blocks{}_heads{}_lr{}/'.format(args.corpus_name, args.hidden_units, \
                                                                                      args.num_blocks, args.num_heads, args.lr)
    args.save_dir = save_dir
    return args


def parse_SASRec_args(args_dict=None):
    parser = argparse.ArgumentParser(description="Run SASRec.")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--valid_batch_size', default=300, type=int, help='Valid batch size')
    parser.add_argument('--maxlen', default=50, type=int, help='Max sequence lengths')
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float, help="Dropout rate.")
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser = common_args(parser)
    args = parser.parse_args()

    if args_dict is not None:
        for key, value in args_dict.items():
            setattr(args, key, value)

    save_dir = '../trained_model/SASRec/{}/hiddendim{}_blocks{}_heads{}_lr{}/'.format(args.corpus_name, args.hidden_units, \
                                                                                      args.num_blocks, args.num_heads, args.lr)
    args.save_dir = save_dir
    return args
