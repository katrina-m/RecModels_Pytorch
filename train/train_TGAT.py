from train.parse_args import parse_tgat_args
from dao.load_test_data import load_data
from dao.tgat_data_loader_dgl import FeatureGen
from model.TGAT import TGAT
import torch
import os
import logging
import random
import numpy as np
from utility.log_helper import *

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(args):

    #random.seed(args.seed)
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    #args.device = "cpu"
    g_df = load_data("ml-1m").sample(frac=0.05)

    featureGen = FeatureGen(uniform=args.uniform, device=args.device)
    train_dataloader, val_dataloader, nn_val_dataloader = featureGen.prepare_loader(g_df, args.batch_size, args.valid_batch_size)

    model = TGAT(featureGen.num_nodes, featureGen.num_relations, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.fit(train_dataloader, val_dataloader, nn_val_dataloader, optimizer)

if __name__ == '__main__':
    args = parse_tgat_args()
    train(args)