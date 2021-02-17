import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import random
from utility.dao_helper import Graph
import pandas as pd


class GraphData(object):

    def __init__(self, src_idx_list, dst_idx_list, ts_list, e_type_list, label_list):
        self.src_idx_list = src_idx_list
        self.dst_idx_list = dst_idx_list
        self.ts_list = ts_list
        self.e_type_list = e_type_list
        self.label_list = label_list
        self.rand_sampler = RandEdgeSampler(src_idx_list, dst_idx_list)


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]


class FeatureGen():

    def __init__(self, uniform=True, device="cpu"):
        self.uniform = uniform
        self.device = device
        self.num_nodes = None
        self.num_relations = None
        pass

    def prepare_loader(self, g_df, batch_size, valid_batch_size):

        train_graph_data, val_graph_data, test_graph_data, new_node_val_graph_data, \
        new_node_test_graph_data, train_graph, full_graph = self.split_data(g_df)

        train_dataset = TGATDataset(train_graph_data, train_graph, mode="train", device=self.device)
        val_dataset = TGATDataset(val_graph_data, full_graph, mode="valid", device=self.device)
        nn_val_dataset = TGATDataset(new_node_val_graph_data, full_graph, mode="valid_new_node", device=self.device)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=valid_batch_size, collate_fn=val_dataset.collate_fn)
        nn_val_dataloader = DataLoader(nn_val_dataset, batch_size=valid_batch_size, collate_fn=nn_val_dataset.collate_fn)

        return train_dataloader, val_dataloader, nn_val_dataloader

    def split_data(self, g_df):

        val_time, test_time = list(np.quantile(g_df.timestamp, [0.70, 0.85]))

        src_idx_list = g_df.srcId.values
        dst_idx_list = g_df.dstId.values
        e_type_list = g_df.eType.values
        label_list = g_df.label.values
        ts_list = g_df.timestamp.values

        total_node_set = set(np.unique(np.hstack([g_df.srcId.values, g_df.dstId.values])))
        self.num_relations = len(set(e_type_list))

        max_idx = max(src_idx_list.max(), dst_idx_list.max())
        self.num_nodes = max_idx+1

        # random selected 10% of nodes from the validation+test sets
        mask_node_set = set(
            random.sample(set(src_idx_list[ts_list > val_time]).union(set(dst_idx_list[ts_list > val_time])),
                          int(0.1 * self.num_nodes)))
        mask_src_flag = g_df.srcId.map(lambda x: x in mask_node_set).values
        mask_dst_flag = g_df.dstId.map(lambda x: x in mask_node_set).values
        none_new_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)  # 两边都不包含new node set

        train_flag = (ts_list <= val_time) * (none_new_node_flag > 0)

        train_src_list = src_idx_list[train_flag]
        train_dst_list = dst_idx_list[train_flag]
        train_ts_list = ts_list[train_flag]
        train_e_type_list = e_type_list[train_flag]
        train_label_list = label_list[train_flag]
        train_graph_data = GraphData(train_src_list, train_dst_list, train_ts_list, train_e_type_list, train_label_list)

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train_src_list).union(train_dst_list)
        assert (len(train_node_set - mask_node_set) == len(train_node_set))
        new_node_set = total_node_set - train_node_set

        # select validation and test dataset
        val_flag = (ts_list <= test_time) * (ts_list > val_time)
        test_flag = ts_list > test_time

        is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_idx_list, dst_idx_list)])
        new_node_val_flag = val_flag * is_new_node_edge
        new_node_test_flag = test_flag * is_new_node_edge

        # validation and test with all edges
        val_src_list = src_idx_list[val_flag]
        val_dst_list = dst_idx_list[val_flag]
        val_ts_list = ts_list[val_flag]
        val_e_type_list = e_type_list[val_flag]
        val_label_list = label_list[val_flag]
        val_graph_data = GraphData(val_src_list, val_dst_list, val_ts_list, val_e_type_list, val_label_list)

        test_src_list = src_idx_list[test_flag]
        test_dst_list = dst_idx_list[test_flag]
        test_ts_list = ts_list[test_flag]
        test_e_type_list = e_type_list[test_flag]
        test_label_list = label_list[test_flag]
        test_graph_data = GraphData(test_src_list, test_dst_list, test_ts_list, test_e_type_list, test_label_list)

        # validation and test with edges that at least has one new node (not in training set)
        new_node_val_src_list = src_idx_list[new_node_val_flag]
        new_node_val_dst_list = dst_idx_list[new_node_val_flag]
        new_node_val_ts_list = ts_list[new_node_val_flag]
        new_node_val_e_type_list = e_type_list[new_node_val_flag]
        new_node_val_label_list = label_list[new_node_val_flag]
        new_node_val_graph_data = GraphData(new_node_val_src_list, new_node_val_dst_list, new_node_val_ts_list, new_node_val_e_type_list, new_node_val_label_list)

        new_node_test_src_list = src_idx_list[new_node_test_flag]
        new_node_test_dst_list = dst_idx_list[new_node_test_flag]
        new_node_test_ts_list = ts_list[new_node_test_flag]
        new_node_test_e_type_list = e_type_list[new_node_test_flag]
        new_node_test_label_list = label_list[new_node_test_flag]
        new_node_test_graph_data = GraphData(new_node_test_src_list, new_node_test_dst_list, new_node_test_ts_list, new_node_test_e_type_list, new_node_test_label_list)

        train_kg = pd.DataFrame({"h":train_graph_data.src_idx_list, "t":train_graph_data.dst_idx_list, "r":train_graph_data.e_type_list, "timestamp":train_graph_data.ts_list})
        train_graph = Graph(train_kg, fan_outs=[15, 15], device=self.device)

        # full graph with all the data for the test and validation purpose
        full_kg = pd.DataFrame({"h":src_idx_list, "t":dst_idx_list, "r":e_type_list, "timestamp":ts_list})
        full_graph = Graph(full_kg, fan_outs=[15, 15], device=self.device)

        return train_graph_data, val_graph_data, test_graph_data, new_node_val_graph_data, \
               new_node_test_graph_data, train_graph, full_graph


class TGATDataset(Dataset):

    def __init__(self, graph_data, graph, mode="train", device="cpu"):
        super().__init__()
        self.mode = mode
        self.device = device
        self.src_idx_list = graph_data.src_idx_list
        self.dst_idx_list = graph_data.dst_idx_list
        self.ts_list = graph_data.ts_list
        self.label_list = graph_data.label_list
        self.rand_sampler = graph_data.rand_sampler
        self.ngh_finder = graph

    def __getitem__(self, index):
        src_l_cut, dst_l_cut = self.src_idx_list[index], self.dst_idx_list[index]
        ts_l_cut = self.ts_list[index]
        label_l_cut = self.label_list[index]
        return src_l_cut, dst_l_cut, ts_l_cut, label_l_cut

    def collate_fn(self, batch):
        src_list, dst_list, ts_list, label_list = zip(*batch)
        src_list_fake, dst_list_fake = self.rand_sampler.sample(len(src_list))
        return np.array(src_list), np.array(dst_list), np.array(ts_list), \
                    np.array(src_list_fake), np.array(dst_list_fake)

    def __len__(self):
        return len(self.src_idx_list)