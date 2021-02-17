import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import random


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


class NeighborFinder:

    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_list: List[int], contains the list of node index.
        node_ts_list: List[int], contain the list of timestamp for the nodes in node_idx_list.
        off_set_list: List[int], such that node_idx_list[off_set_list[i]:off_set_list[i + 1]] = adjacent_list[i]. \
                Using this can help us quickly find the adjacent node indexes.
        """

        node_idx_l, node_ts_l, edge_type_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_list = node_idx_l
        self.node_ts_list = node_ts_l
        self.edge_type_list = edge_type_l

        self.off_set_list = off_set_l

        self.uniform = uniform

    def init_off_set(self, adj_list):
        """
        Params ------ Input: adj_list: List[List[(node_idx, edge_idx, node_ts)]], the inner list at each index is the
        adjacent node info of the node with the given index.

        Return:
            n_idx_list: List[int], contain the node index.
            n_ts_list: List[int], contain the timestamp of node index.
            e_idx_list: List[int], contain the edge index.
            off_set_list: List[int], such that node_idx_list[off_set_list[i]:off_set_list[i + 1]] = adjacent_list[i]. \
                Using this can help us quickly find the adjacent node indexes.
        """
        n_idx_list = []
        n_ts_list = []
        e_type_list = []
        off_set_list = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_list.extend([x[0] for x in curr])
            e_type_list.extend([x[1] for x in curr])
            n_ts_list.extend([x[2] for x in curr])

            off_set_list.append(len(n_idx_list))
        n_idx_list = np.array(n_idx_list)
        n_ts_list = np.array(n_ts_list)
        e_type_list = np.array(e_type_list)
        off_set_list = np.array(off_set_list)

        assert(len(n_idx_list) == len(n_ts_list))
        assert(off_set_list[-1] == len(n_ts_list))

        return n_idx_list, n_ts_list, e_type_list, off_set_list

    def find_before(self, src_idx, cut_time):
        """
        Find the neighbors for src_idx with edge time right before the cut_time.
        Params
        ------
        Input:
            src_idx: int
            cut_time: float
        Return:
            neighbors_idx: List[int]
            neighbors_e_idx: List[int]
            neighbors_ts: List[int]
        """
        node_idx_list = self.node_idx_list
        node_ts_list = self.node_ts_list
        edge_type_list = self.edge_type_list
        off_set_list = self.off_set_list

        neighbors_idx = node_idx_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_ts = node_ts_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_e_type = edge_type_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]

        if (neighbors_ts == 0).any():
            # If the edge is stationary, set the edge time to the same as the cut_time.
            return neighbors_idx, neighbors_e_type, np.ones_like(neighbors_ts)*cut_time

        # If no neighbor find, returns the empty list.
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_ts, neighbors_e_type

        # Find the neighbors which has timestamp < cut_time.
        left = 0
        right = len(neighbors_idx) - 1

        while left + 1 < right:
            mid = (left + right) // 2
            curr_t = neighbors_ts[mid]
            if curr_t < cut_time:
                left = mid
            else:
                right = mid

        if neighbors_ts[right] < cut_time:
            return neighbors_idx[:right], neighbors_e_type[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_e_type[:left], neighbors_ts[:left]

    def get_temporal_neighbor(self, src_idx_list, cut_time_list, num_neighbors=20):
        """
        Find the neighbor nodes before cut_time in batch.
        Params
        ------
        Input:
            src_idx_list: List[int]
            cut_time_list: List[float],
            num_neighbors: int
        Return:
            out_ngh_node_batch: int32 matrix (len(src_idx_list), num_neighbors)
            out_ngh_t_batch: int32 matrix (len(src_idx_list), num_neighbors)
            out_ngh_eType_batch: int32 matrix (len(src_idx_list), num_neighbors)
        """
        assert(len(src_idx_list) == len(cut_time_list))

        out_ngh_node_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.float32)
        out_ngh_eType_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_list, cut_time_list)):
            ngh_idx, ngh_eType, ngh_ts = self.find_before(src_idx, cut_time)

            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_eType_batch[i, :] = ngh_eType[sampled_idx]

                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eType_batch[i, :] = out_ngh_eType_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eType = ngh_eType[:num_neighbors]

                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eType) <= num_neighbors)

                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eType_batch[i,  num_neighbors - len(ngh_eType):] = ngh_eType

        return out_ngh_node_batch, out_ngh_eType_batch, out_ngh_t_batch


class FeatureGen():

    def __init__(self, uniform=True, device="cpu"):
        self.uniform = uniform
        self.device = device
        self.num_nodes = None
        self.num_relations = None
        pass

    def prepare_loader(self, g_df, batch_size, valid_batch_size):

        train_graph_data, val_graph_data, test_graph_data, new_node_val_graph_data, \
        new_node_test_graph_data, train_ngh_finder, full_ngh_finder = self.split_data(g_df)

        train_dataset = TGATDataset(train_graph_data, train_ngh_finder, mode="train", device=self.device)
        val_dataset = TGATDataset(val_graph_data, full_ngh_finder, mode="valid", device=self.device)
        nn_val_dataset = TGATDataset(new_node_val_graph_data, full_ngh_finder, mode="valid_new_node", device=self.device)

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

        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eType, ts in zip(train_graph_data.src_idx_list, train_graph_data.dst_idx_list, train_graph_data.e_type_list, train_graph_data.ts_list):
            adj_list[src].append((dst, eType, ts))
            adj_list[dst].append((src, eType, ts))
        train_ngh_finder = NeighborFinder(adj_list, uniform=self.uniform)

        # full graph with all the data for the test and validation purpose
        full_adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eType, ts in zip(src_idx_list, dst_idx_list, e_type_list, ts_list):
            full_adj_list[src].append((dst, eType, ts))
            full_adj_list[dst].append((src, eType, ts))
        full_ngh_finder = NeighborFinder(full_adj_list, uniform=self.uniform)

        return train_graph_data, val_graph_data, test_graph_data, new_node_val_graph_data, \
               new_node_test_graph_data, train_ngh_finder, full_ngh_finder


class TGATDataset(Dataset):

    def __init__(self, graph_data, ngh_finder, mode="train", device="cpu"):
        super().__init__()
        self.mode = mode
        self.device = device
        self.src_idx_list = graph_data.src_idx_list
        self.dst_idx_list = graph_data.dst_idx_list
        self.ts_list = graph_data.ts_list
        self.label_list = graph_data.label_list
        self.rand_sampler = graph_data.rand_sampler
        self.ngh_finder = ngh_finder

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