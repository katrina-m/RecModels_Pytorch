import dgl
import numpy as np
import torch


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]


class NeighborFinder:

    def __init__(self, kg_data, uniform=False, bidirectional=True):
        """
        Params
        ------
        node_idx_list: List[int], contains the list of node index.
        node_ts_list: List[int], contain the list of timestamp for the nodes in node_idx_list.
        off_set_list: List[int], such that node_idx_list[off_set_list[i]:off_set_list[i + 1]] = adjacent_list[i]. \
                Using this can help us quickly find the adjacent node indexes.
        """
        self.bidirectional = bidirectional
        adj_list = self.init_data(kg_data)
        node_idx_l, node_ts_l, s_type_l, d_type_l, edge_type_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_list = node_idx_l
        self.node_ts_list = node_ts_l
        self.edge_type_list = edge_type_l
        self.src_type_list = s_type_l
        self.dst_type_list = d_type_l

        self.off_set_list = off_set_l

        self.uniform = uniform

    def init_data(self, kg_data):
        src_idx_list = kg_data.h
        dst_idx_list = kg_data.t
        e_type_list = kg_data.r
        h_type_list = kg_data.r
        t_type_list = kg_data.r
        ts_list = kg_data.timestamp.values
        max_idx = max(max(src_idx_list), max(dst_idx_list))

        # The graph is bi-directional
        if self.bidirectional is True:
            adj_list = [[] for _ in range(max_idx + 1)]
            for src, dst, hType, tType, eType, ts in zip(src_idx_list, dst_idx_list, h_type_list, t_type_list, e_type_list, ts_list):
                adj_list[src].append((dst, hType, tType, eType, ts))
                adj_list[dst].append((src, tType, hType, eType, ts))
        else:
            adj_list = [[] for _ in range(max_idx + 1)]
            for src, dst, hType, tType, eType, ts in zip(src_idx_list, dst_idx_list, h_type_list, t_type_list, e_type_list, ts_list):
                adj_list[src].append((dst, hType, tType, eType, ts))
        return adj_list

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
        s_type_list = []
        d_type_list = []
        e_type_list = []
        off_set_list = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[4])
            n_idx_list.extend([x[0] for x in curr])
            s_type_list.extend([x[1] for x in curr])
            d_type_list.extend([x[2] for x in curr])
            e_type_list.extend([x[3] for x in curr])
            n_ts_list.extend([x[4] for x in curr])

            off_set_list.append(len(n_idx_list))
        n_idx_list = np.array(n_idx_list)
        s_type_list = np.array(s_type_list)
        d_type_list = np.array(d_type_list)
        n_ts_list = np.array(n_ts_list)
        e_type_list = np.array(e_type_list)
        off_set_list = np.array(off_set_list)

        assert(len(n_idx_list) == len(n_ts_list))
        assert(off_set_list[-1] == len(n_ts_list))

        return n_idx_list, n_ts_list, s_type_list, d_type_list, e_type_list, off_set_list

    def find_before(self, src_idx, cut_time=None, sort_by_time=True):
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
        src_type_list = self.src_type_list
        dst_type_list = self.dst_type_list
        edge_type_list = self.edge_type_list
        off_set_list = self.off_set_list
        node_ts_list = self.node_ts_list

        neighbors_ts = node_ts_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_idx = node_idx_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_e_type = edge_type_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_src_type = src_type_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]
        neighbors_dst_type = dst_type_list[off_set_list[src_idx]:off_set_list[src_idx + 1]]

        if sort_by_time is False:
            return neighbors_idx, neighbors_src_type, neighbors_dst_type, neighbors_e_type, neighbors_ts

        # If no neighbor find, returns the empty list.
        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_src_type, neighbors_dst_type, neighbors_e_type, neighbors_ts

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
            return neighbors_idx[:right], neighbors_src_type[:right], neighbors_dst_type[:right], neighbors_e_type[:right], neighbors_ts[:right]
        else:
            return neighbors_idx[:left], neighbors_src_type[:left], neighbors_dst_type[:left], neighbors_e_type[:left], neighbors_ts[:left]

    def get_temporal_neighbor(self, src_idx_list, cut_time_list, num_neighbors=20, sort_by_time=True):
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
            out_ngh_sType_batch: int32 matrix (len(src_type_list), num_neighbors)
            out_ngh_dType_batch: int32 matrix (len(dst_type_list), num_neighbors)
        """
        #assert(len(src_idx_list) == len(cut_time_list))

        out_ngh_node_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.float32)
        out_ngh_eType_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.int32)
        out_ngh_sType_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.int32)
        out_ngh_dType_batch = np.zeros((len(src_idx_list), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_list, cut_time_list)):
            ngh_idx, ngh_sType, ngh_dType, ngh_eType, ngh_ts = self.find_before(src_idx, cut_time, sort_by_time)
            ngh_ts[ngh_ts == 0] = cut_time
            if len(ngh_idx) > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                    out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                    out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                    out_ngh_sType_batch[i, :] = ngh_sType[sampled_idx]
                    out_ngh_dType_batch[i, :] = ngh_dType[sampled_idx]
                    out_ngh_eType_batch[i, :] = ngh_eType[sampled_idx]

                    # resort based on time
                    pos = out_ngh_t_batch[i, :].argsort()
                    out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                    out_ngh_sType_batch = out_ngh_sType_batch[i, :][pos]
                    out_ngh_dType_batch = out_ngh_dType_batch[i, :][pos]
                    out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                    out_ngh_eType_batch[i, :] = out_ngh_eType_batch[i, :][pos]
                else:
                    ngh_ts = ngh_ts[:num_neighbors]
                    ngh_idx = ngh_idx[:num_neighbors]
                    ngh_eType = ngh_eType[:num_neighbors]
                    ngh_sType = ngh_sType[:num_neighbors]
                    ngh_dType = ngh_dType[:num_neighbors]

                    assert(len(ngh_idx) <= num_neighbors)
                    assert(len(ngh_ts) <= num_neighbors)
                    assert(len(ngh_eType) <= num_neighbors)
                    assert(len(ngh_sType) <= num_neighbors)
                    assert(len(ngh_dType) <= num_neighbors)

                    out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                    out_ngh_sType_batch[i, num_neighbors - len(ngh_sType):] = ngh_sType
                    out_ngh_dType_batch[i, num_neighbors - len(ngh_dType):] = ngh_dType
                    out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                    out_ngh_eType_batch[i,  num_neighbors - len(ngh_eType):] = ngh_eType

        return out_ngh_node_batch, out_ngh_sType_batch, out_ngh_dType_batch, out_ngh_eType_batch, out_ngh_t_batch

    def find_k_hop_temporal(self, src_idx_l, cut_time_l=None, fan_outs=[15], sort_by_time=True):
        """Sampling the k-hop sub graph before the cut_time
        """

        x, s, d, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, fan_outs[0], sort_by_time=sort_by_time)
        node_records = [x]
        sType_records = [s]
        dType_records = [d]
        eType_records = [y]
        t_records = [z]
        for i in range(1, len(fan_outs)):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[-1] # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_sType_batch, out_ngh_dType_batch, out_ngh_eType_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngn_node_est, ngn_t_est, fan_outs[i])
            out_ngh_node_batch = out_ngh_node_batch.reshape(*orig_shape, fan_outs[i]) # [N, *([num_neighbors] * k)]
            out_ngh_sType_batch = out_ngh_sType_batch.reshape(*orig_shape, fan_outs[i]) # [N, *([num_neighbors] * k)]
            out_ngh_dType_batch = out_ngh_dType_batch.reshape(*orig_shape, fan_outs[i]) # [N, *([num_neighbors] * k)]
            out_ngh_eType_batch = out_ngh_eType_batch.reshape(*orig_shape, fan_outs[i])
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, fan_outs[i])

            node_records.append(out_ngh_node_batch)
            sType_records.append(out_ngh_sType_batch)
            dType_records.append(out_ngh_dType_batch)
            eType_records.append(out_ngh_eType_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, sType_records, dType_records, eType_records, t_records


class Graph(object):

    def __init__(self, kg_df, num_nodes, device="cpu"):
        self.num_relations = None
        self.device = device
        self.num_nodes = num_nodes
        self.g = self.construct_graph(kg_df)

        pass

    def construct_graph(self, kg_df):

        g = dgl.DGLGraph()
        g.add_nodes(self.num_nodes)
        g.add_edges(kg_df['t'].astype(np.int32), kg_df['h'].astype(np.int32))
        #g.edata["timestamp"] = torch.LongTensor(kg_df["timestamp"])#.to(self.device)
        g.edata["type"] = torch.LongTensor(kg_df["r"])#.to(self.device)

        self.num_nodes = g.num_nodes()
        self.num_relations = kg_df.r.nunique()
        return g

    def sample_blocks(self, seeds, fan_outs):

        seeds = torch.LongTensor(np.asarray(seeds))
        blocks = []
        for fan_out in fan_outs:
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fan_out, replace=True)
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return [block.to(self.device) for block in blocks]


def sample_neg_items_for_u(pos_items, start_item_id, end_item_id, n_sample_neg_items, sequential=False):
    """!
    Sample the negative items for a user, if sequential is true, the items are sampled only with respect to one positive item.
    """

    sample_neg_items = []

    if sequential is True:
        for pos_item in pos_items:
            for _ in range(n_sample_neg_items):
                while True:
                    neg_item_id = np.random.randint(low=start_item_id, high=end_item_id, size=1)[0]
                    if neg_item_id != pos_item and neg_item_id not in sample_neg_items:
                        sample_neg_items.append(neg_item_id)
                        break
    else:
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break
            else:
                neg_item_id = np.random.randint(low=start_item_id, high=end_item_id, size=1)[0]
                if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                    sample_neg_items.append(neg_item_id)

    return np.array(sample_neg_items)


def sequence_data_partition(df, with_time=False, discretize_time=False):
    """
    partition the data into train/val/test for sequence modeling.
    :param df:
    :return:
    """
    # Hard-coded, filtered the invalid items and users.
    user_count = df.groupby("userId")[['itemId']].nunique()
    item_count = df.groupby("itemId")[['userId']].nunique()
    valid_user_count = user_count.query("itemId>=5").reset_index()
    valid_item_count = item_count.query("userId>=5").reset_index()
    df = df.merge(valid_user_count[["userId"]], on="userId", how="right")
    df = df.merge(valid_item_count[["itemId"]], on="itemId", how="right")

    def norm_time(time_vectors):
        time_vectors = np.array(time_vectors)
        time_min = time_vectors.min()
        time_diff = np.diff(time_vectors)
        if len(time_diff) == 1:
            time_scale = 1
        else:
            time_scale = time_diff.min()
        time_vectors = int(np.round((time_vectors - time_min)/time_scale) + 1)
        return time_vectors

    if discretize_time:
        user_dict = generate_user_dict(df, sort=True, with_time=with_time, norm_func = norm_time)
    else:
        user_dict = generate_user_dict(df, sort=True, with_time=with_time, norm_func = None)

    user_train = {}
    user_valid = {}
    user_test = {}
    for user, item_infos in user_dict.items():
        nfeedback = len(item_infos)
        if nfeedback < 3:
            user_train[user] = item_infos
        else:
            user_train[user] = item_infos[:-2]
            user_valid[user] = []
            user_valid[user].append(item_infos[-2])
            user_test[user] = []
            user_test[user].append(item_infos[-1])
    print('Preparing done...')
    return [user_train, user_valid, user_test]


def generate_user_dict(df, sort=True, with_time=False, norm_func=None):
    """
    Generate the user dict: {userId: [item list]}, or {userId: ([item list], [timestamp list])}
    :param with_time:
    :param df:
    :param sort:
    :return:
    """
    # def offset_timestamp(sub):
    #     timestamps = sub.sort_values("timestamp")["timestamp"].values
    #     time_scale = min(np.diff(timestamps))
    #     time_scale = time_scale if time_scale > 0 else 1
    if with_time is True:
        if sort is True:
            tmp = df.groupby("userId").apply(lambda sub: sub.sort_values("timestamp")["itemId"].tolist()).reset_index().rename(columns={0:"itemId"})
            tmp_time = df.groupby("userId").apply(lambda sub: sub.sort_values("timestamp")["timestamp"].tolist()).reset_index().rename(columns={0:"timestamp"})
            tmp = tmp.merge(tmp_time, on="userId")
        else:
            tmp = df.groupby("userId").apply(lambda sub: sub["itemId"].tolist()).reset_index().rename(columns={0:"itemId"})
            tmp_time = df.groupby("userId").apply(lambda sub: sub["timestamp"].tolist()).reset_index().rename(columns={0:"timestamp"})
            tmp = tmp.merge(tmp_time, on="userId")

        userInfo = []
        if norm_func is not None:
            tmp["timestamp"] = tmp.timestamp.apply(lambda time_vectors: norm_func(time_vectors))
        for itemIds, timestamps in zip(tmp.itemId.values, tmp.timestamp.values):
            userInfo.append(list(zip(itemIds, timestamps)))

        return dict(zip(tmp.userId.values, userInfo))
    else:
        if sort is True:
            tmp = df.groupby("userId").apply(lambda sub: sub.sort_values("timestamp")["itemId"].tolist()).reset_index().rename(columns={0:"itemId"})
        else:
            tmp = df.groupby("userId").apply(lambda sub: sub["itemId"].tolist()).reset_index().rename(columns={0:"itemId"})
        return dict(zip(tmp.userId.values, tmp.itemId.values))


def computeRePos(time_seq, time_span):

    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix