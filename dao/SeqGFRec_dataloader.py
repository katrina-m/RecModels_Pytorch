from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
from utility.dao_helper import NeighborFinder, sequence_data_partition, sample_neg_items_for_u, RandEdgeSampler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


class FeatureGen(object):

    def __init__(self, df, kg_df, input_max_length, fan_outs, device="cpu"):

        self.user_id_map = None
        self.item_id_map = None
        self.num_users = None
        self.num_items = None
        self.fan_outs = fan_outs
        self.device = device
        self.input_max_length = input_max_length

        if kg_df is not None:
            item_ids = set(df.itemId.unique())
            node_ids = set(list(kg_df.h.unique())+list(kg_df.t.unique()))
            rest_node_ids = node_ids - item_ids # kg already contains the itemIds, we want to put itemId at the begining
            self.user_offset = max(max(node_ids), max(item_ids))

            user_ids = list(df.userId.unique()) + self.user_offset # assuming user started from 1
            node_ids = list(item_ids)+list(user_ids)+list(rest_node_ids)
            self.node_id_map = dict(zip(node_ids, range(1, len(node_ids)+1)))
        else:
            item_ids = set(df.itemId.unique())
            self.user_offset = max(item_ids)

            user_ids = list(df.userId.unique()) + self.user_offset # assuming user started from 1
            node_ids = list(item_ids)+list(user_ids)#+list(rest_node_ids)
            self.node_id_map = dict(zip(node_ids, range(1, len(node_ids)+1)))

        self.num_items = len(item_ids)
        self.num_nodes = len(node_ids)

        # used for inference
        # formated_data, formated_kg_data = self.format_data(df, kg_df)
        # self.user_dict = generate_user_dict(formated_data, sort=True)

    def prepare_loader(self, data, kg_data, batch_size, valid_batch_size, kg_batch_size):

        data, kg_data = self.format_data(data, kg_data)

        graph = self.create_graph(data, kg_data)
        user_train, user_valid, user_test = sequence_data_partition(data, with_time=True)
        train_data = SASGFRecDataset(train_user_dict=user_train, valid_user_dict=user_valid, test_user_dict=user_test, g=graph, num_items=self.num_items, fan_outs=self.fan_outs, input_max_length=self.input_max_length, mode="train", device=self.device)
        valid_data = SASGFRecDataset(train_user_dict=user_train, valid_user_dict=user_valid, test_user_dict=user_test, g=graph, num_items=self.num_items, fan_outs=self.fan_outs, input_max_length=self.input_max_length, mode="valid", device=self.device)

        pre_train_graph = self.create_pre_train_graph(kg_data)
        pre_train_data = GraphDataset(kg_data.h, kg_data.t, pre_train_graph, fan_outs=self.fan_outs, device=self.device)
        loader_kg = DataLoader(pre_train_data, batch_size=kg_batch_size, collate_fn=pre_train_data.collate_fn)
        loader_train = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn)
        loader_val = DataLoader(valid_data, batch_size=valid_batch_size, collate_fn=valid_data.collate_fn)

        return loader_train, loader_val, loader_kg

    def create_graph(self, cf_data, kg_data):
        """
        Create the graph based on the item knowledge graph and user-item interaction data.
        :param user_item_data:
        :param item_kg_data:
        :return:
        """
        item_kg_data = kg_data.copy()
        user_item_data = cf_data.copy()
        if item_kg_data is not None:
            item_kg_data["timestamp"] = np.zeros(len(item_kg_data))
            #item_kg_data["r"] += 1
            #cf_kg_data = user_item_data[["userId", "itemId", "timestamp"]].rename(columns={"userId":"h", "itemId":"t"})
            #cf_kg_data["r"] = 0
            #cf_kg_data["hType"] = 0
            #cf_kg_data["tType"] = 1
            #kg_data = pd.concat([cf_kg_data, item_kg_data])
            kg_data = item_kg_data
            self.num_relations = kg_data.r.nunique()

        else:
            kg_data = user_item_data[["userId", "itemId", "timestamp"]].rename(columns={"userId":"h", "itemId":"t"})
            kg_data["r"] = 0
            kg_data["hType"] = 0
            kg_data["tType"] = 1
            self.num_realtions = 1

        graph = NeighborFinder(kg_data)
        return graph

    def create_pre_train_graph(self, kg_data):
        item_kg_data = kg_data.copy()
        item_kg_data["timestamp"] = np.zeros(len(item_kg_data))
        item_kg_data["r"] += 1
        item_kg_data["hType"] = item_kg_data.r
        item_kg_data["tType"] = item_kg_data.r

        graph = NeighborFinder(item_kg_data)
        return graph

    def format_data(self, df, kg_df):

        tmp_data = df.copy()
        tmp_data.loc[:, "userId"] = [self.node_id_map[u + self.user_offset] for u in tmp_data.userId]
        tmp_data.loc[:, "itemId"] = [self.node_id_map[u] for u in tmp_data.itemId]
        if kg_df is not None:
            tmp_kg_df = kg_df.copy()
            tmp_kg_df.loc[:, "h"] = [self.node_id_map[u] for u in tmp_kg_df.h]
            tmp_kg_df.loc[:, "t"] = [self.node_id_map[u] for u in tmp_kg_df.t]
            tmp_kg_df["r"] = LabelEncoder().fit_transform(tmp_kg_df.r)
        else:
            tmp_kg_df = None

        return tmp_data, tmp_kg_df

    def generate_feature(self, userId, itemIds):
        pass
        # return torch.LongTensor(np.array([userId])).to(self.device),\
        #        torch.LongTensor(np.array(self.user_dict[userId][-self.input_max_length:])).unsqueeze(1).to(self.device), \
        #        torch.LongTensor(np.array(itemIds)).unsqueeze(0).to(self.device)

    def format_data_single(self, userId, itemIds):
        pass
        # if userId not in self.user_id_map:
        #     return None, None
        # return self.user_id_map[userId], [self.item_id_map[itemId] for itemId in itemIds]


class SASGFRecDataset(Dataset):
    """
     SASRec dataset class in order to use Pytorch DataLoader
     """
    def __init__(self, train_user_dict, g, num_items, valid_user_dict=None, test_user_dict=None, fan_outs=[20], input_max_length=200, mode="train", device="cpu"):
        super().__init__()

        self.mode = mode
        self.device = device
        self.num_items = num_items
        self.fan_outs = fan_outs
        self.input_max_length = input_max_length
        self.train_user_dict = train_user_dict
        self.test_user_dict = test_user_dict
        self.valid_user_dict = valid_user_dict
        self.train_data = list(self.train_user_dict.items())
        self.test_data = list(self.test_user_dict.items())
        self.valid_data = list(self.valid_user_dict.items())
        self.g = g

        if mode == "valid":
            assert valid_user_dict is not None
        elif mode == "test":
            assert valid_user_dict, test_user_dict is not None

    def collate_fn(self, batch):
        if self.mode == "train":
            user, seq, seq_ts, pos, neg, blocks = zip(
                *batch)

            blocks = zip(*blocks)
            block_tensors = []
            seeds = torch.LongTensor(np.array(seq)[:, -self.graph_maxlen:]).to(self.device)
            seeds_ts = torch.LongTensor(np.array(seq)[:, -self.graph_maxlen:]).to(self.device)
            for i, block in enumerate(blocks):
                ngh_batch, ngh_src_type, ngh_dst_type, ngh_edge_type, ngh_ts = zip(*block)
                ngh_batch = torch.LongTensor(ngh_batch).to(self.device)
                ngh_src_type = torch.LongTensor(ngh_src_type).to(self.device)
                ngh_dst_type = torch.LongTensor(ngh_dst_type).to(self.device)
                ngh_edge_type = torch.LongTensor(ngh_edge_type).to(self.device)
                ngh_ts = torch.FloatTensor(ngh_ts).to(self.device)
                block_tensors.append((ngh_batch.view(-1, self.fan_outs[i]), \
                                   seeds.flatten(), ngh_src_type.view(-1, self.fan_outs[i]), \
                                   ngh_dst_type.view(-1, self.fan_outs[i]),  \
                                   ngh_edge_type.view(-1, self.fan_outs[i]), \
                                   ngh_ts.view(-1, self.fan_outs[i]), \
                                   seeds_ts.flatten()))
                seeds = ngh_batch.view(-1)
                seeds_ts = ngh_ts.view(-1)

            return torch.LongTensor(user).to(self.device), torch.LongTensor(seq).to(self.device), torch.LongTensor(seq_ts).to(self.device), \
                   torch.LongTensor(pos).to(self.device), torch.LongTensor(neg).to(self.device), block_tensors

        else:
            user, seq, seq_ts, valid_item_idx, blocks = zip(*batch)
            blocks = zip(*blocks)
            block_tensors = []
            seeds = torch.LongTensor(np.array(seq)[:, -20:]).to(self.device)
            seeds_ts = torch.LongTensor(np.array(seq)[:, -20:]).to(self.device)
            for i, block in enumerate(blocks):
                ngh_batch, ngh_src_type, ngh_dst_type, ngh_edge_type, ngh_ts = zip(*block)
                ngh_batch = torch.LongTensor(ngh_batch).to(self.device)
                ngh_src_type = torch.LongTensor(ngh_src_type).to(self.device)
                ngh_dst_type = torch.LongTensor(ngh_dst_type).to(self.device)
                ngh_edge_type = torch.LongTensor(ngh_edge_type).to(self.device)
                ngh_ts = torch.FloatTensor(ngh_ts).to(self.device)
                block_tensors.append((ngh_batch.view(-1, self.fan_outs[i]), \
                                      seeds.flatten(), ngh_src_type.view(-1, self.fan_outs[i]), \
                                      ngh_dst_type.view(-1, self.fan_outs[i]), \
                                      ngh_edge_type.view(-1, self.fan_outs[i]), \
                                      ngh_ts.view(-1, self.fan_outs[i]), \
                                      seeds_ts.flatten()))
                seeds = ngh_batch.view(-1)
                seeds_ts = ngh_ts.view(-1)

            return torch.LongTensor(user).to(self.device), torch.LongTensor(seq).to(self.device), torch.FloatTensor(seq_ts).to(self.device),\
                   torch.LongTensor(valid_item_idx).to(self.device), block_tensors

    def __getitem__(self, index):

        if self.mode == "train":
            user, item_list = self.train_data[index]
            seq = np.zeros([self.input_max_length], dtype=np.long)
            seq_time = np.zeros([self.input_max_length], dtype=np.long)
            pos = np.zeros([self.input_max_length], dtype=np.long)
            neg = np.zeros([self.input_max_length], dtype=np.long)
            nxt, nxt_time = item_list[-1]
            idx = self.input_max_length - 1

            ts = set([item for item, time in item_list])

            for itemInfo in reversed(item_list[:-1]):
                seq[idx] = itemInfo[0]
                seq_time[idx] = itemInfo[1]
                pos[idx] = nxt
                if nxt != 0:
                    neg[idx] = sample_neg_items_for_u(ts, n_sample_neg_items=1, start_item_id=1, end_item_id=self.num_items, sequential=False)
                nxt = itemInfo[0]
                idx -= 1
                if idx == -1:
                    break

            blocks = self.g.find_k_hop_temporal(seq[-20:], seq_time[-20:], self.fan_outs)
            blocks = list(zip(*blocks))
            return user, seq, seq_time, pos, neg, blocks

        elif self.mode == "valid":
            seq = np.zeros([self.input_max_length], dtype=np.long)
            seq_time = np.zeros([self.input_max_length], dtype=np.long)
            idx = self.input_max_length - 1
            user, target_item = self.valid_data[index]

            for itemInfo in reversed(self.train_user_dict[user]):
                seq[idx] = itemInfo[0]
                seq_time[idx] = itemInfo[1]
                idx -= 1
                if idx == -1: break

            rated = set([item for item, time in self.train_user_dict[user]])
            rated.add(target_item[0][0])
            valid_item_idx = [target_item[0][0]]
            for _ in range(100):
                t = sample_neg_items_for_u(rated, n_sample_neg_items=1, start_item_id=1, end_item_id=self.num_items, sequential=False)[0]
                valid_item_idx.append(t)
            blocks = self.g.find_k_hop_temporal(seq[-20:], seq_time[-20:], self.fan_outs)
            blocks = list(zip(*blocks))
            return user, seq, seq_time, valid_item_idx, blocks

        elif self.mode == "test":
            seq = np.zeros([self.input_max_length], dtype=np.long)
            idx = self.input_max_length - 1
            user, target_item = self.test_data[index]
            valid_user_info = self.valid_user_dict[user]
            seq[idx] = valid_user_info[0]
            idx -= 1
            for i in reversed(self.train_user_dict[user]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            rated = set(self.train_user_dict[user])
            rated.add(target_item[0])
            rated.add(valid_user_info[0])

            test_item_idx = [target_item[0]]
            for _ in range(100):
                t = sample_neg_items_for_u(rated, n_sample_neg_items=1, start_item_id=1, end_item_id=self.num_items, sequential=False)[0]
                test_item_idx.append(t)
            return user, seq, test_item_idx

    def __len__(self):
        if self.mode == "train":
            return len(self.train_data)
        elif self.mode == "valid":
            return len(self.valid_data)
        elif self.mode == 'test':
            return len(self.test_data)


class GraphDataset(Dataset):

    def __init__(self, src_idx_list, dst_idx_list, ngh_finder, fan_outs, device="cpu"):
        super().__init__()
        self.device = device
        self.fan_outs = fan_outs
        self.src_idx_list = src_idx_list
        self.dst_idx_list = dst_idx_list

        self.rand_sampler = RandEdgeSampler(src_idx_list, dst_idx_list)
        self.g = ngh_finder

    def __getitem__(self, index):
        src_l_cut, dst_l_cut = self.src_idx_list[index], self.dst_idx_list[index]
        return src_l_cut, dst_l_cut

    def collate_fn(self, batch):
        src_list, dst_list = zip(*batch)
        src_list_fake, dst_list_fake = self.rand_sampler.sample(len(src_list))
        src_blocks = self.g.find_k_hop_temporal(src_list, cut_time_l=np.zeros_like(src_list), fan_outs=self.fan_outs, sort_by_time=False)
        dst_blocks = self.g.find_k_hop_temporal(dst_list, cut_time_l=np.zeros_like(src_list), fan_outs=self.fan_outs, sort_by_time=False)
        src_fake_blocks = self.g.find_k_hop_temporal(src_list_fake, cut_time_l=np.zeros_like(src_list), fan_outs=self.fan_outs, sort_by_time=False)

        return self.convert_block_to_gpu(src_blocks, src_list), self.convert_block_to_gpu(dst_blocks, dst_list), self.convert_block_to_gpu(src_fake_blocks, src_list_fake)

    def convert_block_to_gpu(self, blocks, seeds):
        blocks = zip(*blocks)

        block_tensors = []
        seeds = torch.LongTensor(seeds).to(self.device)
        seeds_ts = torch.zeros_like(seeds).to(self.device)

        for i, block in enumerate(blocks):
            ngh_batch, ngh_src_type, ngh_dst_type, ngh_edge_type, ngh_ts = block
            ngh_batch = torch.LongTensor(ngh_batch).to(self.device)
            ngh_src_type = torch.LongTensor(ngh_src_type).to(self.device)
            ngh_dst_type = torch.LongTensor(ngh_dst_type).to(self.device)
            ngh_edge_type = torch.LongTensor(ngh_edge_type).to(self.device)
            ngh_ts = torch.FloatTensor(ngh_ts).to(self.device)
            block_tensors.append((ngh_batch.view(-1, self.fan_outs[i]), \
                                  seeds.flatten(), ngh_src_type.view(-1, self.fan_outs[i]), \
                                  ngh_dst_type.view(-1, self.fan_outs[i]), \
                                  ngh_edge_type.view(-1, self.fan_outs[i]), \
                                  ngh_ts.view(-1, self.fan_outs[i]), \
                                  seeds_ts.flatten()))
            seeds = ngh_batch.view(-1)
            seeds_ts = ngh_ts.view(-1)

        return block_tensors

    def __len__(self):
        return len(self.src_idx_list)