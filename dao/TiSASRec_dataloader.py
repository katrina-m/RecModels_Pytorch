from torch.utils.data import Dataset
from utility.dao_helper import *
from torch.utils.data.dataloader import DataLoader
import torch


class FeatureGen(object):

    def __init__(self, df, input_max_length, device="cpu"):

        self.user_id_map = None
        self.item_id_map = None
        self.num_users = None
        self.num_items = None
        self.device = device
        self.input_max_length = input_max_length

        user_ids = list(df.userId.unique())
        self.user_id_map = dict(zip(user_ids, range(0, len(user_ids))))
        self.num_users = df.userId.nunique()

        item_ids = list(df.itemId.unique())
        self.item_id_map = dict(zip(item_ids, range(1, len(item_ids)+1)))
        self.num_items = df.itemId.nunique()

        #data = self.format_data(df)
        #elf.user_dict = generate_user_dict(df, sort=True)

    def prepare_loader(self, data, batch_size, valid_batch_size):

        data = self.format_data(data)
        user_train, user_valid, user_test = sequence_data_partition(data)

        train_data = SASRecDataset(train_user_dict=user_train, valid_user_dict=user_valid, test_user_dict=user_test, num_items=self.num_items, input_max_length=self.input_max_length, mode="train", device=self.device)
        valid_data = SASRecDataset(train_user_dict=user_train, valid_user_dict=user_valid, test_user_dict=user_test, num_items=self.num_items, input_max_length=self.input_max_length, mode="valid", device=self.device)
        test_data = SASRecDataset(train_user_dict=user_train, valid_user_dict=user_valid, test_user_dict=user_test, num_items=self.num_items, input_max_length=self.input_max_length, mode="test", device=self.device)

        loader_train = DataLoader(train_data, batch_size=batch_size, collate_fn=train_data.collate_fn)
        loader_val = DataLoader(valid_data, batch_size=valid_batch_size, collate_fn=valid_data.collate_fn)

        return loader_train, loader_val

    def format_data(self, data):

        tmp_data = data.copy()

        tmp_data.loc[:, "userId"] = [self.user_id_map[u] for u in tmp_data.userId]
        tmp_data.loc[:, "itemId"] = [self.item_id_map[u] for u in tmp_data.itemId]

        return tmp_data

    def generate_feature(self, userId, itemIds):
        return torch.LongTensor(np.array([userId])).to(self.device),\
               torch.LongTensor(np.array(self.user_dict[userId][-self.input_max_length:])).unsqueeze(1).to(self.device), \
               torch.LongTensor(np.array(itemIds)).unsqueeze(0).to(self.device)

    def format_data_single(self, userId, itemIds):
        if userId not in self.user_id_map:
            return None, None
        return self.user_id_map[userId], [self.item_id_map[itemId] for itemId in itemIds]


class SASRecDataset(Dataset):
    """
     SASRec dataset class in order to use Pytorch DataLoader
     """
    def __init__(self, train_user_dict, num_items, valid_user_dict=None, test_user_dict=None, input_max_length=200, mode="train", device="cpu"):
        super().__init__()

        self.mode = mode
        self.device = device
        self.num_items = num_items
        self.input_max_length = input_max_length
        self.train_user_dict = train_user_dict
        self.test_user_dict = test_user_dict
        self.valid_user_dict = valid_user_dict
        self.train_data = list(self.train_user_dict.items())
        self.test_data = list(self.test_user_dict.items())
        self.valid_data = list(self.valid_user_dict.items())

        if mode == "valid":
            assert valid_user_dict is not None
        elif mode == "test":
            assert valid_user_dict, test_user_dict is not None

    def collate_fn(self, batch):
        if self.mode == "train":
            user, seq, pos, neg = zip(
                *batch)
            return torch.LongTensor(user).to(self.device), torch.LongTensor(seq).to(self.device), \
                   torch.LongTensor(pos).to(self.device), torch.LongTensor(neg).to(self.device)
        else:
            user, seq, valid_item_idx = zip(*batch)
            return torch.LongTensor(user).to(self.device), torch.LongTensor(seq).to(self.device), \
                   torch.LongTensor(valid_item_idx).to(self.device)

    def __getitem__(self, index):

        if self.mode == "train":
            user, item_list = self.train_data[index]
            seq = np.zeros([self.input_max_length], dtype=np.long)
            pos = np.zeros([self.input_max_length], dtype=np.long)
            neg = np.zeros([self.input_max_length], dtype=np.long)
            nxt = item_list[-1]
            idx = self.input_max_length - 1

            ts = set(item_list)

            for i in reversed(item_list[:-1]):
                seq[idx] = i
                pos[idx] = nxt
                if nxt != 0:
                    neg[idx] = sample_neg_items_for_u(ts, n_sample_neg_items=1, start_item_id=1, end_item_id=self.num_items, sequential=False)
                nxt = i
                idx -= 1
                if idx == -1:
                    break
            return user, seq, pos, neg

        elif self.mode == "valid":
            seq = np.zeros([self.input_max_length], dtype=np.long)
            idx = self.input_max_length - 1
            user, target_item = self.valid_data[index]
            for i in reversed(self.train_user_dict[user]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            rated = set(self.train_user_dict[user])
            rated.add(target_item[0])
            valid_item_idx = [target_item[0]]
            for _ in range(100):
                t = sample_neg_items_for_u(rated, n_sample_neg_items=1, start_item_id=1, end_item_id=self.num_items, sequential=False)[0]
                valid_item_idx.append(t)
            return user, seq, valid_item_idx

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