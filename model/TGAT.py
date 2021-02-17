import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import logging
from time import time
import dgl


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention module
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v, bias=False)
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)

        self.fc = torch.nn.Linear(n_head * d_v, d_model)

        torch.nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class MapBasedMultiHeadAttention(torch.nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = torch.nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = torch.nn.LayerNorm(d_model)

        self.fc = torch.nn.Linear(n_head * d_v, d_model)

        self.act = torch.nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = torch.nn.Linear(2 * d_k, 1, bias=False)

        torch.nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()

        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)

        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)

        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk

        # Map based Attention
        #output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]

        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn


def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        basis_freq = self.basis_freq.view(1, 1, -1)
        map_ts = ts * basis_freq  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = torch.nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim

        self.att_dim = feat_dim + edge_dim + time_dim

        self.act = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=self.att_dim,
                                  hidden_size=self.feat_dim,
                                  num_layers=1,
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)

        _, (hn, _) = self.lstm(seq_x)

        hn = hn[-1, :, :]  # hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2)  # [B, N, De + D]
        hn = seq_x.mean(dim=1)  # [B, De + D]
        output = self.merger(hn, src_x)
        return output, None


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim

        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        #self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        #self.act = torch.nn.ReLU()

        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                                        d_model=self.model_dim,
                                                        d_k=self.model_dim // n_head,
                                                        d_v=self.model_dim // n_head,
                                                        dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                                                d_model=self.model_dim,
                                                                d_k=self.model_dim // n_head,
                                                                d_v=self.model_dim // n_head,
                                                                dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, 1, D + De + Dt] -> [B, 1, D]

        mask = torch.unsqueeze(mask, dim=2)  # mask [B, N, 1]
        mask = mask.permute([0, 2, 1])  # mask [B, 1, N]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()
        attn = attn.squeeze()

        output = self.merger(output, src)
        return output, attn


class TGAT(torch.nn.Module):
    def __init__(self, num_node, num_relation, args):
        super(TGAT, self).__init__()
        self.__dict__.update(vars(args))

        self.num_relations = num_relation
        self.num_nodes = num_node
        self.num_layers = self.num_layers
        self.logger = logging.getLogger(__name__)
        #self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        #self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))

        self.edge_raw_embed = torch.nn.Embedding(num_relation, self.node_dim, padding_idx=0)
        # from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding(num_node, self.node_dim, padding_idx=0)
        # from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        self.feat_dim = self.node_dim

        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim

        self.W_R = torch.nn.Parameter(torch.Tensor(self.num_relations, self.n_feat_dim, self.e_feat_dim))

        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)

        if self.agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                                  self.feat_dim,
                                                                  self.feat_dim,
                                                                  attn_mode=self.attn_mode,
                                                                  n_head=self.num_heads,
                                                                  drop_out=self.drop_out) for _ in range(self.num_layers)])
        elif self.agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(self.num_layers)])
        elif self.agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(self.num_layers)])
        else:

            raise ValueError('invalid agg_method value, use attn or lstm')

        if self.use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.node_dim)
        elif self.use_time == 'pos':
            assert(self.num_neighbors is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.node_dim, seq_len=self.num_neighbors)
        elif self.use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.node_dim)
        else:
            raise ValueError('invalid time option!')

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)
        self.criterion = torch.nn.BCELoss()

        #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)

        # Merge layer
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score

    def contrast(self, src_idx_l, pos_idx_l, neg_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv_v1(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_embed = self.tem_conv_v1(pos_idx_l, cut_time_l, self.num_layers, num_neighbors)
        neg_embed = self.tem_conv_v1(neg_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, pos_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, neg_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def att_score(self, edges):

        # Equation (4)
        src_node_feat = edges.data["src_node_feat"]
        dst_node_feat = self.node_raw_embed(edges.dst[dgl.NID])
        mask = edges.data["mask"]
        r_mul_t = torch.matmul(src_node_feat, self.W_r)                       # (n_edge, relation_dim)
        r_mul_h = torch.matmul(dst_node_feat, self.W_r)                       # (n_edge, relation_dim)
        r_embed = self.edge_raw_embed(edges.data['type'])                                               # (1, relation_dim)
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)   # (n_edge, 1)
        att_feat = mask*att*src_node_feat
        return {'att': att, 'att_feat':att_feat}

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20):
        assert(curr_layers >= 0)

        batch_size = len(src_idx_l)

        src_node_batch_th = torch.LongTensor(src_idx_l).to(self.device)
        cut_time_l_th = torch.FloatTensor(cut_time_l).to(self.device)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)  
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th).to(self.device))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l,
                                               cut_time_l,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors)

            src_ngh_node_batch, src_ngh_eType_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                src_idx_l,
                cut_time_l,
                num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(self.device)
            src_ngh_eType_batch = torch.from_numpy(src_ngh_eType_batch).long().to(self.device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(self.device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)

            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eType_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   src_ngn_edge_feat,
                                   mask)
            return local

    def fit(self, train_loader, val_loader, nn_val_loader, optimizer):

        # Training use only training graph
        self.g = train_loader.dataset.ngh_finder
        self.to(self.device)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     self = torch.nn.DataParallel(self)

        self.train()
        for epoch in range(self.num_epochs):

            acc, ap, f1, auc, m_loss = [], [], [], [], []

            n_batch = int(len(train_loader.dataset) / self.batch_size)
            time1 = time()
            total_loss = 0
            time2 = time()
            for step, batch in enumerate(train_loader):

                src_l_cut, dst_l_cut, ts_l_cut, src_l_fake, dst_l_fake = batch
                size = len(src_l_cut)
                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=self.device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=self.device)

                optimizer.zero_grad()

                pos_prob, neg_prob = self.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, self.num_neighbors)

                #print("output_size", pos_prob.size())

                loss = self.criterion(pos_prob, pos_label)
                loss += self.criterion(neg_prob, neg_label)

                loss.backward()
                optimizer.step()
                # get training results
                with torch.no_grad():
                    self.eval()
                    pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                    pred_label = pred_score > 0.5
                    true_label = np.concatenate([np.ones(size), np.zeros(size)])
                    acc.append((pred_label == true_label).mean())
                    ap.append(average_precision_score(true_label, pred_score))
                    # f1.append(f1_score(true_label, pred_label))
                    m_loss.append(loss.item())
                    total_loss += loss.item()
                    auc.append(roc_auc_score(true_label, pred_score))

                if self.verbose and step % self.print_every == 0 and step != 0:
                    logging.info(
                        'Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean '
                        'Loss {:.4f}'.format(
                            epoch, step, n_batch, time() - time2, loss.item(), total_loss / step))
                    time2 = time()

            # validation phase use all information
            self.eval()
            self.ngh_finder = val_loader.dataset.ngh_finder
            val_acc, val_ap, val_f1, val_auc = self.evaluate(val_loader)
            nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = self.evaluate(nn_val_loader)
            self.train()

            logging.info('epoch: {}:'.format(epoch))
            logging.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
            logging.info('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
            logging.info('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
            logging.info('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))

    def evaluate(self, val_loader):

        val_acc, val_ap, val_f1, val_auc = [], [], [], []

        batch_size = val_loader.batch_size
        with torch.no_grad():
            for batch in val_loader:
                src_l_cut, dst_l_cut, ts_l_cut, src_l_fake, dst_l_fake = batch

                pos_prob, neg_prob = self.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, self.num_neighbors)

                pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])

                val_acc.append((pred_label == true_label).mean())
                val_ap.append(average_precision_score(true_label, pred_score))
                val_f1.append(f1_score(true_label, pred_label))
                val_auc.append(roc_auc_score(true_label, pred_score))

        return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)
