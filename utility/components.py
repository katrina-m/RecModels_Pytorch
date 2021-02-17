import torch
import numpy as np
import logging
import dgl


class Aggregator(torch.nn.Module):
    """
    Neighbor aggregator.
    """

    def __init__(self, in_dim, out_dim, num_relations, dropout, aggregator_type="graphsage", propagate_type="residual"):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.num_relations = num_relations
        self.aggregator_type = aggregator_type
        self.propagate_type = propagate_type

        self.message_dropout = torch.nn.Dropout(dropout)

        self.relation_embed = torch.nn.Embedding(self.num_relations, self.in_dim, padding_idx=0)  # updated later
        # updated transformation relation matrix
        self.W_R = torch.nn.Parameter(torch.Tensor(self.num_relations, self.in_dim, self.in_dim))

        if aggregator_type == 'gcn':
            self.W = torch.nn.Linear(self.in_dim, self.out_dim)  # W in Equation (6)
        elif aggregator_type == 'graphsage':
            self.W = torch.nn.Linear(self.in_dim * 2, self.out_dim)  # W in Equation (7)
        elif aggregator_type == 'bi-interaction':
            self.W1 = torch.nn.Linear(self.in_dim, self.out_dim)  # W1 in Equation (8)
            self.W2 = torch.nn.Linear(self.in_dim, self.out_dim)  # W2 in Equation (8)
        else:
            raise NotImplementedError

        if propagate_type == "residual":
            if self.in_dim != out_dim:
                self.res_fc = torch.nn.Linear(
                    self.in_dim, out_dim, bias=False)
            else:
                self.res_fc = torch.nn.Identity()

        self.activation = torch.nn.LeakyReLU()

    def edge_attention(self, edges):
        r_mul_t = torch.matmul(edges.srcdata['node_feat'], self.W_r)  # (n_edge, relation_dim)
        r_mul_h = torch.matmul(edges.srcdata['node_feat'], self.W_r)  # (n_edge, relation_dim)
        r_embed = self.relation_embed(edges.data['type'])  # (1, relation_dim)
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)  # (n_edge, 1)
        return {'attention_score': att}

    def compute_attention(self, g):
        with g.local_scope():
            for i in range(self.num_relations):
                edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)
                self.W_r = self.W_R[i]
                g.apply_edges(self.edge_attention, edge_idxs)

            return g.edata.pop('attention_score')

    def forward(self, g, entity_embed):
        g = g.local_var()
        if g.is_block:
            h_src = entity_embed
            h_dst = entity_embed[:g.num_dst_nodes()]
            g.srcdata['node_feat'] = h_src
            g.dstdata['node_feat'] = h_dst
        else:
            g.ndata['node_feat'] = entity_embed
            h_dst = entity_embed

        g.edata["attention_score"] = self.compute_attention(g)

        g.update_all(dgl.function.u_mul_e('node_feat', 'attention_score', 'side_feat'),
                     dgl.function.sum('side_feat', 'neighbor_feat'))

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            out = self.activation(
                self.W(g.dstdata['node_feat'] + g.dstdata['neighbor_feat']))  # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            out = self.activation(
                self.W(torch.cat([g.dstdata['node_feat'], g.dstdata['neighbor_feat']],
                                 dim=1)))  # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            out1 = self.activation(
                self.W1(g.dstdata['node_feat'] + g.dstdata['neighbor_feat']))  # (n_users + n_entities, out_dim)
            out2 = self.activation(
                self.W2(g.dstdata['node_feat'] * g.dstdata['neighbor_feat']))  # (n_users + n_entities, out_dim)
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)

        if self.propagate_type == "residual":
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self.out_dim)
                out = out + resval.squeeze(1)
        return out


class TimeEncode(torch.nn.Module):
    def __init__(self, time_dim, factor=5):
        super(TimeEncode, self).__init__()

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
    def __init__(self, time_dim, seq_len):
        super().__init__()

        self.pos_embeddings = torch.nn.Embedding(num_embeddings=seq_len, embedding_dim=time_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.time_dim)
        return out


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class TemporalAggregator(torch.nn.Module):

    def __init__(self, fan_outs, hidden_units, num_nodes, num_relations, num_layers, use_time, num_heads, drop_out, attn_mode="prod", agg_method="attn"):
        super(TemporalAggregator, self).__init__()

        self.drop_out = drop_out
        self.fan_outs = fan_outs
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.num_nodes = num_nodes
        self.hidden_units = hidden_units
        self.use_time = use_time
        self.agg_method = agg_method
        self.attn_mode = attn_mode
        self.num_heads = num_heads
        self.logger = logging.getLogger(__name__)

        self.edge_embed = torch.nn.Embedding(self.num_relations, self.hidden_units, padding_idx=0)
        self.node_embed = torch.nn.Embedding(self.num_nodes + 1, self.hidden_units, padding_idx=0)

        self.W_R = torch.nn.Parameter(torch.Tensor(self.num_relations, self.hidden_units, self.hidden_units))
        self.merge_layer = MergeLayer(self.hidden_units, self.hidden_units, self.hidden_units, self.hidden_units)

        if self.use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(time_dim=self.hidden_units)
        elif self.use_time == 'pos':
            assert(self.fan_outs is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(time_dim=self.hidden_units, seq_len=self.fan_outs[0])
        elif self.use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(time_dim=self.hidden_units)
        else:
            raise ValueError('invalid time option!')


        if self.agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.hidden_units,
                                                                  self.hidden_units,
                                                                  self.hidden_units,
                                                                  attn_mode=self.attn_mode,
                                                                  n_head=self.num_heads,
                                                                  drop_out=self.drop_out) for _ in range(self.num_layers)])
        elif self.agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.hidden_units,
                                                                 self.hidden_units,
                                                                 self.hidden_units) for _ in range(self.num_layers)])
        elif self.agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.hidden_units,
                                                                 self.hidden_units) for _ in range(self.num_layers)])
        else:

            raise ValueError('invalid agg_method value, use attn or lstm')

        pass

    def forward(self, blocks):
        return self.tem_conv(blocks)

    def tem_conv(self, blocks):

        for i, (src_ngh_idx, dst_idx, src_node_type, dst_node_type, src_ngh_edge_type, src_ngh_ts, dst_ts) in enumerate(reversed(blocks)): # The first contains the original src nodes.

            # Reshape
            dst_ts = dst_ts.unsqueeze(1)
            src_node_raw_feat = self.node_embed(src_ngh_idx)  # (batch_size, -1)

            if i == 0:
                src_ngh_feat = src_node_raw_feat
            else:
                src_ngh_feat = src_ngh_feat.view(-1, self.fan_outs[i], self.hidden_units)

            # query node always has the start time -> time span == 0
            dst_node_t_embed = self.time_encoder(torch.zeros_like(dst_ts))
            dst_node_feat = self.node_embed(dst_idx)

            src_ngh_t_delta = dst_ts - src_ngh_ts
            src_ngh_t_embed = self.time_encoder(src_ngh_t_delta)
            src_ngn_edge_feat = self.edge_embed(src_ngh_edge_type)

            # attention aggregation
            mask = src_ngh_idx == 0
            attn_m = self.attn_model_list[i]

            local, weight = attn_m(dst_node_feat,
                                   dst_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   src_ngn_edge_feat,
                                   mask)

            src_ngh_feat = dst_node_feat + local

        return src_ngh_feat


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

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

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


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None, attn_mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1e10)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(torch.nn.Module):
    ''' Multi-Head Attention module '''

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

    def forward(self, q, k, v, mask=None, attn_mask=None):

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
        output, attn = self.attention(q, k, v, mask=mask, attn_mask=attn_mask)

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


