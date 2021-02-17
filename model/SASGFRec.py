import torch
import logging
from time import time
from model.BaseModel import BaseModel
from utility.model_helper import EarlyStopping, adjust_learning_rate
from utility.components import PointWiseFeedForward, MultiHeadAttention
from utility.components import TimeEncode, PosEncode, EmptyEncode, TemporalAggregator, MergeLayer

# reference: https://github.com/pmixer/SASRec.pytorch.git


class SASGFRec(BaseModel):
    def __init__(self, num_user, num_node, num_relation, args):
        super(SASGFRec, self).__init__(args)

        self.num_user = num_user
        self.num_node = num_node
        self.num_relation = num_relation
        self.args = args

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.node_emb = torch.nn.Embedding(self.num_node + 1, self.hidden_units, padding_idx=0)

        if self.use_time == 'time':
            self.time_encoder = TimeEncode(expand_dim=self.hidden_units)
        elif self.use_time == 'pos':
            self.time_encoder = PosEncode(time_dim=self.hidden_units, seq_len=self.maxlen)
        elif self.use_time == 'empty':
            self.time_encoder = EmptyEncode(time_dim=self.hidden_units)
        else:
            raise ValueError('invalid time option!')


        #self.pos_emb = torch.nn.Embedding(self.maxlen, self.hidden_units)  # TO IMPROVE
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        hidden_units = self.hidden_units  # node_dim + time_dim
        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # new_attn_layer = torch.nn.MultiheadAttention(hidden_units,
            #                                              self.num_heads,
            #                                              self.dropout_rate)
            new_attn_layer = MultiHeadAttention(self.num_blocks, hidden_units,\
                                    hidden_units, hidden_units, dropout=self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.new_attn_layer_graph = torch.nn.MultiheadAttention(self.hidden_units,
                                                     self.num_heads,
                                                     self.dropout_rate)

        self.temporal_aggregator = TemporalAggregator(self.fan_outs, self.hidden_units, self.num_node, \
                            self.num_relation, num_layers=len(self.fan_outs), drop_out=self.dropout_rate,\
                                                      num_heads=self.num_heads, use_time=self.use_time)
        self.temporal_aggregator.node_embed = self.node_emb

        # Used for kg pre-train
        self.kg_aggregator = TemporalAggregator(self.fan_outs, self.hidden_units, self.num_node, \
                            self.num_relation, num_layers=len(self.fan_outs), drop_out=self.dropout_rate, \
                                                num_heads=self.num_heads, use_time="empty")
        self.kg_aggregator.node_embed = self.node_emb
        self.kg_aggregator.edge_embed = self.temporal_aggregator.edge_embed
        self.affinity_score = MergeLayer(self.hidden_units, self.hidden_units, self.hidden_units, 1)


    def temporal_graph_embedding(self, src_idx, cut_time_list, blocks):

        batch_size, maxlen = src_idx.shape

        for i, (src_ngh_idx, src_ngh_node_type, src_ngh_ts) in enumerate(reversed(blocks)):

            src_ngh_idx_reshape = src_ngh_idx.view(-1, self.num_neighbors)

            if len(blocks) == 1:
                dst_node_embed = self.node_emb(src_idx).view(-1, self.hidden_units).unsqueeze(
                    1)  # (batch_size * maxlen, 1, node_dim)
            else:
                dst_node_embed = self.node_emb(blocks[i + 1]).view(-1, self.hidden_units).unsqueeze(1)

            if i == 0:
                dst_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_list)).view(-1, self.hidden_units).unsqueeze(1)
                src_node_embed = self.node_emb(src_ngh_idx_reshape)  # (batch_size * maxlen, num_neighbors, node_dim)

            src_node_t_embed = self.time_encoder(src_ngh_ts.view(-1, self.num_neighbors))
            src_node_feat = src_node_embed + src_node_t_embed
            mask = ~(src_ngh_idx_reshape == 0).unsqueeze(-1)
            src_node_feat *= mask

            dst_node_feat = dst_node_embed + dst_node_t_embed
            # used for next iteration.
            src_node_feat = torch.transpose(src_node_feat, 0, 1)
            dst_node_feat = torch.transpose(dst_node_feat, 0, 1)
            src_node_embed, _ = self.attention_layers[i](dst_node_feat, src_node_feat, src_node_feat)
            src_node_embed = dst_node_feat + src_node_embed
            src_node_embed = src_node_embed.transpose(0, 1)
            src_node_embed = self.forward_layernorms[i](src_node_embed)
            src_node_embed = self.forward_layers[i](src_node_embed)
            dst_node_t_embed = self.time_encoder(src_ngh_ts.view(-1, self.num_neighbors))

        return src_node_embed.squeeze(1).reshape(batch_size, maxlen, self.hidden_units)

    def log2feats(self, log_seqs, seq_ts, blocks):

        seqs = self.node_emb(log_seqs)
        temporal_embedding = self.temporal_aggregator(blocks).view(-1, self.graph_maxlen, self.hidden_units)
        seqs[:, -self.graph_maxlen:, :] = temporal_embedding
        #seqs *= self.node_emb.embedding_dim ** 0.5
        seqs += self.time_encoder(seq_ts)

        seqs = self.dropout(seqs)

        timeline_mask = (log_seqs == 0).unsqueeze(-1)
        #seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            #seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask, mask=timeline_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            #seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            #seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        return log_feats

    def forward(self, user_ids, log_seqs, seq_ts, pos_seqs, neg_seqs, block):  # for training
        log_feats = self.log2feats(log_seqs, seq_ts, block)  # user_ids hasn't been used yet

        pos_embs = self.node_emb(pos_seqs)
        neg_embs = self.node_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, seq_ts, item_indices, block):  # for inference
        log_feats = self.log2feats(log_seqs, seq_ts, block)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :].unsqueeze(1)  # only use last QKV classifier, a waste

        item_embs = self.node_emb(item_indices)  # .squeeze(1) # (I, C)
        logits = final_feat.matmul(item_embs.transpose(1, 2))

        return logits.squeeze(1)  # preds # (U, I)

    def calc_loss(self, optimizer, batch_data):
        (u, seq, seq_ts, pos, neg, block) = batch_data
        pos_logits, neg_logits = self.forward(*batch_data)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(
            neg_logits.shape, device=self.device)
        optimizer.zero_grad()
        indices = pos != 0
        loss = self.criterion(pos_logits[indices], pos_labels[indices])
        loss += self.criterion(neg_logits[indices], neg_labels[indices])

        for param in self.node_emb.parameters():
            loss += self.args.l2_emb * torch.norm(param)
        loss.backward()
        optimizer.step()
        return loss

    def calc_kg_loss(self, optimizer, batch_data):
        src_blocks, dst_blocks, src_fake_blocks = batch_data
        src_embed = self.kg_aggregator(src_blocks)
        pos_embed = self.kg_aggregator(dst_blocks)
        neg_embed = self.kg_aggregator(src_fake_blocks)
        pos_score = self.affinity_score(src_embed, pos_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, neg_embed).squeeze(dim=-1)
        size = len(src_blocks[0][0])
        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=self.device)
            neg_label = torch.zeros(size, dtype=torch.float, device=self.device)

        optimizer.zero_grad()

        loss = self.criterion(pos_score, pos_label)
        loss += self.criterion(neg_score, neg_label)
        loss.backward()
        optimizer.step()
        return loss

    def reset_parameters(self):
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass  # just ignore those failed init layers

    def fit(self, loader_train, loader_val, loader_kg, optimizer):

        self.reset_parameters()

        earlyStopper = EarlyStopping(self.stopping_steps, self.verbose)
        self.train().to(device=self.device)

        logging.info(self)

        # Train CF
        best_epoch = -1
        n_kg_batch = int(len(loader_kg.dataset) / self.kg_batch_size)
        n_batch = int(len(loader_train.dataset) / self.batch_size)
        epoch_start_idx = 0
        for epoch in range(epoch_start_idx, self.num_epochs + 1):

            # if epoch % 5 == 0:
            #     time1 = time()
            #     total_loss = 0
            #     time2 = time()
            #     for step, batch in enumerate(loader_kg):
            #         loss = self.calc_kg_loss(optimizer, batch)
            #         total_loss += loss.item()
            #         if self.verbose and step % self.print_every == 0 and step != 0:
            #             logging.info(
            #                 'KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean '
            #                 'Loss {:.4f}'.format(
            #                     epoch, step, n_kg_batch, time() - time2, loss.item(), total_loss / step))
            #             time2 = time()
            # logging.info(
            #     'Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch,
            #                                                                                                   n_kg_batch,
            #                                                                                                   time() - time1,
            #                                                                                                   total_loss / n_kg_batch))

            time1 = time()
            total_loss = 0
            time2 = time()
            for step, batch_data in enumerate(loader_train):
                loss = self.calc_loss(optimizer, batch_data)
                total_loss += loss.item()
                if self.verbose and step % self.print_every == 0 and step != 0:
                    logging.info(
                        'Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean '
                        'Loss {:.4f}'.format(
                            epoch, step, n_batch, time() - time2, loss.item(), total_loss / step))
                    time2 = time()
            logging.info(
                'Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch,
                                                                                                               n_batch,
                                                                                                               time() - time1,
                                                                                                               total_loss / n_batch))

            if epoch % self.evaluate_every == 0:
                time1 = time()
                self.eval()
                ndcg, recall = self.evaluate(loader_val)
                logging.info(
                    'Evaluation: Epoch {:04d} | Total Time {:.1f}s | Recall {:.4f} NDCG {'':.4f}'.format(
                        epoch, time() - time1, recall, ndcg))

                earlyStopper(recall, self, self.save_dir, epoch, best_epoch)

                if earlyStopper.early_stop:
                    break
                self.train()

        adjust_learning_rate(optimizer, epoch, self.lr)
