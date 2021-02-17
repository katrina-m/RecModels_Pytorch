import torch
import numpy as np
from model.BaseModel import BaseModel
from utility.components import PointWiseFeedForward


class SASRec(BaseModel):
    def __init__(self, num_user, num_item, args):
        super(SASRec, self).__init__(args)

        self.num_user = num_user
        self.num_item = num_item
        self.args = args

        self.item_emb = torch.nn.Embedding(self.num_item + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(self.maxlen, self.hidden_units)  # TO IMPROVE
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_units,
                                                         self.num_heads,
                                                         self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def log2feats(self, log_seqs):

        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.dropout(seqs)

        timeline_mask = log_seqs == 0
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        #log_feats = log_feats[:, -1, :].unsqueeze(1)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :].unsqueeze(1)  # only use last QKV classifier, a waste

        item_embs = self.item_emb(item_indices)#.squeeze(1) # (I, C)
        logits = final_feat.matmul(item_embs.transpose(1, 2))

        return logits.squeeze(1)  # preds # (U, I)

    def calc_loss(self, optimizer, batch_data):
        (u, seq, pos, neg) = batch_data
        pos_logits, neg_logits = self.forward(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(
            neg_logits.shape, device=self.device)
        optimizer.zero_grad()
        indices = pos != 0
        loss = self.criterion(pos_logits[indices], pos_labels[indices])
        loss += self.criterion(neg_logits[indices], neg_labels[indices])

        for param in self.item_emb.parameters():
            loss += self.args.l2_emb * torch.norm(param)
        loss.backward()
        optimizer.step()
        return loss

    def reset_parameters(self):
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass  # just ignore those failed init layers

