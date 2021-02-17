import torch
from time import time
import logging
from utility.model_helper import *
from utility.metrics import *
import abc
from tqdm import tqdm
import numpy as np


class BaseModel(torch.nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.__dict__.update(vars(args))

    @abc.abstractmethod
    def update_loss(self, optimizer, batch_data):
        """
        This method is used for calculate the loss and update the gradient based on the given batch_data.
        :param model:
        :param optimizer:
        :param batch_data:
        :return:
        """
        pass

    @abc.abstractmethod
    def reset_parameters(self):
        pass

    @abc.abstractmethod
    def predict(self, data):
        """
        This method is used for prediction based on the given data.
        :param data:
        :return:
        """
        pass

    def fit(self, loader_train, loader_val, optimizer):

        self.reset_parameters()

        earlyStopper = EarlyStopping(self.stopping_steps, self.verbose)
        self.train().to(device=self.device)

        logging.info(self)

        epoch_start_idx = 0

        # initialize metrics
        best_epoch = -1

        n_batch = int(len(loader_train.dataset) / self.batch_size)

        for epoch in range(epoch_start_idx, self.num_epochs + 1):
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
                f1, auc = self.evaluate_ctr(loader_val)
                logging.info(
                    'Evaluation: Epoch {:04d} | Total Time {:.1f}s | Recall {:.4f} NDCG {'':.4f}'.format(
                        epoch, time() - time1, recall, ndcg))

                earlyStopper(recall, self, self.save_dir, epoch, best_epoch)

                if earlyStopper.early_stop:
                    break
                self.train()

        adjust_learning_rate(optimizer, epoch, self.lr)

    def evaluate(self, loader):

        num_test_user = len(loader.dataset)

        NDCG = 0
        HT = 0

        with torch.no_grad():
            with tqdm(total=int(num_test_user / self.valid_batch_size + 1), desc='Evaluating Iteration') as pbar:
                for batch_input in loader:
                    predictions = -self.predict(*batch_input)
                    rank_indices = torch.argsort(predictions).argsort()
                    rank_indices = rank_indices.cpu().numpy()[:,0]
                    NDCG += np.sum((rank_indices < self.K) * (1 / np.log2(rank_indices + 2)))
                    HT += np.sum(rank_indices < self.K)
                    pbar.update(1)

        return NDCG / num_test_user, HT / num_test_user

    # ----------------------------------------Used for calculating F1 score-------------------------------------------

    def fit_ctr(self, loader_train, loader_val, optimizer):

        self.reset_parameters()
        earlyStopper = EarlyStopping(self.patience, self.verbose)

        self.train().to(device=self.device)
        logging.info(self)

        best_epoch = -1

        epoch_start_idx = 0
        n_batch = int(len(loader_train.dataset) / self.batch_size)
        for epoch in range(epoch_start_idx, self.num_epochs + 1):
            time1 = time()
            total_loss = 0
            time2 = time()
            for step, batch_data in enumerate(loader_train):
                optimizer.zero_grad()
                batch_feature, batch_labels = batch_data
                logits = self.predict(batch_feature)
                loss = self.criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
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
                accuracy, f1_score = self.evaluate_ctr(loader_val)
                logging.info(
                    'Evaluation: Epoch {:04d} | Total Time {:.1f}s | Accuracy {:.4f} F1 {:.4f}'.format(
                        epoch, time() - time1, accuracy, f1_score))

                earlyStopper(f1_score, self)
                # 若满足 early stopping 要求
                if earlyStopper.early_stop:
                    earlyStopper.save_checkpoint(f1_score, self, self.save_dir, epoch, best_epoch)
                    best_epoch = epoch

    # def evaluate_ctr(self, loader_val):
    #
    #     targets = []
    #     predicts = []
    #     with torch.no_grad():
    #         with tqdm(total=len(loader_val.dataset) / self.valid_batch_size + 1, desc='Evaluating Iteration') as pbar:
    #             for batch_features, batch_labels in loader_val:
    #                 logits = self.predict(batch_features)
    #                 preds = (torch.sigmoid(logits) > 0.5)
    #                 targets = targets + batch_labels.cpu().numpy().tolist()
    #                 predicts = predicts + preds.cpu().numpy().tolist()
    #                 pbar.update(1)
    #
    #     return calc_metrics_at_k_ctr(predicts, targets)

    def evaluate_ctr(self, loader_val):

        f1_scores = []
        roc_auc_scores = []
        with torch.no_grad():
            with tqdm(total=int(len(loader_val.dataset) / self.valid_batch_size) + 1, desc='Evaluating Iteration') as pbar:
                for batch_input in loader_val:
                    logits = self.predict(*batch_input)
                    pos_logits = logits[:, 0]
                    neg_logits = logits[:, 1]
                    pos_preds = (torch.sigmoid(pos_logits) > 0.5).cpu().numpy().flatten()
                    neg_preds = (torch.sigmoid(neg_logits) > 0.5).cpu().numpy().flatten()
                    pos_labels = np.ones(len(batch_input[0])*1)
                    neg_labels = np.zeros(len(batch_input[0])*1)
                    f1_scores.append(f1_score(np.concatenate([pos_preds, neg_preds]), np.concatenate([pos_labels, neg_labels])))
                    roc_auc_scores.append(roc_auc_score(np.concatenate([pos_preds, neg_preds]), np.concatenate([pos_labels, neg_labels])))
                    pbar.update(1)

        f1_scores = np.mean(f1_scores)
        auc_scores = np.mean(roc_auc_scores)
        print(f"F1:{f1_scores}, AUC:{auc_scores}")
        return f1_scores, auc_scores