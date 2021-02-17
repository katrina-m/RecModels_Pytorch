import torch
import os
from collections import OrderedDict
import numpy as np


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model_by_state(model, model_dir, current_epoch, last_best_epoch=None):
    """
    Save purely the model state.
    :param model:
    :param model_dir:
    :param current_epoch:
    :param last_best_epoch:
    :return:
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


def load_model_by_state(model, model_path):
    """
    Load purely the model state.
    :param model:
    :param model_path:
    :return:
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]              # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)
    return model


def load_model(model_path):
    """
    Load the whole model.
    :param model_path:
    :return:
    """
    model_state_file = os.path.join(model_path, 'model.pth')
    model = torch.load(model_state_file)
    model.eval()
    return model


def save_model(model, model_path):
    """
    Save the whole model.
    :param model:
    :param model_path:
    :return:
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_state_file = os.path.join(model_path, 'model.pth')
    model = torch.save(model, model_state_file)
    return model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_epoch = -1

    def __call__(self, val_loss, model, model_dir, current_epoch, last_best_epoch):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir, current_epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir, current_epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_dir, current_epoch):
        """
        Saves model when validation loss decrease.
        :param val_loss:
        :param model:
        :return:
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model_by_state(model, model_dir, current_epoch, last_best_epoch=self.best_epoch)
        self.best_epoch = current_epoch
        self.val_loss_min = val_loss
