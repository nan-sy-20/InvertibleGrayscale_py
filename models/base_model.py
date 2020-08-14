from abc import ABC, abstractmethod
import torch
import os
from collections import OrderedDict

import models.networks_old as networks

class BaseModel(ABC):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids \
            else torch.device('cpu')

        if self.isTrain:
            self.save_dir = os.path.join(opt.log_dir, 'checkpoints')
            os.makedirs(self.save_dir, exist_ok=True)

        if opt.preprocess != 'scale_width':
            torch.backends.cudnn.benchmark = True

        self.loss_names = []
        self.image_paths = []
        self.metric_names = []
        self.visual_names = []
        self.optimizers = []

    @staticmethod
    def modify_command_options(parser, isTrain):
        return parser

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass


    @abstractmethod
    def save_networks(self, epoch):
        pass

    def setup(self, opt, verbose=True):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or self.opt.continue_train:
            self.load_networks(verbose=verbose)

    def update_learning_rate(self, logger=None):
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.opt.metric)
            else:
                scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        if logger is not None:
            logger.print_info('learning rate = %.7f\n' % lr)
        else:
            print('\u270f learning rate = %.7f' % lr)


    def get_current_losses(self):
        losses_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses_ret[name] = float(getattr(self, 'loss_' + name))
        return losses_ret

    def get_current_metrics(self):
        metrics_ret = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str):
                metrics_ret[name] = float(getattr(self, 'metric_' + name))
        return metrics_ret


    def get_current_visuals(self):
        visuals_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visuals_ret[name] = getattr(self, name)
        return visuals_ret

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad