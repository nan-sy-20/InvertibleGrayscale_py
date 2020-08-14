import warnings
import numpy as np
import torch
import random
import time
import os

from data import create_train_dataloader

from utils.logger import Logger
from utils.visualizer import Visualizer

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer():
    def __init__(self, task_name):
        if 'train' in task_name:
            from options.train_options import TrainOptions as Options
            from models import create_model as create_model
        else:
            raise NotImplementedError('\u2757 Task [%s] is not found!' % task_name)

        # Settings
        opt = Options(task_name).parse()

        if opt.phase != 'train':
            warnings.warn('\u2757 Phase is wrong for training.')

        # seed
        set_seed(opt.seed)

        # Create dataloader
        dataloader = create_train_dataloader(opt)

        # Create model
        model = create_model(opt)
        model.setup(opt)
        visualizer = Visualizer(opt, task_name)
        logger = Logger(opt)

        self.opt = opt
        self.dataloader = dataloader
        self.model = model
        self.logger = logger
        self.visualizer = visualizer


    def evaluate(self, iter):
        self.model.evaluate_model(iter)

    def start(self):
        opt = self.opt
        dataloader = self.dataloader
        dataset_size = len(dataloader)
        model = self.model
        logger = self.logger

        start_epoch = opt.epoch_base
        end_epoch = opt.epoch_base + opt.nepochs + opt.nepochs_decay - 1
        total_iter = opt.iter_base

        for epoch in range(start_epoch, end_epoch+1):
            # timer for the entire epoch
            epoch_start_time = time.time()
            epoch_iter = 0
            iter_data_time = time.time()
            self.visualizer.reset()
            self.visualizer.vis.close()
            for i, data in enumerate(dataloader):
                iter_start_time = time.time()

                if total_iter % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time # time consumed for loading data in each iteration

                epoch_iter += opt.batch_size

                model.set_input(data)
                model.optimize_parameters()

                if total_iter % opt.display_freq == 0:
                    save_result = total_iter % opt.update_html_freq == 0
                    self.visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # print losses
                if total_iter % opt.print_freq == 0:
                    t_comp = time.time() - iter_start_time
                    losses = model.get_current_losses()
                    logger.print_current_losses(epoch, total_iter, losses, t_comp, t_data)

                    if opt.display_id > 0 :
                        self.visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                # print metrics
                if total_iter % opt.eval_freq == 0:
                    self.evaluate(total_iter)
                    metrics = model.get_current_metrics()
                    self.logger.print_current_metrics(epoch, total_iter, metrics)
                    if opt.display_id > 0:
                        self.visualizer.plot_current_metrics(epoch, float(epoch_iter) / dataset_size, metrics)

                total_iter += 1
                iter_data_time = time.time()

            # print
            logger.print_info('\U0001F899 End of epoch %d / %d \t Time Taken: %.2f sec' % (epoch, end_epoch, time.time() - epoch_start_time))

            # save checkpoints
            if epoch % opt.save_epoch_freq == 0 or epoch == end_epoch:
                self.evaluate(total_iter)
                model.save_networks(epoch)

            # update learning rate
            model.update_learning_rate(logger)