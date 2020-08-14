import argparse
import torch
import os
import pickle

import models
import data

class BaseOptions():

    def __init__(self):
        self.isTrain = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--data_dir', required=True,
                            help='path to images (should have subfolders trainA, trainB, valA, valB, train, val, etc)')
        parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--seed', type=int, default=1234, help='random seed')

        parser.add_argument('--norm_type', type=str, default='batch')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='single_decolor',
                            help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        #parser.add_argument('--batch_size', type=int, default=6, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--aspect_ratio', type=float, default=1.0,
                            help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--max_dataset_size', type=int, default=-1,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        #parser.add_argument('--load_in_memory', action='store_true', help='whether you will load the data into the memory to bypass the IO.')


        return parser



    def gather_options(self):
        parser = argparse.ArgumentParser(
            prog='configurations',
            description='Settings:',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser = self.initialize(parser)

        # get the basic settings
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = '-' * 20 + 'Options' + '-' * 20 + '\n'
        for k, v in sorted(vars(opt).items()):
            print_default_v = ''
            default_v = self.parser.get_default(k)
            if v != default_v:
                print_default_v = '\t[default: %s]' % str(default_v)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), print_default_v)
        message += '-' * 20 + 'End' + '-' * 20 + '\n'
        print(message)

        # save to disk under train mode
        if self.isTrain:
            # create checkpoints dir
            os.makedirs(opt.log_dir, exist_ok=True)
            with open(os.path.join(opt.log_dir, 'opt.txt'), 'wt') as file:
                file.write(message)
            with open(os.path.join(opt.log_dir, 'opt.pkl'), 'wb') as file:
                pickle.dump(opt, file)

    def check_freq(self, opt):
        assert opt.eval_freq != opt.print_freq, '\u2755 eval frequency should not be equal to print frequency!'
        assert opt.print_freq != opt.display_freq, '\u2755 print frequency should not be equal to print frequency!'

    def parse(self, verbose=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        # check frequency
        self.check_freq(opt)

        if verbose:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        return opt





