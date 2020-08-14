import torch
import importlib
import os

from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_mode):
    dataset_filename = 'data.' + dataset_mode + '_dataset'
    datasetlib = importlib.import_module(dataset_filename)
    target_dataset_name = dataset_mode.replace('_', '') + 'dataset'
    dataset = None
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
            and issubclass(cls, BaseDataset):
            dataset = cls
    if dataset is None:
        raise NotImplementedError('In %s.py there should be a subclass of BaseDataset with class name that matches %s in lowercase.'\
                                  % (dataset_filename, target_dataset_name))
    return dataset


def create_dataloader(opt, verbose=True):
    dataloader = CustomDataLoader(opt, verbose)
    return dataloader.load_data()

class CustomDataLoader():
    def __init__(self, opt, verbose=True):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)

        if verbose:
            print('\u270f Dataset [%s] was created.' % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = opt.batch_size,
            shuffle = not opt.serial_batches,
            num_workers = int(opt.num_threads)
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


def create_eval_dataloader(opt):
    opt.no_flip = True
    opt.serial_batches = True
    opt.batch_size = opt.eval_batch_size
    opt.phase = 'eval'
    opt.dataroot = os.path.join(opt.data_dir, 'eval')
    dataloader = CustomDataLoader(opt)
    print('\u270f %s dataloader was created.' % opt.phase)
    return dataloader.load_data()

def create_train_dataloader(opt):
    opt.no_flip = True
    opt.serial_batches = False
    opt.batch_size = opt.train_batch_size
    opt.phase = 'train'
    opt.dataroot = os.path.join(opt.data_dir, 'train')
    dataloader = CustomDataLoader(opt)
    print('\u270f %s dataloader was created.' % opt.phase)
    return dataloader.load_data()

# def create_dataset(opt):
#     data_loader = CustomDataLoader(opt)
#     dataset = data_loader.load_data()
#     return dataset

def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options