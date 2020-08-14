import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os

from utils.pytorch_ssim import pytorch_ssim

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array   # single image
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        if len(image_tensor.shape) == 4:
            image_numpy = image_tensor[0].cpu().float().numpy()
        else:
            image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 4:
        image_numpy = image_numpy[0]
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))

# metrics
def compute_psnr(prediction, target):
    assert len(prediction.shape) == 4, '\u2755 Prediction tensor dimension error.'
    assert len(target.shape) == 4, '\u2755 Target tensor dimension error.'

    pred = (prediction + 1.0) / 2.0
    targ = (target + 1.0) / 2.0
    mse = torch.mean((pred - targ) ** 2, dim=(1,2,3))
    psnr = 10 * np.log10(1. / mse.detach().cpu())
    assert psnr.shape == mse.shape
    return torch.mean(psnr) # np.mean(psnr)

def compute_ssim(prediction, target):
    # https://github.com/Po-Hsun-Su/pytorch-ssim
    pred = (prediction + 1.0) / 2.0
    targ = (target + 1.0) / 2.0
    ssim = pytorch_ssim.SSIM()
    return ssim(pred, targ)

def compute_ae():
    return

# network
def load_network(net, load_path, verbose=True):
    if verbose:
        print('\u270f Load network at %s' % load_path)
    weights = torch.load(load_path)
    if isinstance(net, nn.DataParallel):
        net = net.module
    net.load_state_dict(weights)
    return net

def clear_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    else:
        os.system('rm -r %s/*' % folder)

def find_nth(string, substring, n):
    if n==1:
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)