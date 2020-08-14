import torch
import torch.nn as nn
from scipy import ndimage
from skimage import color
from torchvision.models import vgg19


class InvertibilityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    def forward(self, input_img, restored_img):
        return self.loss(input_img, restored_img)

class GrayscaleConformityLoss(nn.Module):
    def __init__(self, img_shape, gpu_ids, c_weight=1e-7, ls_weight=0.5):
        super().__init__()

        self.c_weight = c_weight
        self.ls_weight = ls_weight
        self.vgg_layer_idx = 21
        self.threshold = 70/127.0

        if len(gpu_ids) > 0:
            self.vgg = vgg19(pretrained=True).features[:self.vgg_layer_idx]
            self.vgg.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            self.vgg = torch.nn.DataParallel(self.vgg, gpu_ids)
        self.vgg.eval()

        self.mse = nn.MSELoss()

        self.zeros = torch.zeros(img_shape).to(gpu_ids[0])
    def lightness(self, converted_g, input_gray):
        loss = torch.mean(torch.max(torch.abs(input_gray - converted_g)- self.threshold, self.zeros))
        return loss

    def contrast(self, converted_g, input_img):
        def _rescale(img):
            # img [-1,1]
            img = (img + 1) / 2 * 255
            img[:, 0, :, :] = img[:, 0, :, :] - 123.68
            img[:, 1, :, :] = img[:, 1, :, :] - 116.779
            img[:, 2, :, :] = img[:, 2, :, :] - 103.939
            return img
        vgg_g = self.vgg(_rescale(converted_g.repeat(1, 3, 1, 1)))
        vgg_o = self.vgg(_rescale(input_img))
        return self.mse(vgg_g, vgg_o)

    def local_structure(self, converted_g, input_img):
        def _tv(img):
            # total_variation = torch.sum(torch.abs(img[:,:,:,1:] - img[:,:,:,:-1])) + \
            #     torch.sum(torch.abs(img[:,:,1:,:] - img[:,:,:-1,:]))
            batch_size, ch, h, w = img.size()
            tv_h = torch.pow(img[:,:,1:,:] - img[:,:,:-1,:], 2).sum()
            tv_w = torch.pow(img[:,:,:,1:] - img[:,:,:,:-1], 2).sum()
            total_variation = 0.5*(tv_h + tv_w)/(batch_size*ch*h*w)
            return total_variation
        # batch_size, _, w, h = converted_g.size()
        # gray_n_pixels = batch_size * w * h
        # rgb_n_pixels = batch_size * w * h *3
        # gray_lv = _tv(converted_g) / gray_n_pixels
        # rgb_lv = _tv(input_img) / rgb_n_pixels
        gray_lv = _tv(converted_g)
        rgb_lv = _tv(input_img)
        loss = abs(gray_lv - rgb_lv)
        return loss

    def forward(self, converted_g, input_img):
        input_gray  = torch.unsqueeze(input_img[:,0,:,:]*0.2989 + input_img[:,1,:,:]*0.5870 + input_img[:,2,:,:]*0.1140, 1)
        l_loss = self.lightness(converted_g, input_gray)
        c_loss = self.contrast(converted_g, input_img)
        ls_loss = self.local_structure(converted_g, input_img)
        return l_loss + (self.c_weight * c_loss + self.ls_weight * ls_loss)