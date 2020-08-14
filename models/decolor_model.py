import torch
import torch.nn as nn
import os

from .base_model import BaseModel
import models.losses as losses
import models.networks_old as networks

from data import create_eval_dataloader

from utils import util

class DecolorModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, isTrain):
        parser = BaseModel.modify_command_options(parser, isTrain)
        parser.set_defaults(norm='batch', init_type='normal', dataset_mode='single_decolor')
        # model options
        parser.add_argument('--n_feat', type=int, default=64)
        if isTrain:
            parser.set_defaults(init_type='normal', init_gain=0.02)
            parser.add_argument('--c_weight', type=float, default=1e-7)
            parser.add_argument('--ls_weight', type=float, default=0.5)

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # visual names
        self.visual_names = ['real_rgb', 'decolor_gray', 'rec_rgb']

        # loss name
        self.loss_names = ['invertibility', 'grayscale_conformity', 'total']

        # metric name
        self.metric_names = ['psnr_train', 'psnr_eval', 'ssim_train', 'ssim_eval']

        # loss functions
        self.criterionI = losses.InvertibilityLoss()
        self.criterionG = losses.GrayscaleConformityLoss(img_shape=(opt.batch_size, 3, opt.crop_size, opt.crop_size), gpu_ids=opt.gpu_ids)

        # model names
        self.model_names = ['E', 'D']

        self.netE = networks.define_E(opt.n_feat, opt.init_type, opt.init_gain, opt.gpu_ids, opt.norm_type)
        self.netD = networks.define_D(opt.n_feat, opt.init_type, opt.init_gain, opt.gpu_ids, opt.norm_type)

        # optimizer
        self.optimizer = torch.optim.Adam(list(self.netE.parameters())+list(self.netD.parameters()), lr=opt.lr, betas=opt.betas)
        self.optimizers = []
        self.optimizers.append(self.optimizer)

        # eval dataloader
        self.eval_dataloader = create_eval_dataloader(self.opt)

    def set_input(self, input):
        self.real_rgb = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        #self.real_lum = input['B'].to(self.device)

    def forward(self):
        self.decolor_gray = self.netE(self.real_rgb)
        self.rec_rgb = self.netD(self.decolor_gray)

    def backward(self):
        self.loss_invertibility = self.criterionI(self.real_rgb, self.rec_rgb)
        #self.loss_grayscale_conformity = self.criterionG(self.decolor_gray, self.real_rgb, self.real_lum)
        self.loss_grayscale_conformity = self.criterionG(self.decolor_gray, self.real_rgb)
        self.loss_total = self.loss_invertibility * 3 + self.loss_grayscale_conformity
        self.metric_psnr_train = util.compute_psnr(self.rec_rgb, self.real_rgb)
        self.metric_ssim_train = util.compute_ssim(self.rec_rgb, self.real_rgb)
        self.loss_total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def load_networks(self, verbose=True):
        for name in self.model_names:
            net = getattr(self, 'net' + name, None)
            path = getattr(self.opt, 'restore_%s_path' % name, None)
            if path is not None:
                util.load_network(net, path, verbose)

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if isinstance(net, torch.nn.DataParallel):
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def test(self):
        netE = getattr(self, 'netE')
        netD = getattr(self, 'netD')
        with torch.no_grad():
            self.decolor_gray = netE(self.real_rgb)
            self.rec_rgb = netD(self.decolor_gray)
            self.psnr = util.compute_psnr(self.rec_rgb, self.real_rgb)
            self.ssim = util.compute_ssim(self.rec_rgb, self.real_rgb)

    def evaluate_model(self, step):

        save_dir = os.path.join(self.opt.log_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netE.eval()
        self.netD.eval()

        fakes, names = [], []
        batch_id = 0
        avg_psnr = 0
        avg_ssim = 0
        eval_dataloader = self.eval_dataloader
        for i, data in enumerate(eval_dataloader):
            # input
            self.set_input(data)
            self.test()
            avg_psnr += self.psnr
            avg_ssim += self.ssim
            batch_id += 1
            fakes.append(self.decolor_gray.cpu())
            for j in range(len(self.image_paths)):
                short_path = os.path.basename(self.image_paths[j])
                name, _ = os.path.splitext(short_path)
                names.append(name)

                real_rgb = util.tensor2im(self.real_rgb[j])
                decolor_gray = util.tensor2im(self.decolor_gray[j])
                util.save_image(real_rgb, os.path.join(save_dir, 'real_rgb', '%s.png' % name), create_dir=True)
                util.save_image(decolor_gray, os.path.join(save_dir, 'decolor_gray', '%s.png' % name), create_dir=True)

        # metric
        self.metric_psnr_eval = avg_psnr / batch_id
        self.metric_ssim_eval = avg_ssim / batch_id
        self.netE.train()
        self.netD.train()

