import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import functools

from .padding_same_conv import Conv2d

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.nepochs) / float(opt.nepochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler  = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    return scheduler

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('\u2755 Normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        # conv
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('\u2755 Initialization method [%s] is not implemented.' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # batchnorm affine param?
        elif classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.weight is not None:
                init.constant_(m.bias.data, 0.0)
    print('\u270f Initialize network [%s] with [%s].' % (net.__class__.__name__, init_type))
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
    if len(gpu_ids) > 1:
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain)
    return net

def define_E(n_feat, init_type, init_gain, gpu_ids, norm='batch'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Encoder(n_feat, norm_layer)
    return init_net(net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)

def define_D(n_feat, init_type, init_gain, gpu_ids, norm='batch'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Decoder(n_feat, norm_layer)
    return init_net(net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)

def define_U(input_nc, output_nc, ngf, netG, norm='batch', dropout_rate=0,
             init_type='normal', init_gain=0.02, gpu_ids=[], opt=None):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        from nir2graynets.modules.resnet_architecture.resnet_generator import ResnetGenerator
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                              dropout_rate=dropout_rate, n_blocks=9)
    return init_net(net, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)


class Identity():
    def forward(self, x):
        return x

class Encoder(nn.Module):
    def __init__(self, n_feat, norm_layer=nn.BatchNorm2d):
        super().__init__()
        input_ch = 3
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.in_layers = nn.Sequential(
            Conv2d(input_ch, n_feat, kernel_size=3, stride=1),
            norm_layer(n_feat),
            nn.ReLU(True),
            ResidualBlock(n_feat),
            ResidualBlock(n_feat)
        )

        self.down_layers_1 = nn.Sequential(
            Conv2d(n_feat, n_feat*2, kernel_size=3, stride=2),
            Conv2d(n_feat*2, n_feat*2, kernel_size=3, stride=1),
            norm_layer(n_feat*2),
            nn.ReLU(True)
        )
        self.down_layers_2 = nn.Sequential(
            Conv2d(n_feat*2, n_feat*4, kernel_size=3, stride=2),
            Conv2d(n_feat*4, n_feat*4, kernel_size=3, stride=1),
            norm_layer(n_feat*4),
            nn.ReLU(True)
        )

        mid_layers = []
        for i in range(4):
            mid_layers += [ResidualBlock(n_feat*4)]
        self.mid_layers = nn.Sequential(*mid_layers)

        self.up_layers_1 = upConv(n_feat*4, n_feat*2)
        self.up_layers_2 = upConv(n_feat*2, n_feat)

        final_res = []
        for i in range(2):
            final_res += [ResidualBlock(n_feat)]
        self.final_res = nn.Sequential(*final_res)

        self.final_out = nn.Sequential(
            Conv2d(n_feat, 1, kernel_size=3, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x_in = self.in_layers(x)
        x_down1 = self.down_layers_1(x_in)
        x_down2 = self.down_layers_2(x_down1)
        x_mid = self.mid_layers(x_down2)
        x_up1 = self.up_layers_1(x_mid)
        x_up2 = self.up_layers_2(x_down1 + x_up1)
        x_res = self.final_res(x_in + x_up2)
        x_out = self.final_out(x_res)

        return x_out

class Decoder(nn.Module):
    def __init__(self, n_feat, norm_layer=nn.BatchNorm2d):
        super().__init__()
        layers = [Conv2d(1, n_feat, kernel_size=3, stride=1)]
        for i in range(8):
            layers += [ResidualBlock(n_feat, norm_layer)]
        layers += [Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1),   # official code is different from the structure in the paper
                Conv2d(n_feat*4, 3, kernel_size=1, stride=1),
                nn.Tanh()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ResidualBlock(nn.Module):
    def __init__(self, n_feat, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2d(n_feat, n_feat, kernel_size=3, stride=1),
            norm_layer(n_feat),
            nn.ReLU(True),
            Conv2d(n_feat, n_feat, kernel_size=3, stride=1),
            norm_layer(n_feat)
        )

    def forward(self, x):
        out = self.layers(x)
        return x + out

class upConv(nn.Module):
    def __init__(self, n_in, n_feat, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            Conv2d(n_in, n_feat, kernel_size=3, stride=1),
            Conv2d(n_feat, n_feat, kernel_size=3, stride=1),
            norm_layer(n_feat),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layers(x)