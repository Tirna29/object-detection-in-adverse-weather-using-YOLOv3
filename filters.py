from util_filters import  rgb2lum, tanh_range, lerp
import math
import torch
import torch.nn.functional as F

# Code adapted from https://github.com/noefford/Image-Adaptive-YOLO-pytorch

class Filter:

    def __init__(self, net, cfg):
        self.cfg = cfg
        self.num_filter_parameters = None
        self.short_name = None
        self.filter_parameters = None

    def get_short_name(self):
        assert self.short_name
        return self.short_name

    def get_num_filter_parameters(self):
        assert self.num_filter_parameters
        return self.num_filter_parameters

    def get_begin_filter_parameter(self):
        return self.begin_filter_parameter

    def extract_parameters(self, features):
        return features[:, self.get_begin_filter_parameter():(
                    self.get_begin_filter_parameter() + self.get_num_filter_parameters())], \
            features[:,
            self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())]

    def filter_param_regressor(self, features):
        assert False

    def process(self, img, param, defog, IcA):
        assert False

    def apply(self,
              img,
              img_features=None,
              defog_A=None,
              IcA=None,
              specified_parameter=None,
              high_res=None,
              net=None):
        assert (img_features is None) ^ (specified_parameter is None)
        if img_features is not None:
            filter_features, mask_parameters = self.extract_parameters(img_features)  # 获取参数
            filter_parameters = self.filter_param_regressor(filter_features)
        else:
            raise NotImplementedError
        low_res_output = self.process(img, filter_parameters, defog_A, IcA)
        return low_res_output, filter_parameters


class UsmFilter(Filter):

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param, defog_A, IcA):
        def make_gaussian_2d_kernel(sigma):
            radius = 12
            x = torch.arange(-radius, radius + 1)
            k = torch.exp(-0.5 * torch.square(x / sigma))
            k = k / torch.sum(k)
            return k.unsqueeze(1) * k

        kernel_i = make_gaussian_2d_kernel(5)
        kernel_i = torch.tensor(kernel_i).to(img.device)

        kernel_i = kernel_i.unsqueeze(0)
        kernel_i = kernel_i.unsqueeze(0)

        kernel_i = kernel_i.repeat([3, 1, 1, 1])

        output = F.conv2d(img, weight=kernel_i, stride=1, groups=3, padding=12)
        img_out = (img - output) * param[:, None, None, :] + img


        return img_out


class DefogFilter(Filter):

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'DF'
        self.begin_filter_parameter = cfg.defog_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.defog_range)(features)

    def process(self, img, param, defog_A, IcA):
        tx = 1 - param[:, None, None, :] * IcA
        tx_1 = tx.repeat([1, 1, 1, 3]).permute(0, 3, 1, 2)
        return (img - defog_A[:, :, None, None]) / torch.max(tx_1.clamp(min=0.01), torch.tensor(0.01)) + defog_A[:, :, None, None]


class GammaFilter(Filter):

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        gamma_range = torch.tensor(self.cfg.gamma_range)
        log_gamma_range = torch.log(gamma_range).to(features.device)
        return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

    def process(self, img, param, defog_A, IcA):
        param_1 = param.repeat([1, 3])
        return torch.pow(torch.max(img, torch.tensor(0.0001)), param_1[:, :, None, None])


class ImprovedWhiteBalanceFilter(Filter):

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'W'
        self.channels = 3
        self.begin_filter_parameter = cfg.wb_begin_param
        self.num_filter_parameters = self.channels

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor(((0, 1, 1))).to(features.device)
        mask = mask.unsqueeze(0)

        features = features * mask
        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))

        color_scaling = color_scaling * 1.0 / (
                                                      1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
                                                      0.06 * color_scaling[:, 2])[:, None]
        return color_scaling

    def process(self, img, param, defog, IcA):
        return img * param[:, :, None, None]




class ToneFilter(Filter):

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param

        self.num_filter_parameters = cfg.curve_steps

    def filter_param_regressor(self, features):
        tone_curve = torch.reshape(
            features, shape=(-1, 1, self.cfg.curve_steps))[:, None, None, :]
        tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
        return tone_curve

    def process(self, img, param, defog, IcA):

        tone_curve = param
        tone_curve_sum = torch.sum(tone_curve, dim=-1) + 1e-30
        total_image = img * 0
        curve_steps = torch.tensor(self.cfg.curve_steps).to(img.device)
        for i in range(self.cfg.curve_steps):
            total_image = total_image + torch.clip(img - 1.0 * i / curve_steps, 0, 1.0 / curve_steps) \
                          * param[:, :, :, :, i]

        total_image = total_image * curve_steps / tone_curve_sum
        img = total_image
        return img


class ContrastFilter(Filter):

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return torch.tanh(features)

    def process(self, img, param, defog, IcA):
        shelod1 = torch.tensor(0.0)
        shelod2 = torch.tensor(1.0)
        luminance = torch.min(torch.max(rgb2lum(img), shelod1), shelod2)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, param[:, :, None, None])

