import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn import LayerNorm
from .coordconv import CoordConv2d, CoordConv1d
from .transformer import Transformer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            lr_l = 0.3 ** max(0, (epoch + opt.epoch_count - opt.n_epochs) // 5)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
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
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

    # for name, parm in net.named_parameters():
    #     if "linear_param" in name:
    #         parm.data *= 0.5


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class SignWithSigmoidGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        result = (x > 0).float()
        sigmoid_result = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_result)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        (sigmoid_result,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input


class Painter(nn.Module):

    def __init__(self, param_per_stroke, total_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3, device="cpu"):
        super().__init__()
        self.local_encoder_t = nn.Sequential(
            nn.ReflectionPad2d(1),
            # nn.Conv2d(3, 32, 3, 1),
            CoordConv2d(3, 32, 3, 1, with_r=True, use_cuda=device),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.local_encoder_c = nn.Sequential(
            nn.ReflectionPad2d(1),
            CoordConv2d(3, 32, 3, 1, with_r=True, use_cuda=device),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True))

        self.local_encoder_d = nn.Sequential(
            nn.ReflectionPad2d(1),
            CoordConv2d(3, 32, 3, 1, with_r=True, use_cuda=device),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(128, 256, 3, 2),
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
        )
        self.conv = nn.Conv2d(128 * 3, hidden_dim, 1)

        self.sub_conv1 = nn.Conv1d(128, 256, 1, 1)
        self.sub_conv2 = nn.Conv1d(64, 8, 1, 1)

        # self.sub_norm = LayerNorm(256, eps=1e-5)

        self.DQ_transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers, batch_first=True)  # , norm_first=True
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, param_per_stroke))
        self.linear_decider = nn.Linear(hidden_dim, 1)


    def forward(self, img, canvas, cha):
        b, _, H, W = img.shape
        It = self.local_encoder_t(img)  # [64, 128, 8, 8]
        Ic = self.local_encoder_c(canvas)
        Isub = self.local_encoder_d(abs(cha))
        h, w = 8, 8

        feat = torch.cat([It, Ic, Isub], dim=1)
        feat_conv = self.conv(feat)  # [64, 128*3, 8, 8]
        # feat_conv = feat
        feat_conv = feat_conv.flatten(2).permute(0, 2, 1).contiguous()

        Isub = Isub.flatten(2)  # [64,128,64]
        Isub = self.sub_conv1(Isub)  # [64,256,64]
        Isub = Isub.permute(0, 2, 1)  # [64,64,256]
        Isub = self.sub_conv2(Isub)  # [64,8,256]

        kv = feat_conv
        hidden_state = self.DQ_transformer(kv, Isub.contiguous())
        # hidden_state = hidden_state.permute(1, 0, 2).contiguous()  # [64, 8, 256]
        param = self.linear_param(hidden_state)  # [64, 8, 5]
        decision = self.linear_decider(hidden_state)

        s = hidden_state.shape[1]
        grid = param[:, :, :2].view(b * s, 1, 1, 2).contiguous()
        img_temp = img.unsqueeze(1).contiguous().repeat(1, s, 1, 1, 1).view(b * s, 3, H, W).contiguous()
        color = nn.functional.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(b, s, 3).contiguous()

        # return torch.cat([param, color, color, torch.rand(b, s, 1, device=img.device)], dim=-1), decision
        return torch.cat([param, color, color, torch.ones(b, s, 1, device=img.device)], dim=-1), decision
