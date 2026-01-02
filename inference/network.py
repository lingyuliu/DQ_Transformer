import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn import LayerNorm

from transformer import Transformer
from coordconv import CoordConv2d, CoordConv1d


class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


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

    def __init__(self, param_per_stroke, total_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3,
                 device="cpu"):
        super().__init__()
        self.local_encoder_t = nn.Sequential(
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
        )
        self.conv = nn.Conv2d(128 * 3, hidden_dim, 1)

        self.sub_conv1 = nn.Conv1d(128, 256, 1, 1)
        self.sub_conv2 = nn.Conv1d(64, 8, 1, 1)

        self.DQ_transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers, batch_first=True)
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, param_per_stroke))
        self.linear_decider = nn.Linear(hidden_dim, 1)

    def forward(self, img, canvas, cha):
        b, _, H, W = img.shape
        It = self.local_encoder_t(img)
        Ic = self.local_encoder_c(canvas)
        Isub = self.local_encoder_d(abs(cha))
        h, w = 8, 8

        feat = torch.cat([It, Ic, Isub], dim=1)
        feat_conv = self.conv(feat)
        feat_conv = feat_conv.flatten(2).permute(0, 2, 1).contiguous()

        Isub = Isub.flatten(2)
        Isub = self.sub_conv1(Isub)
        Isub = Isub.permute(0, 2, 1)
        Isub = self.sub_conv2(Isub)

        kv = feat_conv
        hidden_state = self.DQ_transformer(kv, Isub.contiguous())
        param = self.linear_param(hidden_state)
        decision = self.linear_decider(hidden_state)
        return param, decision
