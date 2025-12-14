import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, AdamW, SGD
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import grad as torch_grad
import torch.nn.utils.weight_norm as weightNorm
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import os
from .coordconv import CoordConv2d, CoordConv1d

dim = 32
LAMBDA = 10  # Gradient penalty lambda hyperparameter


class TReLU(nn.Module):
    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim=6, device="cpu"):
        super(Discriminator, self).__init__()
        # self.conv0 = weightNorm(nn.Conv2d(input_dim, 16, 5, 2, 2))
        self.conv0 = weightNorm(CoordConv2d(input_dim, 16, 5, 2, 2, with_r=True, use_cuda=device))
        self.conv1 = weightNorm(nn.Conv2d(16, 32, 5, 2, 2))
        self.conv2 = weightNorm(nn.Conv2d(32, 64, 5, 2, 2))
        self.conv3 = weightNorm(nn.Conv2d(64, 128, 5, 2, 2))
        self.conv4 = weightNorm(nn.Conv2d(128, 1, 5, 2, 2))
        self.relu0 = TReLU()
        self.relu1 = TReLU()
        self.relu2 = TReLU()
        self.relu3 = TReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 1)
        return x


class Wgan:
    def __init__(self, input_dim=3, dataset_inside=False, device="cpu"):
        # self.opts = opts
        self.device = device
        self.dataset_inside = dataset_inside
        self.input_dim = input_dim
        self.net = Discriminator(input_dim, device=self.device).to(self.device)
        # self.optimizerD = AdamW(self.net.parameters(), lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-2)
        self.optimizerD = Adam(self.net.parameters(), lr=3e-4, betas=(0.5, 0.999))

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def cal_gradient_penalty(self, real_data, fake_data, batch_size):
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
        alpha = alpha.view(batch_size, -1, dim, dim)
        alpha = alpha.to(self.device)
        fake_data = fake_data.view(batch_size, -1, dim, dim)
        interpolates = Variable(alpha * real_data.data + ((1 - alpha) * fake_data.data), requires_grad=True)
        disc_interpolates = self.net(interpolates)
        gradients = autograd.grad(disc_interpolates, interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def cal_reward(self, fake_data, real_data):
        if self.input_dim == 6:
            return self.net(torch.cat([real_data, fake_data], 1))
        else:
            return self.net(fake_data)

    def update(self, fake_data, real_data, blur_data=None):

        #fake = fake_data.detach()
        #real = real_data.detach()
        fake = fake_data
        real = real_data
        D_real = self.net(real)
        D_fake = self.net(fake)
        gradient_penalty = self.cal_gradient_penalty(real, fake, real.shape[0])
        self.optimizerD.zero_grad()
        D_cost = D_fake.mean() - D_real.mean() + gradient_penalty
        D_cost.backward(retain_graph=True)
        self.optimizerD.step()
        return D_fake.mean(), D_real.mean(), gradient_penalty
