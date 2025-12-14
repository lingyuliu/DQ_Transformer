import torch
import numpy as np
from .base_model import BaseModel
from . import networks
from util import morphology
from scipy.optimize import linear_sum_assignment
from PIL import Image
from . import wgan
from torch.autograd import Variable


def shuffle_dim(x: torch.Tensor, dim: int):
    length = x.shape[dim]
    shuffled_indices = torch.randperm(length)

    # 使用 torch.index_select 打乱维度
    return torch.index_select(x, dim, shuffled_indices)


class PainterModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='null')
        parser.add_argument('--used_strokes', type=int, default=8,
                            help='actually generated strokes number')
        parser.add_argument('--num_blocks', type=int, default=3,
                            help='number of transformer blocks for stroke generator')
        parser.add_argument('--lambda_w', type=float, default=10.0, help='weight for w loss of stroke shape')
        parser.add_argument('--lambda_pixel', type=float, default=8.0, help='weight for pixel-level L1 loss')
        parser.add_argument('--lambda_gt', type=float, default=1.0, help='weight for ground-truth loss')
        parser.add_argument('--lambda_decision', type=float, default=1.0, help='weight for stroke decision loss')
        parser.add_argument('--lambda_recall', type=float, default=10.0, help='weight of recall for stroke decision loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['pixel', 'gt', 'w', 'decision', "decision_sum", "gan", "D_fake", "D_real", "G", "D"]
        self.visual_names = ['old', 'render', 'rec']
        self.model_names = ['g']
        self.d = 12  # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
        self.d_shape = 5

        def read_img(img_path, img_type='RGB'):
            img = Image.open(img_path).convert(img_type)
            img = np.array(img)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.
            return img

        brush_large_vertical = read_img('brush/brush_large_vertical.png', 'L').to(self.device)
        brush_large_horizontal = read_img('brush/brush_large_horizontal.png', 'L').to(self.device)
        self.meta_brushes = torch.cat(
            [brush_large_vertical, brush_large_horizontal], dim=0)
        net_g = networks.Painter(self.d_shape, opt.used_strokes, opt.ngf,
                                 n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks, device=self.device)
        self.net_g = networks.init_net(net_g, opt.init_type, opt.init_gain, self.gpu_ids)

        self.old = None
        self.render = None
        self.rec = None
        self.gt_param = None
        self.pred_param = None
        self.gt_decision = None
        self.pred_decision = None
        self.patch_size = 32
        self.loss_pixel = torch.tensor(0., device=self.device)
        self.loss_gt = torch.tensor(0., device=self.device)
        self.loss_w = torch.tensor(0., device=self.device)
        self.loss_decision = torch.tensor(0., device=self.device)
        self.loss_decision_sum = torch.tensor(0., device=self.device)
        self.loss_D_fake = torch.tensor(0., device=self.device)
        self.loss_D_real = torch.tensor(0., device=self.device)
        self.loss_G = torch.tensor(0., device=self.device)
        self.loss_D = torch.tensor(0., device=self.device)

        self.criterion_pixel = torch.nn.L1Loss().to(self.device)
        self.criterion_decision = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(opt.lambda_recall)).to(self.device)
        if self.isTrain:
            self.optimizer = torch.optim.AdamW(self.net_g.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=1e-2)
            self.optimizers.append(self.optimizer)

        self.cha_map = None

        self.critic = wgan.Wgan(input_dim=3, dataset_inside=False, device=self.device)
        self.loss_gan = torch.tensor(0., device=self.device)

    def param2stroke(self, param, H, W):
        # param: b, 12
        b = param.shape[0]
        param_list = torch.split(param, 1, dim=1)
        x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
        R0, G0, B0, R2, G2, B2, _ = param_list[5:]
        sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        index = torch.full((b,), -1, device=param.device)
        index[h > w] = 0
        index[h <= w] = 1
        brush = self.meta_brushes[index.long()]
        alphas = torch.cat([brush, brush, brush], dim=1)
        alphas = (alphas > 0).float()
        t = torch.arange(0, brush.shape[2], device=param.device).unsqueeze(0) / brush.shape[2]
        color_map = torch.stack([R0 * (1 - t) + R2 * t, G0 * (1 - t) + G2 * t, B0 * (1 - t) + B2 * t], dim=1)
        color_map = color_map.unsqueeze(-1).repeat(1, 1, 1, brush.shape[3])
        brush = brush * color_map

        warp_00 = cos_theta / w
        warp_01 = sin_theta * H / (W * w)
        warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
        warp_10 = -sin_theta * W / (H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        warp = torch.stack([warp_0, warp_1], dim=1)
        grid = torch.nn.functional.affine_grid(warp, torch.Size((b, 3, H, W)), align_corners=False)
        brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
        alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)

        return brush, alphas

    def set_input(self, input_dict):
        self.image_paths = input_dict['A_paths']
        with torch.no_grad():
            old_param = torch.rand(self.opt.batch_size // 4, self.opt.used_strokes, self.d, device=self.device)
            old_param[:, :, :4] = old_param[:, :, :4] * 0.5 + 0.2
            old_param[:, :, -4:-1] = old_param[:, :, -7:-4]
            old_param = old_param.view(-1, self.d).contiguous()
            foregrounds, alphas = self.param2stroke(old_param, self.patch_size * 2, self.patch_size * 2)
            foregrounds = morphology.Dilation2d(m=1)(foregrounds)
            alphas = morphology.Erosion2d(m=1)(alphas)
            foregrounds = foregrounds.view(self.opt.batch_size // 4, self.opt.used_strokes, 3, self.patch_size * 2,
                                           self.patch_size * 2).contiguous()
            alphas = alphas.view(self.opt.batch_size // 4, self.opt.used_strokes, 3, self.patch_size * 2,
                                 self.patch_size * 2).contiguous()
            old = torch.zeros(self.opt.batch_size // 4, 3, self.patch_size * 2, self.patch_size * 2, device=self.device)
            for i in range(self.opt.used_strokes):
                foreground = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                old = foreground * alpha + old * (1 - alpha)
            old = old.view(self.opt.batch_size // 4, 3, 2, self.patch_size, 2, self.patch_size).contiguous()
            old = old.permute(0, 2, 4, 1, 3, 5).contiguous()
            self.old = old.view(self.opt.batch_size, 3, self.patch_size, self.patch_size).contiguous()

            gt_param = torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device)
            gt_param[:, :, :4] = gt_param[:, :, :4] * 0.5 + 0.2
            gt_param[:, :, -4:-1] = gt_param[:, :, -7:-4]
            self.gt_param = gt_param[:, :, :self.d_shape]
            gt_param = gt_param.view(-1, self.d).contiguous()
            foregrounds, alphas = self.param2stroke(gt_param, self.patch_size, self.patch_size)
            foregrounds = morphology.Dilation2d(m=1)(foregrounds)
            alphas = morphology.Erosion2d(m=1)(alphas)
            foregrounds = foregrounds.view(self.opt.batch_size, self.opt.used_strokes, 3, self.patch_size,
                                           self.patch_size).contiguous()
            alphas = alphas.view(self.opt.batch_size, self.opt.used_strokes, 3, self.patch_size,
                                 self.patch_size).contiguous()
            self.render = self.old.clone()

            gt_decision = torch.ones(self.opt.batch_size, self.opt.used_strokes, device=self.device)
            for i in range(self.opt.used_strokes):
                foreground = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                for j in range(i):
                    iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                            torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                    gt_decision[:, i] = ((iou < 0.75) | (~gt_decision[:, j].bool())).float() * gt_decision[:, i]
                decision = gt_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
                self.render = foreground * alpha * decision + self.render * (1 - alpha * decision)
            self.gt_decision = gt_decision

            self.cha_map = self.render - self.old

    def forward(self):
        param, decisions = self.net_g(self.render, self.old, self.cha_map)
        self.pred_decision = decisions.view(-1, self.opt.used_strokes).contiguous()
        self.pred_param = param[:, :, :self.d_shape]
        param = param.view(-1, self.d).contiguous()
        foregrounds, alphas = self.param2stroke(param, self.patch_size, self.patch_size)
        foregrounds = morphology.Dilation2d(m=1)(foregrounds)
        alphas = morphology.Erosion2d(m=1)(alphas)
        foregrounds = foregrounds.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        alphas = alphas.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        decisions = networks.SignWithSigmoidGrad.apply(decisions.view(-1, self.opt.used_strokes, 1, 1, 1).contiguous())
        self.rec = self.old.clone()
        for j in range(foregrounds.shape[1]):
            foreground = foregrounds[:, j, :, :, :]
            alpha = alphas[:, j, :, :, :]
            decision = decisions[:, j, :, :, :]
            self.rec = foreground * alpha * decision + self.rec * (1 - alpha * decision)

    @staticmethod
    def get_sigma_sqrt(w, h, theta):
        sigma_00 = w * (torch.cos(theta) ** 2) / 2 + h * (torch.sin(theta) ** 2) / 2
        sigma_01 = (w - h) * torch.cos(theta) * torch.sin(theta) / 2
        sigma_11 = h * (torch.cos(theta) ** 2) / 2 + w * (torch.sin(theta) ** 2) / 2
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    @staticmethod
    def get_sigma(w, h, theta):
        sigma_00 = w * w * (torch.cos(theta) ** 2) / 4 + h * h * (torch.sin(theta) ** 2) / 4
        sigma_01 = (w * w - h * h) * torch.cos(theta) * torch.sin(theta) / 4
        sigma_11 = h * h * (torch.cos(theta) ** 2) / 4 + w * w * (torch.sin(theta) ** 2) / 4
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    def gaussian_w_distance(self, param_1, param_2):
        mu_1, w_1, h_1, theta_1 = torch.split(param_1, (2, 1, 1, 1), dim=-1)
        w_1 = w_1.squeeze(-1)
        h_1 = h_1.squeeze(-1)
        theta_1 = torch.acos(torch.tensor(-1., device=param_1.device)) * theta_1.squeeze(-1)
        trace_1 = (w_1 ** 2 + h_1 ** 2) / 4
        mu_2, w_2, h_2, theta_2 = torch.split(param_2, (2, 1, 1, 1), dim=-1)
        w_2 = w_2.squeeze(-1)
        h_2 = h_2.squeeze(-1)
        theta_2 = torch.acos(torch.tensor(-1., device=param_2.device)) * theta_2.squeeze(-1)
        trace_2 = (w_2 ** 2 + h_2 ** 2) / 4
        sigma_1_sqrt = self.get_sigma_sqrt(w_1, h_1, theta_1)
        sigma_2 = self.get_sigma(w_2, h_2, theta_2)
        trace_12 = torch.matmul(torch.matmul(sigma_1_sqrt, sigma_2), sigma_1_sqrt)
        trace_12 = torch.sqrt(trace_12[..., 0, 0] + trace_12[..., 1, 1] + 2 * torch.sqrt(
            trace_12[..., 0, 0] * trace_12[..., 1, 1] - trace_12[..., 0, 1] * trace_12[..., 1, 0]))
        return torch.sum((mu_1 - mu_2) ** 2, dim=-1) + trace_1 + trace_2 - 2 * trace_12

    def optimize_parameters(self, epoch):
        self.forward()
        self.loss_pixel = self.criterion_pixel(self.rec, self.render)
        self.loss_decision_sum = torch.norm(self.pred_decision, p=1, dim=1).sum() / self.opt.batch_size * 0.1
        cur_valid_gt_size = 0
        with torch.no_grad():
            r_idx = []
            c_idx = []
            for i in range(self.gt_param.shape[0]):
                is_valid_gt = self.gt_decision[i].bool()
                valid_gt_param = self.gt_param[i, is_valid_gt]
                cost_matrix_l1 = torch.cdist(self.pred_param[i].to(torch.float32), valid_gt_param.to(torch.float32), p=1)
                pred_param_broad = self.pred_param[i].unsqueeze(1).contiguous().repeat(
                    1, valid_gt_param.shape[0], 1)
                valid_gt_param_broad = valid_gt_param.unsqueeze(0).contiguous().repeat(
                    self.pred_param.shape[1], 1, 1)
                cost_matrix_w = self.gaussian_w_distance(pred_param_broad.to(torch.float32), valid_gt_param_broad.to(torch.float32))
                decision = self.pred_decision[i]
                cost_matrix_decision = (1 - decision).unsqueeze(-1).repeat(1, valid_gt_param.shape[0])
                r, c = linear_sum_assignment((cost_matrix_l1 + cost_matrix_w + cost_matrix_decision).to(torch.float32).cpu())
                r_idx.append(torch.tensor(r + self.pred_param.shape[1] * i, device=self.device))
                c_idx.append(torch.tensor(c + cur_valid_gt_size, device=self.device))
                cur_valid_gt_size += valid_gt_param.shape[0]
            r_idx = torch.cat(r_idx, dim=0)
            c_idx = torch.cat(c_idx, dim=0)
            paired_gt_decision = torch.zeros(self.gt_decision.shape[0] * self.gt_decision.shape[1], device=self.device)
            paired_gt_decision[r_idx] = 1.
        all_valid_gt_param = self.gt_param[self.gt_decision.bool(), :]
        all_pred_param = self.pred_param.view(-1, self.pred_param.shape[2]).contiguous()
        all_pred_decision = self.pred_decision.view(-1).contiguous()
        paired_gt_param = all_valid_gt_param[c_idx, :]
        paired_pred_param = all_pred_param[r_idx, :]
        self.loss_gt = self.criterion_pixel(paired_pred_param, paired_gt_param)
        self.loss_w = self.gaussian_w_distance(paired_pred_param, paired_gt_param).mean()
        self.loss_decision = self.criterion_decision(all_pred_decision, paired_gt_decision)
        loss = (self.loss_pixel * self.opt.lambda_pixel + self.loss_gt * self.opt.lambda_gt + self.loss_w * self.opt.lambda_w
                + self.loss_decision * self.opt.lambda_decision + self.loss_decision_sum)

        if epoch > 200:
            D_fake, D_real, gradient_penalty = self.critic.update(Variable(self.rec), Variable(self.render))
            self.loss_D_fake = round(float(D_fake.detach()), 4)
            self.loss_D_real = round(float(D_real.detach()), 4)

            gan_loss = D_fake - D_real + gradient_penalty
            lambda_gan = float((self.loss_pixel * self.opt.lambda_pixel).detach())
            self.loss_gan = gan_loss * lambda_gan

            self.loss_D = D_fake - D_real + gradient_penalty
            loss += self.loss_gan

        self.loss_G = loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    # def test_one_iter(self):
    #     self.set_input()
