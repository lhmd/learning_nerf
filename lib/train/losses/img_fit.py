import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg

class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))# 将 MSE 损失转换为 PSNR（峰值信噪比）

    def forward(self, batch):
        output = self.net(batch)

        scalar_stats = {}
        loss = 0
        color_loss = self.color_crit(output['rgb'], batch['rgb'])# 计算颜色损失，这里使用了之前定义的 MSE 损失函数 self.color_crit，将模型输出的颜色值 output['rgb'] 与批次中的真实颜色值 batch['rgb'] 进行比较
        scalar_stats.update({'color_mse': color_loss})
        loss += color_loss

        psnr = -10. * torch.log(color_loss.detach()) / \
                torch.log(torch.Tensor([10.]).to(color_loss.device))
        scalar_stats.update({'psnr': psnr})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
