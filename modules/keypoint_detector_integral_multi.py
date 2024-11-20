from torch import nn
import torch
import torch.nn.functional as F

from modules.integral_base_modules.network import get_default_network_config, get_pose_net
class KPDetector3DMulti(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """
    def __init__(self, name, num_kp, depth_dim, num_hypo, neighbor_size, num_layers=50):
        super(KPDetector3DMulti, self).__init__()

        cfg = get_default_network_config()
        cfg.depth_dim = depth_dim
        cfg.num_layers = num_layers

        self.num_hypo = num_hypo
        self.neighbor_size = neighbor_size

        self.num_kp = num_kp
        self.net = get_pose_net(cfg, num_joints=num_kp)
        self.name = name

    def find_peak(self, heatmap):
        # thred = 1 / (self.num_hypo * 3)
        # peaks in [B, N, D - 2]
        peaks = (heatmap[..., 1:-1] >= heatmap[..., :-2]) & (heatmap[..., 1:-1] >= heatmap[..., 2:])

        peaks = peaks.float()
        peaks = peaks * heatmap[..., 1:-1]
        peaks = peaks.view(heatmap.size(0), heatmap.size(1), -1)
        peaks, indices = torch.topk(peaks, self.num_hypo, dim=-1)
        indices = indices + 1 # keep the original index
        return indices

    def generate_3d_integral_preds_tensor(self, heatmaps, x_dim, y_dim, z_dim):
        assert isinstance(heatmaps, torch.Tensor)

        accu_x = heatmaps.sum(dim=2)
        accu_x = accu_x.sum(dim=2)
        accu_y = heatmaps.sum(dim=2)
        accu_y = accu_y.sum(dim=3)
        accu_z = heatmaps.sum(dim=3)
        accu_z = accu_z.sum(dim=3)

        device = heatmaps.device
        # for multi hypothesis vis
        depth_prob_map = accu_z[0].clone()

        accu_x = accu_x * torch.arange(float(x_dim)).to(device)
        accu_y = accu_y * torch.arange(float(y_dim)).to(device)

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        
        z_peak_indices = self.find_peak(accu_z)
        z_weighted_heatmap = accu_z * torch.arange(float(z_dim)).to(device)

        z_sum_heatmap = F.avg_pool1d(z_weighted_heatmap, kernel_size=self.neighbor_size, stride=1, padding=self.neighbor_size // 2)
        z_sum_weight = F.avg_pool1d(accu_z, kernel_size=self.neighbor_size, stride=1, padding=self.neighbor_size // 2)

        accu_z = torch.gather(z_sum_heatmap, -1, z_peak_indices) / torch.gather(z_sum_weight, -1, z_peak_indices)

        return accu_x, accu_y, accu_z, depth_prob_map

    def forward(self, x):
        heatmap = self.net(x)

        B, C, H, W = heatmap.shape
        heatmap = heatmap.view(B, self.num_kp, -1)
        heatmap = F.softmax(heatmap, 2)

        D = C // self.num_kp
        heatmap = heatmap.view(B, self.num_kp, D, H, W)

        x, y, z, depth_prob_map = self.generate_3d_integral_preds_tensor(heatmap, D, H, W)

        x = x / H * 2 - 1
        y = y / W * 2 - 1
        z = z / D * 2 - 1

        x = x.unsqueeze(1).repeat(1, self.num_hypo, 1, 1)
        y = y.unsqueeze(1).repeat(1, self.num_hypo, 1, 1)
        z = z.permute(0, 2, 1).unsqueeze(-1)

        kps = torch.cat((x, y, z), dim=-1)

        return kps, depth_prob_map
