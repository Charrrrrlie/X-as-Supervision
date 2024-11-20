from torch import nn
import torch
import torch.nn.functional as F

from modules.integral_base_modules.network import get_default_network_config, get_pose_net
class KPDetector3D(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """
    def __init__(self, name, num_kp, depth_dim, num_layers=50):
        super(KPDetector3D, self).__init__()

        cfg = get_default_network_config()
        cfg.depth_dim = depth_dim
        cfg.num_layers = num_layers

        self.num_kp = num_kp
        self.net = get_pose_net(cfg, num_joints=num_kp)
        self.name = name

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
        accu_z = accu_z * torch.arange(float(z_dim)).to(device)

        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        accu_z = accu_z.sum(dim=2, keepdim=True)

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
        kps = torch.cat((x, y, z), dim=2)

        # kps: [B, 1, num_kp, 3] to align multi-hypothesis
        kps = kps.unsqueeze(1)

        return kps, depth_prob_map
