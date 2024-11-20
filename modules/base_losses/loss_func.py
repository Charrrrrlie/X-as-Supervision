import torch.nn as nn
import torch

def compute_mask_reconstruction_loss(mask, gt, weight=None, use_clip=False):
    mse_loss = nn.MSELoss(reduction='mean' if weight is None else 'none')
    loss = mse_loss(mask, gt)

    if use_clip:
        filter = (mask > 0.1).float()
        loss = loss * filter

    if weight is not None:
        loss = loss * weight
        loss = loss.mean()

    return loss

def compute_bone_sym_loss(keypoints):
    # constrain arms and legs to be the same length
    bone = keypoints[:, [16, 15, 13, 12, 3, 2, 6, 5], :] - keypoints[:, [15, 14, 12, 11, 2, 1, 5, 4], :]
    bone = torch.norm(bone, dim=2) * 1e-3

    mse_loss = nn.MSELoss(reduction='mean')

    return mse_loss(bone[:, [0, 2, 4, 6]], bone[:, [1, 3, 5, 7]])

def compute_kp_sym_loss(keypoints, is_3D=True):
    center = (keypoints[:, [11, 1], :] + keypoints[:, [14, 4], :]) / 2

    mse_loss = nn.MSELoss(reduction='mean')

    if is_3D:
        return mse_loss(center * 1e-3, keypoints[:, [-1, 0], :] * 1e-3)
    else:
        return mse_loss(center, keypoints[:, [-1, 0], :])


def compute_supervision(keypoint, keypoint_gt, feature_shape=None, mode='mean'):
    if feature_shape is not None:
        keypoint = keypoint.clone()
        keypoint[:, :, :2] = (keypoint[:, :, :2] + 1) / 2.0
        keypoint[:, :, 0] = keypoint[:, :, 0] * (feature_shape[0] - 1)
        keypoint[:, :, 1] = keypoint[:, :, 1] * (feature_shape[1] - 1)
        if keypoint.shape[-1] == 3:
            keypoint[:, :, 2] = keypoint[:, :, 2] * (feature_shape[2] - 1)

    mse_loss = nn.MSELoss(reduction=mode)
    loss = mse_loss(keypoint, keypoint_gt)

    if mode == 'sum':
        loss = loss / keypoint.shape[0]
    return loss

def compute_disc_loss(pred_logits, gt_logits):
    if gt_logits is None:
        if pred_logits.dim() == 2:
            return (pred_logits - 1).pow(2).mean()
        elif pred_logits.dim() == 3:
            return (pred_logits - 1).pow(2).min(dim=1)[0].mean()
        else:
            raise ValueError('Invalid dimension of pred_logits')
    loss = 0
    if gt_logits.dim() == 2:
        loss += 0.5 * (gt_logits - 1).pow(2).mean()
    elif gt_logits.dim() == 3:
        loss += 0.5 * (gt_logits - 1).pow(2).min(dim=1)[0].mean()
    else:
        raise ValueError('Invalid dimension of gt_logits')

    if pred_logits.dim() == 2:
        loss += 0.5 * pred_logits.pow(2).mean()
    elif pred_logits.dim() == 3:
        loss += 0.5 * pred_logits.pow(2).min(dim=1)[0].mean()
    else:
        raise ValueError('Invalid dimension of pred_logits')
    return loss