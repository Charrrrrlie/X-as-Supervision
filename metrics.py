# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

def compute_similarity_transform(source_points, target_points):
    """Computes a similarity transform (sR, t) that takes a set of 3D points
    source_points (N x 3) closest to a set of 3D points target_points, where R
    is an 3x3 rotation matrix, t 3x1 translation, s scale. And return the
    transformed 3D points source_points_hat (N x 3). i.e. solves the orthogonal
    Procrutes problem.

    Note:
        Points number: N

    Args:
        source_points (np.ndarray): Source point set with shape [N, 3].
        target_points (np.ndarray): Target point set with shape [N, 3].

    Returns:
        np.ndarray: Transformed source point set with shape [N, 3].
    """

    assert target_points.shape[0] == source_points.shape[0]
    assert target_points.shape[1] == 3 and source_points.shape[1] == 3

    source_points = source_points.T
    target_points = target_points.T

    # 1. Remove mean.
    mu1 = source_points.mean(axis=1, keepdims=True)
    mu2 = target_points.mean(axis=1, keepdims=True)
    X1 = source_points - mu1
    X2 = target_points - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, _, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Transform the source points:
    source_points_hat = scale * R.dot(source_points) + t

    source_points_hat = source_points_hat.T

    return source_points_hat


def keypoint_mpjpe(pred, gt, mask, alignment='none'):
    """Calculate the mean per-joint position error (MPJPE) and the error after
    rigid alignment with the ground truth (P-MPJPE).

    Note:
        - batch_size: N
        - num_keypoints: K
        - keypoint_dims: C

    Args:
        pred (np.ndarray): Predicted keypoint location with shape [N, K, C].
        gt (np.ndarray): Groundtruth keypoint location with shape [N, K, C].
        mask (np.ndarray): Visibility of the target with shape [N, K].
            False for invisible joints, and True for visible.
            Invisible joints will be ignored for accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:

                - ``'none'``: no alignment will be applied
                - ``'scale'``: align in the least-square sense in scale
                - ``'procrustes'``: align in the least-square sense in
                    scale, rotation and translation.
    Returns:
        tuple: A tuple containing joint position errors

        - (float | np.ndarray): mean per-joint position error (mpjpe).
        - (float | np.ndarray): mpjpe after rigid alignment with the
            ground truth (p-mpjpe).
    """
    assert mask.any()

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1) * mask

    return error


def keypoint_3d_pck(pred, gt, mask, alignment='none', threshold=0.15):
    """Calculate the Percentage of Correct Keypoints (3DPCK) w. or w/o rigid
    alignment.

    Paper ref: `Monocular 3D Human Pose Estimation In The Wild Using Improved
    CNN Supervision' 3DV'2017. <https://arxiv.org/pdf/1611.09813>`__ .

    Note:
        - batch_size: N
        - num_keypoints: K
        - keypoint_dims: C

    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:

            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.

        threshold:  If L2 distance between the prediction and the groundtruth
            is less then threshold, the predicted result is considered as
            correct. Default: 0.15 (m).

    Returns:
        pck: percentage of correct keypoints.
    """
    assert mask.any()

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1)
    pck = (error < threshold).astype(np.float32) * mask * 100

    return pck


def keypoint_3d_auc(pred, gt, mask, alignment='none'):
    """Calculate the Area Under the Curve (3DAUC) computed for a range of 3DPCK
    thresholds.

    Paper ref: `Monocular 3D Human Pose Estimation In The Wild Using Improved
    CNN Supervision' 3DV'2017. <https://arxiv.org/pdf/1611.09813>`__ .
    This implementation is derived from mpii_compute_3d_pck.m, which is
    provided as part of the MPI-INF-3DHP test data release.

    Note:
        batch_size: N
        num_keypoints: K
        keypoint_dims: C

    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:

            - ``'none'``: no alignment will be applied
            - ``'scale'``: align in the least-square sense in scale
            - ``'procrustes'``: align in the least-square sense in scale,
                rotation and translation.

    Returns:
        auc: AUC computed for a range of 3DPCK thresholds.
    """
    assert mask.any()

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')

    error = np.linalg.norm(pred - gt, ord=2, axis=-1)

    thresholds = np.linspace(0., 0.15, 31)
    pck_values = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        pck_values[i] = ((error < thresholds[i]).astype(np.float32) * mask).mean()

    auc = pck_values.mean() * 100

    return auc


def keypoint_pckh(pred, gt, head_size, PCKh_thred=0.5):
    """Calculate the Percentage of Correct Keypoints (PCKh) with head size."""
    error = torch.linalg.norm(pred - gt, ord=2, dim=-1)
    error = error / head_size.unsqueeze(-1)

    pckh = (error < PCKh_thred).float().mean(dim=-1) * 100

    return pckh