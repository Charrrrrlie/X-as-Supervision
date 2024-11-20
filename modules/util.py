import torch

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed

def draw_lines(keypoints, image_size, parent_ids, child_ids, body_width):
    """
    Draw lines on a grid.
    :param keypoints: (batch_size, n_points, 2) in [-1, 1]
    :return: (batch_size, 1, grid_size, grid_size)
    dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2
    """
    # keypoints shape [16, 18, 2]
    # image_size = x.shape[-1] 256
    # line_select_ids = [0, 1 ,2 ... 20]
    # return heatmaps_size = [16, 21, 256, 256]
    B, N = keypoints.shape[:2]

    start = keypoints[:, child_ids, :].clone()
    end = keypoints[:, parent_ids, :].clone()
    paired_diff = end - start # bone length
    grid = make_coordinate_grid((image_size, image_size), keypoints.dtype).to(keypoints.device).reshape(1, 1, -1, 2)
    diff_to_start = grid - start.unsqueeze(-2)

    t = (diff_to_start @ paired_diff.unsqueeze(-1)).squeeze(-1) / (1e-8+paired_diff.square().sum(dim=-1, keepdim=True))

    diff_to_end = grid - end.unsqueeze(-2)

    before_start = (t <= 0).float() * diff_to_start.square().sum(dim=-1)
    after_end = (t >= 1).float() * diff_to_end.square().sum(dim=-1)
    between_start_end = (0 < t).float() * (t < 1).float() * (grid - (start.unsqueeze(-2) + t.unsqueeze(-1) * paired_diff.unsqueeze(-2))).square().sum(dim=-1)

    squared_dist = (before_start + after_end + between_start_end).reshape(B, -1, image_size, image_size)

    if squared_dist.shape[1] >= 21:
        # condition: arms use fine lines
        squared_dist = -squared_dist / body_width
        squared_dist[:, [11, 12, 14, 15], :, :] = squared_dist[:, [11, 12, 14, 15], :, :] * 2

        heatmaps = torch.exp(squared_dist)
    else:
        heatmaps = torch.exp(-squared_dist / body_width)

    return heatmaps

def convert_patch_to_image(kps, trans, image_depth, image_height, image_width, depth_scale, pelvis, is_norm=True):
    # inverse
    trans = trans.clone()
    trans[..., :, :2] = torch.linalg.inv(trans[..., :, :2])
    trans[..., :, 2] = - trans[..., :, 2]

    kp_image = kps.clone()

    if is_norm:
        kp_image[..., 0] = (kp_image[..., 0] + 1) / 2.0 * (image_width - 1)
        kp_image[..., 1] = (kp_image[..., 1] + 1) / 2.0 * (image_height - 1)
        kp_image[..., 2] = kp_image[..., 2] * (image_depth - 1)

    trans = trans.unsqueeze(1).repeat(1, kp_image.shape[1], 1, 1)
    # x = inv(R) (x' - t)
    kp_image[..., :2] = (trans[..., :, :2] @ (kp_image[..., :2].unsqueeze(-1) + trans[..., :, [2]])).squeeze(-1)
    
    # pixel to mm
    kp_image[..., 2] = kp_image[..., 2] * depth_scale
    kp_image[..., 2] = kp_image[..., 2] + pelvis[..., 2].unsqueeze(1)

    return kp_image


def convert_image_to_world(kps, fx, fy, u, v, trans, rot):
    kp_world = kps.clone()

    # back-projection
    kp_world[..., 0] = (kp_world[..., 0].clone() - u) / fx * kp_world[..., 2].clone()
    kp_world[..., 1] = (kp_world[..., 1].clone() - v) / fy * kp_world[..., 2].clone()

    rot = rot.unsqueeze(1).repeat(1, kp_world.shape[1], 1, 1)
    kp_world = (torch.linalg.inv(rot) @ (kp_world - trans.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)

    return kp_world


def convert_image_to_patch(kps, trans, image_depth, image_height, image_width, depth_scale, pelvis, is_norm=True):
    kp_patch = kps.clone()

    # mm to pixel
    kp_patch[..., 2] = kp_patch[..., 2] - pelvis[..., 2].unsqueeze(1)
    kp_patch[..., 2] = kp_patch[..., 2] / depth_scale

    trans = trans.unsqueeze(1).repeat(1, kp_patch.shape[1], 1, 1)
    kp_patch[..., :2] = (trans[..., :, :2] @ kp_patch[..., :2].unsqueeze(-1) + trans[..., :, [2]]).squeeze(-1)

    if is_norm:
        kp_patch[..., 0] = kp_patch[..., 0] / (image_width - 1) * 2 - 1
        kp_patch[..., 1] = kp_patch[..., 1] / (image_height - 1) * 2 - 1
        kp_patch[..., 2] = kp_patch[..., 2] / (image_depth - 1)

    return kp_patch


def convert_world_to_image(kps, fx, fy, u, v, trans, rot):
    kp_image = kps.clone()

    rot = rot.unsqueeze(1).repeat(1, kp_image.shape[1], 1, 1)
    kp_image = (rot @ kp_image.unsqueeze(-1) + trans.unsqueeze(1).unsqueeze(-1)).squeeze(-1)

    kp_image[..., 0] = kp_image[..., 0].clone() / kp_image[..., 2].clone() * fx + u
    kp_image[..., 1] = kp_image[..., 1].clone() / kp_image[..., 2].clone() * fy + v

    return kp_image


def convert_patch_to_world(keypoints, params, mode, is_norm=True, RECT_WIDTH=2000, mono=False, patch=True):
    trans_img = params['{}_trans_image'.format(mode)].clone()
    shape_img = params['{}_img'.format(mode)].shape
    pelvis = params['{}_pelvis'.format(mode)]
    k_mat = params['{}_k_mat'.format(mode)]
    trans_world = params['{}_trans_world'.format(mode)]
    rot_world = params['{}_rot_world'.format(mode)]

    if patch:
        kp_img = convert_patch_to_image(keypoints, trans_img, shape_img[-1],shape_img[-2],
                                        shape_img[-1], 1.0 / shape_img[-1] * RECT_WIDTH, pelvis, is_norm=is_norm)
    else:
        kp_img = keypoints

    if not mono:
        kp_world = convert_image_to_world(kp_img,k_mat[..., 0, [0]], k_mat[..., 1, [1]],\
                                        k_mat[..., 0, [2]], k_mat[..., 1, [2]], trans_world, rot_world)
    else:
        kp_world = kp_img.clone()
        # for visualization
        kp_world[..., 2] = kp_world[..., 2] + 128
        kp_world = kp_world[..., [0,2,1]]
        kp_world = -kp_world

    return kp_world


def convert_world_to_patch(keypoints, params, mode, is_norm=True, RECT_WIDTH=2000):
    trans_img = params['{}_trans_image'.format(mode)].clone()
    shape_img = params['{}_img'.format(mode)].shape
    pelvis = params['{}_pelvis'.format(mode)]
    k_mat = params['{}_k_mat'.format(mode)]
    trans_world = params['{}_trans_world'.format(mode)]
    rot_world = params['{}_rot_world'.format(mode)]

    kp_img = convert_world_to_image(keypoints, k_mat[..., 0, [0]], k_mat[..., 1, [1]],\
                                        k_mat[..., 0, [2]], k_mat[..., 1, [2]], trans_world, rot_world)
    kp_patch = convert_image_to_patch(kp_img, trans_img, shape_img[-1], shape_img[-2],
                                        shape_img[-1], 1.0 / shape_img[-1] * RECT_WIDTH, pelvis, is_norm=is_norm)

    return kp_patch


def triangulation(keypoints, params, cam_id_list, is_norm=True, RECT_WIDTH=2000):
    points_all = []
    p_matrix_all = []

    for cam_id in cam_id_list:
        mode = 'cam_{}'.format(cam_id)
        trans_img = params['{}_trans_image'.format(mode)].clone()
        shape_img = params['{}_img'.format(mode)].shape
        pelvis = params['{}_pelvis'.format(mode)]
        k_mat = params['{}_k_mat'.format(mode)]
        trans_world = params['{}_trans_world'.format(mode)].clone()
        rot_world = params['{}_rot_world'.format(mode)].clone()

        kp_img= convert_patch_to_image(keypoints[mode], trans_img, shape_img[-1],shape_img[-2],
                                            shape_img[-1], 1.0 / shape_img[-1] * RECT_WIDTH, pelvis, is_norm=is_norm)
        points_all.append(kp_img.unsqueeze(1))

        p_matrix = k_mat @ torch.cat([rot_world, trans_world.unsqueeze(-1)], dim=-1)
        p_matrix_all.append(p_matrix.unsqueeze(1))

    points_all = torch.cat(points_all, dim=1)
    p_matrix_all = torch.cat(p_matrix_all, dim=1)
    keypoints_3d = batch_triangulate(points_all, p_matrix_all)[..., :3]

    return keypoints_3d


def batch_triangulate(keypoints_, Pall):
    """ triangulate the keypoints of whole body
    Args:
        keypoints_ (Batch, nViews, nJoints, 3): 2D detections
        Pall (Batch, nViews, 3, 4): projection matrix of each view
        min_view (int, optional): min view for visible points. Defaults to 2.
    Returns:
        keypoints3d: (nJoints, 4)
    """

    v = (keypoints_[:, :, :, -1]>0).sum(axis=1)
    keypoints = keypoints_
    conf3d = keypoints[..., -1].sum(axis=1) / v
    # P2: (B, 1, nViews, 4)
    P0 = Pall[:, :, 0, :].unsqueeze(1)
    P1 = Pall[:, :, 1, :].unsqueeze(1)
    P2 = Pall[:, :, 2, :].unsqueeze(1)
    # uP2: (B, nJoints, nViews, 4)
    uP2 = keypoints[..., [0]].permute(0, 2, 1, 3) * P2
    vP2 = keypoints[..., [1]].permute(0, 2, 1, 3) * P2
    conf = keypoints[..., [2]].permute(0, 2, 1, 3)
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = torch.cat([Au, Av], dim=2)
    u, s, v = torch.linalg.svd(A)
    X = v[:, :, -1, :]
    X = X / X[:, :, 3:]

    # out: (nJoints, 4)
    result = torch.zeros((keypoints_.shape[0], keypoints_.shape[2], 4)).to(keypoints_.device)
    result[:, :, :3] = X[:, :, :3]
    result[:, :, 3] = conf3d    # BNJ
    return result


def my_truncated_normal(pos, neg, size=[1,1], ignore=0.4, mean=0.0):

    if torch.rand(1) < ignore:
        return torch.zeros(size)

    # 50% use positive, 50% use negative
    if torch.rand(1) < 0.5:
        width = pos
        low = -pos
        up = pos
        flag = 1
        if pos == mean:
            return torch.zeros(size)
    else:
        width = neg
        low = -neg
        up = neg
        flag = -1
        if neg == mean:
            return torch.zeros(size)

    # 95% of the data is within 1.96 std
    std = width / 1.96
    return torch.abs(torch.clip(torch.normal(0, std, size=size), low, up)) * flag + mean


def rule_transformation(pose, beta, batch_size, gen_nagative=False):

    beta = my_truncated_normal(1.5, 1.5, size=[batch_size, 10], ignore=0)  # roughly set

    if not gen_nagative:
        range_list = [[5], [180], [5],
                    [45, 60], [10, 10], [30, 0], 
                    [45, 60], [10, 10], [0, 30],
                    [60, 20], [30, 30], [30, 30],
                    [70, 0], [20, 20], [10, 10],
                    [70, 0], [20, 20], [10, 10],
                    [20, 10], [0, 0], [15, 15],
                    [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0],
                    [15, 15], [50, 50], [15, 15],
                    [90, 90], [50, 120], [150, 30, -60],
                    [90, 90], [120, 50], [30, 150, 60],
                    [60, 60], [0, 120], [15, 15],
                    [60, 60], [120, 0], [15, 15],
                    [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0],
        ]

    else:
        range_list = [[5], [180], [5],
                        [70,90], [10, 10], [30, 0], ##
                        [70, 90], [10, 10], [0, 30], ##
                        [30, 40], [30, 30], [30, 30], ##
                        [10, 50], [20, 20], [10, 10], ##
                        [10, 50], [20, 20], [10, 10], ## 
                        [20, 10], [0, 0], [15, 15],
                        [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0],
                        [15, 15], [50, 50], [15, 15],
                        [90, 90], [50, 120], [150, 30, -60],
                        [90, 90], [120, 50], [30, 150, 60],
                        [60, 60], [0, 120], [15, 15],
                        [60, 60], [120, 0], [15, 15],
                        [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0],
                        [0, 0], [0, 0], [0, 0],
        ]
        
    for i, r_l in enumerate(range_list):
        if len(r_l) == 1:
            # NOTE(yyc): for root only
            pose[:, i] = my_truncated_normal(r_l[0] * torch.pi / 180.0, r_l[0] * torch.pi / 180.0, size=[batch_size], ignore=0)
        elif len(r_l) == 2:
            pose[:, i] = my_truncated_normal(r_l[0] * torch.pi / 180.0, r_l[1] * torch.pi / 180.0, size=[batch_size])
        elif len(r_l) == 3:
            pose[:, i] = my_truncated_normal(r_l[0] * torch.pi / 180.0, r_l[1] * torch.pi / 180.0, mean=r_l[2] * torch.pi / 180.0, size=[batch_size])
        else:
            pass

    return pose, beta


def smpl_to_h36m(verts, h36m_regressor):
    joints = torch.einsum('bki,lk->bli', verts, h36m_regressor)
    joints[:, [11, 12, 13, 14, 15, 16]] = joints[:, [14, 15, 16, 11, 12, 13]]

    # add thorax to joints
    joints = torch.cat([joints, joints[:, [11, 14], :].mean(dim=1, keepdim=True)], dim=1)

    # move pelvis to center
    joints = joints - joints[:, [0], :]

    return joints

def convert_pelvis_to_world(x, mode):
    # convert pelvis to world coord
    pelvis = x['{}_pelvis'.format(mode)].clone().unsqueeze(1)

    trans_world = x['{}_trans_world'.format(mode)].clone()
    rot_world = x['{}_rot_world'.format(mode)].clone()
    rot_world = rot_world.unsqueeze(1).repeat(1, pelvis.shape[1], 1, 1)
    pelvis = (torch.linalg.inv(rot_world) @ (pelvis - trans_world.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)

    return pelvis

# NOTE(yyc): we control it by global_rot_params
# thus we do not use joints = joints[:, :, [0, 2, 1]]; joints[:, :, 1] = -joints[:, :, 1]
def project_smpl_to_patch_kps(global_rot_params, pose_params, shape_params,\
                               smpl_layer, h36m_regressor, x, mode, convert_verts=False):
    # we separately use global rot and smpl layer
    # for better h36m regressor performance
    full_pose_params = torch.zeros(pose_params.shape[0], 72).to(pose_params.device)
    full_pose_params[:, 3:] = pose_params
    verts, _ = smpl_layer(full_pose_params, shape_params)

    # convert pelvis to world coord
    pelvis = convert_pelvis_to_world(x, mode)

    if convert_verts:
        verts = torch.bmm(verts, global_rot_params)
        verts = verts * 1000
        verts = verts + pelvis.to(verts.device)

        # in world coord
        return verts
    else:
        joints = smpl_to_h36m(verts, h36m_regressor)
        joints = torch.bmm(joints, global_rot_params)

        # from m to mm
        joints = joints * 1000
        joints = joints + pelvis

        joints = convert_world_to_patch(joints, x, mode, is_norm=False)
        return joints

    # DEBUG use
    # joints_gt = convert_patch_to_world(x['cam_0_joints'], x, 'cam_0', is_norm=False)
    # return joints, verts

def random_rotation_3D(keypoints):
    # we only rotate around z-axis in [-1/4, 1/4] pi
    B = keypoints.shape[0]
    rot_range = torch.pi
    rot_angle = (torch.rand(B, 1) - 0.5) * 0.5 * rot_range

    rot_angle = rot_angle.squeeze()
    rot_matrix = torch.zeros(B, 3, 3).to(keypoints.device)

    rot_matrix[:, 0, 0] = torch.cos(rot_angle)
    rot_matrix[:, 0, 1] = -torch.sin(rot_angle)
    rot_matrix[:, 1, 0] = torch.sin(rot_angle)
    rot_matrix[:, 1, 1] = torch.cos(rot_angle)
    rot_matrix[:, 2, 2] = 1

    keypoints = keypoints.clone()
    keypoints = torch.bmm(keypoints, rot_matrix)

    return keypoints

def flip_3D(keypoints):
    keypoints = keypoints.clone()

    if torch.rand(1) < 0.5:
        keypoints[:, [1, 2, 3, 4, 5, 6], :] = keypoints[:, [4, 5, 6, 1, 2, 3], :]
    else:
        keypoints[:, [11, 12, 13, 14, 15, 16], :] = keypoints[:, [14, 15, 16, 11, 12, 13], :]

    return keypoints