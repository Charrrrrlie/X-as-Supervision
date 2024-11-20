import cv2
import glob
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from surreal_utils import project_vertices, get_extrinsic, rotateBody, draw_joints2D

def construct_dataset(info, m, t, intrinsic, h36m_regressor, box_size=2.0):

    root_pos = m.J_transformed.r[0]

    zrot = info['zrot']
    zrot = zrot[0][0]  # body rotation in euler angles
    RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0),
                       (math.sin(zrot), math.cos(zrot), 0),
                       (0, 0, 1)))
    info['camLoc'] = info['camLoc'].reshape(3, 1)
    extrinsic, R, T = get_extrinsic(info['camLoc'])

    joints3D = info['joints3D'][:, :, t].T
    pose = info['pose'][:, t]
    pose[0:3] = rotateBody(RzBody, pose[0:3])
    # Set model shape
    m.betas[:] = info['shape'][:, 0]
    # Set model pose
    m.pose[:] = pose
    # Set model translation

    m.trans[:] = joints3D[0] - root_pos

    smpl_vertices = m.r
    smpl_joints3D = m.J_transformed.r

    # construct h36m joints3D
    smpl_joints3D = np.dot(h36m_regressor, smpl_vertices)
    smpl_joints3D[[11, 12, 13, 14, 15, 16], :] = smpl_joints3D[[14, 15, 16, 11, 12, 13], :]
    smpl_joints3D = np.concatenate([smpl_joints3D, smpl_joints3D[[11, 14], :].mean(axis=0, keepdims=True)], axis=0)

    # Project 3D -> 2D image coords
    proj_smpl_vertices = project_vertices(smpl_vertices, intrinsic, extrinsic)
    proj_smpl_joints3D = project_vertices(smpl_joints3D, intrinsic, extrinsic, centralize_joints=True)

    # determine crop bbox
    smpl_box_lt_3D = smpl_joints3D[0].copy()
    smpl_box_rb_3D = smpl_joints3D[0].copy()
    smpl_box_lt_3D[1:] -= box_size / 2
    smpl_box_rb_3D[1:] += box_size / 2

    proj_smpl_box_lt = project_vertices(smpl_box_lt_3D[None, :], intrinsic, extrinsic)
    proj_smpl_box_rb = project_vertices(smpl_box_rb_3D[None, :], intrinsic, extrinsic)

    return proj_smpl_joints3D, proj_smpl_vertices, proj_smpl_box_lt.astype(np.int16), proj_smpl_box_rb.astype(np.int16)


def vis_check(check_img, mask, proj_joints3D, proj_vertices, save_path, kintree_table):
    check_img = check_img[:, :, [2, 1, 0]].copy()

    plt.figure(figsize=(18, 10))

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(check_img)

    # Show joints projection
    draw_joints2D(proj_joints3D, ax1, kintree_table, color='r')

    # Show vertices projection
    plt.subplot(1, 2, 2)
    plt.scatter(proj_vertices[:, 0], proj_vertices[:, 1], 1)
    plt.imshow(check_img * mask)

    plt.savefig(save_path.replace('.png', '_check.png'))
    plt.close()


def load_info_npy(root_path, mode):
    # ***.mp4, ***_info.mat
    if os.path.exists(os.path.join(root_path, '{}_info.npy'.format(mode))):
        print('Loading info from existing file {}'.format(os.path.join(root_path, '{}_info.npy'.format(mode))))
        return np.load(os.path.join(root_path, '{}_info.npy'.format(mode)))

    print('Loading info files from {}'.format(os.path.join(root_path, mode)))
    info_files = glob.glob(os.path.join(root_path, mode, '*', '*', '*_info.mat'), recursive=True)
    info_files.extend(glob.glob(os.path.join(root_path, '*', '*_info.mat'), recursive=True))

    info_files = [info_file for info_file in info_files if 'ung_' not in info_file]
    np.save(os.path.join(root_path, '{}_info.npy'.format(mode)), info_files)
    info_files = sorted(info_files)
    print('{} segments have been recorded'.format(len(info_files)))
    return info_files

def sample_time_idx(info, margin, sample_times, max_iter=100, ignore_center=False):
    time_list = []
    for _ in range(max_iter):
        t = np.random.randint(0, info['joints3D'].shape[2])
        # NOTE(yyc): centered frame
        if ignore_center or (info['joints2D'][0, 0, t] >= 150 and info['joints2D'][0, 0, t] <= 170 \
            and info['joints2D'][1, 0, t] >= 110 and info['joints2D'][1, 0, t] <= 130 \
                and np.all(info['joints2D'][0, :, t] >= margin) and np.all(info['joints2D'][0, :, t] <= 320) \
                and np.all(info['joints2D'][1, :, t] >= 0) and np.all(info['joints2D'][1, :, t] <= 240)):
            time_list.append(t)
            if len(time_list) >= sample_times:
                break
    return time_list

def crop_and_resize(rgb, mask, joints3D, vertices, box_lt, box_rb, target_x_px, target_y_px):
    def center_padding(img):
        length = max(img.shape[0], img.shape[1])
        pad_img = np.zeros((length, length, img.shape[2]), dtype=img.dtype)

        if img.shape[0] > img.shape[1]:
            start = (length - img.shape[1]) // 2
            pad_img[:, start:start + img.shape[1], :] = img
        else:
            start = (length - img.shape[0]) // 2
            pad_img[start:start + img.shape[0], :, :] = img

        return pad_img

    # NOTE(yyc): reshape to target rgb
    # TODO: refactor this part
    x_start = max(0, min(box_lt[0, 0], box_rb[0, 0]))
    y_start = max(0, min(box_lt[0, 1], box_rb[0, 1]))
    x_end = min(rgb.shape[1] - 1, max(box_rb[0, 0] + 1, box_lt[0, 0] + 1))
    y_end = min(rgb.shape[0] - 1, max(box_rb[0, 1] + 1, box_lt[0, 1] + 1))

    width = max(x_end - x_start, y_end - y_start)
    y_center = (y_start + y_end) // 2
    x_center = (x_start + x_end) // 2

    rgb = rgb[max(0, y_center-width//2): min(rgb.shape[0]-1, y_center+width//2),
               max(0, x_center-width//2): min(rgb.shape[1]-1, x_center+width//2), :]
    mask = mask[max(0, y_center-width//2): min(mask.shape[0]-1, y_center+width//2),
                max(0, x_center-width//2): min(mask.shape[1]-1, x_center+width//2), :]

    x_pad, y_pad = 0, 0
    # center padding to square
    if rgb.shape[0] > rgb.shape[1]:
        x_pad = (rgb.shape[0] - rgb.shape[1]) // 2
        rgb = center_padding(rgb)
        mask = center_padding(mask)
    elif rgb.shape[0] < rgb.shape[1]:
        y_pad = (rgb.shape[1] - rgb.shape[0]) // 2
        rgb = center_padding(rgb)
        mask = center_padding(mask)

    ori_x_px = rgb.shape[0]
    ori_y_px = rgb.shape[1]

    rgb = cv2.resize(rgb, (target_x_px, target_y_px))
    mask = cv2.resize(mask, (target_x_px, target_y_px))
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)

    joints3D[:, 0] = joints3D[:, 0] - max(0, (x_center - width // 2)) + x_pad
    vertices[:, 0] = vertices[:, 0] - max(0, (x_center - width // 2)) + x_pad
    joints3D[:, 1] = joints3D[:, 1] - max(0, (y_center - width // 2)) + y_pad
    vertices[:, 1] = vertices[:, 1] - max(0, (y_center - width // 2)) + y_pad

    joints3D[:, 0] = joints3D[:, 0] * target_y_px / ori_y_px
    vertices[:, 0] = vertices[:, 0] * target_y_px / ori_y_px
    joints3D[:, 1] = joints3D[:, 1] * target_x_px / ori_x_px
    vertices[:, 1] = vertices[:, 1] * target_x_px / ori_x_px

    return rgb, mask, joints3D, vertices