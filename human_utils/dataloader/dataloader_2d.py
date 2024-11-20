import numpy as np
import os
import pickle
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import skfmm
import glob

from PIL import Image

from human_utils.common.imglib.affine import gen_patch_image_from_box_cv, trans_point2d
from human_utils.common.imglib.format import convert_cvimg_to_tensor
from human_utils.common.utility.geodesic import compute_geodesic_dis

def center_padding(img):
    assert img.shape[0] > img.shape[1]

    length = img.shape[0]
    pad_img = np.zeros((length, length, img.shape[2]), dtype=img.dtype)

    start = (length - img.shape[1]) // 2
    pad_img[:, start:start+img.shape[1], :] = img

    return pad_img

def generate_item(smp, ct_padding=True, use_mask_center=True):

    data_path = smp['image']
    mask_path = smp['mask']
    patch_width, patch_height = 256, 256
    mean, std = [0,0,0], [255,255,255]

    cvimg = cv2.imread(data_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % data_path)

    cvmask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)[..., None]

    if not isinstance(cvmask, np.ndarray):
        raise IOError("Fail to read %s" % mask_path)

    if cvmask.shape[0] != cvimg.shape[0] or cvmask.shape[1] != cvimg.shape[1]:
        cvmask = cv2.resize(cvmask, (cvimg.shape[1], cvimg.shape[0]), interpolation=cv2.INTER_NEAREST)[..., None]

    # center padding image, assume shape[1] is shorter side
    if ct_padding:
        cvimg = center_padding(cvimg)
        cvmask = center_padding(cvmask)

    if use_mask_center:
        locs = np.where(cvmask == 255)
        tl, br = (np.min(locs[1]), np.min(locs[0])), (np.max(locs[1]), np.max(locs[0]))
        tl = (max(0, tl[0] - 20), max(0, tl[1] - 20))
        br = (min(cvimg.shape[1], br[0] + 20), min(cvimg.shape[0], br[1] + 20))

        center_x = (tl[0] + br[0]) / 2
        center_y = (tl[1] + br[1]) / 2

        # use the longer side as width
        width = max(br[0] - tl[0], br[1] - tl[1])
        height = width
    else:
        center_x = smp['center_x']
        center_y = smp['center_y']
        width = smp['width']
        height = smp['height']

    img_height, img_width, img_channels = cvimg.shape

    img_patch_cv, trans = gen_patch_image_from_box_cv(cvimg, center_x, center_y,
                                                      width, height,
                                                      patch_width, patch_height, False, 1.0, 0.0)
    img_patch = convert_cvimg_to_tensor(img_patch_cv)
    
    mask_patch_cv = cv2.warpAffine(cvmask.copy(), trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    mask_patch = mask_patch_cv[None, ...]

    # apply normalization
    for n_c in range(img_channels):
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]

    return img_patch, mask_patch, trans

class TikTok_dataset(data.Dataset):
    def __init__(self, data_path, geodesic_param_list, smpl_pseudo_img, norm_param, mode='train', rect_3d_width=256):
        self.mode = mode
        if mode == 'train':
            video_list = [34,35,36,37,40,42,43,44,45,58,\
                        59,61,62,63,76,77,104,107,112,140,\
                        142,144,146,152,158,165,195,208,221,234,\
                        238,249,251,257,275,277,280,283,303,313,\
                        323]
        else:
            video_list = [326]

        self.data_path = data_path
        self.data_db = []
        for v_id in video_list:
            root_path = os.path.join(data_path, '{:05d}'.format(v_id), 'images', '*.png')
            img_list = glob.glob(root_path)
            img_list.sort()
            # remove start and end frames
            self.data_db += img_list[20:-20]

        self.geodesic_param_list = geodesic_param_list

        if smpl_pseudo_img is not None:
            self.smpl_pseudo_img_path = smpl_pseudo_img['data_path']
            self.use_smpl_pseudo_img = smpl_pseudo_img['use_flag']
            self.use_smpl_pseudo_mask = smpl_pseudo_img['use_mask']
            if 'smpl_pseudo_img' in self.smpl_pseudo_img_path or 'smpl_part_seg_img' in self.smpl_pseudo_img_path:
                self.smpl_pseudo_img_type = 'no_texture'
                self.smpl_pseudo_img_info = np.load(os.path.join(self.smpl_pseudo_img_path, 'info.npy'), allow_pickle=True).item()
            elif 'surreal_h36m_pose' in self.smpl_pseudo_img_path:
                self.smpl_pseudo_img_type = 'ori_surreal'
                self.smpl_pseudo_img_info = np.load(os.path.join(self.smpl_pseudo_img_path, 'info.npy'))
                # self.smpl_pseudo_img_info_idx_list = np.random.choice(len(self.smpl_pseudo_img_info), len(self.cam_id_list) * self.db_length, replace=False)
            else:
                self.smpl_pseudo_img_type = None
                self.smpl_pseudo_img_info = None
                raise ValueError('smpl_pseudo_img_path is not supported')
        else:
            self.use_smpl_pseudo_img = False

        self.rect_3d_width = rect_3d_width
        self.mean, self.std = norm_param['mean'], norm_param['std']

    def generate_pseudo_smpl_data(self, out):
        cam_key = 'cam_mono'
        if self.smpl_pseudo_img_type == 'no_texture':
            iter_num = random.randint(0, self.smpl_pseudo_img_info['max_iter_num'] - 1)
            batch_idx = random.randint(0, self.smpl_pseudo_img_info['batch_size'] - 1)
            id_idx = random.randint(0, len(self.smpl_pseudo_img_info['cam_id_list']) - 1)
            pseudo_cam_id = self.smpl_pseudo_img_info['cam_id_list'][id_idx]

            img_path = os.path.join(self.smpl_pseudo_img_path, 'image', '{}_cam_{}_{}.png'.format(iter_num, pseudo_cam_id, batch_idx))
            joint_path = os.path.join(self.smpl_pseudo_img_path, 'joints', '{}_cam_{}_{}.npy'.format(iter_num, pseudo_cam_id, batch_idx))

        elif self.smpl_pseudo_img_type == 'ori_surreal':
            # info_idx = self.smpl_pseudo_img_info_idx_list[index * len(self.cam_id_list) + ii]
            info_idx = random.randint(0, len(self.smpl_pseudo_img_info) - 1)
            img_path = os.path.join(self.smpl_pseudo_img_path, 'image', 'image_{:06d}.png'.format(self.smpl_pseudo_img_info[info_idx]))
            joint_path = os.path.join(self.smpl_pseudo_img_path, 'joints', 'joint_{:06d}.npy'.format(self.smpl_pseudo_img_info[info_idx]))
            mask_path = os.path.join(self.smpl_pseudo_img_path, 'mask', 'mask_{:06d}.png'.format(self.smpl_pseudo_img_info[info_idx]))

        pseudo_img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if self.use_smpl_pseudo_mask:
            pseudo_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            pseudo_img = pseudo_img * pseudo_mask[..., None]
        pseudo_img = convert_cvimg_to_tensor(pseudo_img)

        if self.mean is not None and  self.std is not None:
            for n_c in range(pseudo_img.shape[0]):
                pseudo_img[n_c, :, :] = (pseudo_img[n_c, :, :] - self.mean[n_c]) / self.std[n_c]
        out['{}_pseudo_img'.format(cam_key)] = pseudo_img.astype(np.float32)

        # NOTE(yyc): we should convert depth from meter unit to pixel unit
        pseudo_joints = np.load(joint_path).astype(np.float32)
        if self.smpl_pseudo_img_type == 'ori_surreal':
            pseudo_joints[..., 2] = pseudo_joints[..., 2] * 1000.0 / self.rect_3d_width
        out['{}_pseudo_joints'.format(cam_key)] = pseudo_joints

        return out

    def data_color_aug(self, img):
        prob = random.random()
        if prob < 0.4:
            return img
        else:
            aug_list = [transforms.ColorJitter(brightness=.5, hue=.3, saturation=.5, contrast=.2), \
                        transforms.RandomEqualize(p=1.0), \
                        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), \
                        transforms.RandomInvert(p=1.0)]

            aug = random.choice(aug_list)

            img = (img.transpose(1,2,0)*255).astype(np.uint8)
            img = Image.fromarray(img)
            img = aug(img)
            img = np.array(img).astype(np.float32).transpose(2,0,1) / 255.0
            return img

    def __getitem__(self, index):
        out_dict = {}

        img_path = self.data_db[index]
        img_patch, mask_patch, _ = generate_item({'image': img_path,
                                                'mask': img_path.replace('images', 'masks')})

        if self.mode == 'train':
            img_patch = self.data_color_aug(img_patch)

        out_dict['cam_mono_img_ori'] = img_patch.astype(np.float32)
        out_dict['cam_mono_img'] = img_patch.astype(np.float32)
        out_dict['cam_mono_mask'] = mask_patch.astype(np.float32) / 255.0

        out_dict['cam_mono_img_path'] = img_path

        out_dict['cam_mono_img'] = out_dict['cam_mono_img'] * out_dict['cam_mono_mask']

        out_dict['cam_mono_geodesic_dis'], out_dict['cam_mono_geodesic_center'] = \
            compute_geodesic_dis(out_dict['cam_mono_mask'], img_path, self.geodesic_param_list)

        k_mat = np.zeros([3,3], dtype=np.float32)
        k_mat[0,0] = 1.0
        k_mat[1,1] = 1.0
        k_mat[2,2] = 1.0

        out_dict['cam_mono_k_mat'] = k_mat

        out_dict['cam_mono_pelvis'] = np.zeros([3], dtype=np.float32)
        out_dict['cam_mono_rot_world'] = np.eye(3, dtype=np.float32)
        out_dict['cam_mono_trans_world'] = np.zeros(3, dtype=np.float32)

        trans = np.zeros([2,3], dtype=np.float32)
        trans[0, 0] = 1.0
        trans[1, 1] = 1.0
        out_dict['cam_mono_trans_image'] = trans

        if self.use_smpl_pseudo_img:
            out_dict = self.generate_pseudo_smpl_data(out_dict)
        return out_dict
    
    def __len__(self):
        return len(self.data_db)
    


class mpii_dataset(data.Dataset):
    def __init__(self, database, mode='valid'):
        assert mode == 'valid', 'only used for validation'
        self.data_db = database.gt_db()
    
    def __getitem__(self, index):
        out_dict = {}

        smp = self.data_db[index]['cam_mono']
        img_patch, mask_patch, trans = generate_item(smp, ct_padding=False, use_mask_center=False)

        out_dict['cam_mono_img_ori'] = img_patch.astype(np.float32)
        out_dict['cam_mono_img'] = img_patch.astype(np.float32)
        out_dict['cam_mono_mask'] = mask_patch.astype(np.float32) / 255.0

        ori_joints = smp['joints_3d']
        ori_joints_ = ori_joints.copy()
        for i in range(len(ori_joints)):
            ori_joints_[i, :2] = trans_point2d(ori_joints[i, :2], trans)
        out_dict['cam_mono_joints'] = ori_joints_

        out_dict['cam_mono_img_path'] = smp['image']

        out_dict['cam_mono_img'] = out_dict['cam_mono_img'] * out_dict['cam_mono_mask']

        k_mat = np.zeros([3,3], dtype=np.float32)
        k_mat[0,0] = 1.0
        k_mat[1,1] = 1.0
        k_mat[2,2] = 1.0
        out_dict['cam_mono_k_mat'] = k_mat

        out_dict['cam_mono_pelvis'] = np.zeros([3], dtype=np.float32)
        out_dict['cam_mono_rot_world'] = np.eye(3, dtype=np.float32)
        out_dict['cam_mono_trans_world'] = np.zeros(3, dtype=np.float32)

        out_dict['cam_mono_trans_image'] = trans.astype(np.float32)

        out_dict['cam_mono_head_size'] = smp['head_size']

        return out_dict
    
    def __len__(self):
        return len(self.data_db)