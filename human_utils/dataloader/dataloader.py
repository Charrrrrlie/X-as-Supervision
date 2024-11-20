import numpy as np
import os
import cv2

import torch.utils.data as data
import random

from human_utils.common.utility.augment import do_augmentation
from human_utils.common.imglib.affine import gen_patch_image_from_box_cv, fliplr_joints, trans_points_3d, norm_rot_angle
from human_utils.common.imglib.format import convert_cvimg_to_tensor

from human_utils.dataset.mpi_inf_3dhp import from_mpi_inf_3dhp_to_hm36
from human_utils.common.utility.geodesic import compute_geodesic_dis


# patch_sample is defined in datasets.imdb
def generate_patch_sample_data(patch_sample, patch_width, patch_height, _rect_3d_width, _rect_3d_height, mean, std,
                               do_augment, aug_config, label_func):
    if _rect_3d_width <= 0 or _rect_3d_height <= 0:
        rect_3d_width = patch_sample.width
        rect_3d_height = patch_sample.height
    else:
        rect_3d_width = _rect_3d_width
        rect_3d_height = _rect_3d_height

    # 1. load image
    cvimg = cv2.imread(patch_sample.image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % patch_sample.image)
    
    if 'hm36' in patch_sample.image:
        mask_path = patch_sample.image.replace('hm36/images', 'sam_masks/hm36').replace('jpg', 'png')
        cvmask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    elif 'mpi_inf_3dhp' in patch_sample.image:
        mask_path = patch_sample.image.replace('images', 'masks').replace('mpi_inf_3dhp', 'sam_masks/mpi_inf_3dhp' )
        cvmask = cv2.imread(mask_path)[..., [2]]

    if not isinstance(cvmask, np.ndarray):
        raise IOError("Fail to read %s" % mask_path)

    img_height, img_width, img_channels = cvimg.shape

    # 2. get augmentation params
    if do_augment:
        scale, rot, do_flip, color_scale = do_augmentation(aug_config)
    else:
        scale, rot, do_flip, color_scale = 1.0, 0, False, [1.0, 1.0, 1.0]

    if do_flip:
        rot = rot - patch_sample.rot
    else:
        rot = rot + patch_sample.rot
    rot = norm_rot_angle(rot)

    # 3. generate image patch
    img_patch_cv, trans = gen_patch_image_from_box_cv(cvimg, patch_sample.center_x, patch_sample.center_y,
                                                      patch_sample.width, patch_sample.height,
                                                      patch_width, patch_height, do_flip, scale, rot)
    img_patch = convert_cvimg_to_tensor(img_patch_cv)

    mask_patch_cv = cv2.warpAffine(cvmask.copy(), trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)
    # (1,H,W)
    mask_patch = mask_patch_cv[None, ...]

    if 'mpi_inf_3dhp' in patch_sample.image:
        # convert into {0, 255}
        mask_patch = cv2.GaussianBlur(mask_patch, (5, 5), 0)
        mask_patch = cv2.threshold(mask_patch, 127, 255, cv2.THRESH_BINARY)[1]

    # apply normalization
    for n_c in range(img_channels):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]

    # 4. generate patch joint ground truth. Flip joints and apply AffineTransform on joints
    if do_flip:
        joints, joints_vis = \
            fliplr_joints(patch_sample.joints_3d, patch_sample.joints_3d_vis, img_width, patch_sample.flip_pairs)
    else:
        joints, joints_vis = patch_sample.joints_3d.copy(), patch_sample.joints_3d_vis.copy()

    # NOTE(xiao): we assume depth == width
    joints = trans_points_3d(joints, trans, 1.0 / (rect_3d_width * scale) * patch_width)

    # 5. get label of some type according to certain need
    label, label_weight = label_func(patch_width, patch_height, joints, joints_vis,
                                     patch_sample.parent_ids)


    return img_patch, mask_patch, label, label_weight, joints, trans


class PatchDataset(data.Dataset):
    def __init__(self, database, is_train, patch_width, patch_height, rect_3d_width, rect_3d_height,
                 batch_size, mean, std, aug_config, label_func, cam_id_list, geodesic_pt_list, geodesic_param_list, smpl_pseudo_img,\
                 rm_bg, convert_to_17kps):
        self.db = database[0].gt_db()

        if convert_to_17kps:
            from_mpi_inf_3dhp_to_hm36(self.db)

        self.num_samples = len(self.db)

        self.is_train = is_train
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.aug_config = aug_config
        self.label_func = label_func
        self.cam_id_list = cam_id_list

        self.geodesic_pt_list = geodesic_pt_list
        self.geodesic_param_list = geodesic_param_list

        self.rm_bg = rm_bg

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False

        # padding samples to match input_batch_size (db_length > len(self.db))
        extra_db = len(self.db) % batch_size
        for i in range(0, batch_size - extra_db):
            self.db.append(self.db[i])
        self.db_length = len(self.db)

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
            else:
                self.smpl_pseudo_img_type = None
                self.smpl_pseudo_img_info = None
                raise ValueError('smpl_pseudo_img_path is not supported')
        else:
            self.use_smpl_pseudo_img = False

    def generate_item(self, smp, cam_key, out_dict):
        img_patch, mask_patch, label, label_weight, joints, trans = generate_patch_sample_data(
            smp,
            self.patch_width,
            self.patch_height,
            self.rect_3d_width,
            self.rect_3d_height,
            self.mean,
            self.std,
            self.do_augment,
            self.aug_config,
            self.label_func)

        img_patch, mask_patch, label, joints, trans =  img_patch.astype(np.float32), mask_patch.astype(np.float32), label.astype(np.float32),\
               joints.astype(np.float32), trans.astype(np.float32)

        out_dict['{}_img'.format(cam_key)] = img_patch     # nomalized image
        out_dict['{}_joints'.format(cam_key)] = joints    # local image coord [N,3] in pixel
        out_dict['{}_img_path'.format(cam_key)] = smp['image']

        k_mat = np.zeros([3,3]).astype(joints.dtype)
        k_mat[0, 0] = smp['fl'][0]
        k_mat[1, 1] = smp['fl'][1]
        k_mat[0, 2] = smp['c_p'][0]
        k_mat[1, 2] = smp['c_p'][1]
        k_mat[2, 2] = 1
        out_dict['{}_k_mat'.format(cam_key)] = k_mat

        out_dict['{}_pelvis'.format(cam_key)] = smp['pelvis'].astype(joints.dtype)
        out_dict['{}_rot_world'.format(cam_key)] = smp['rot_world'].astype(joints.dtype)
        out_dict['{}_trans_world'.format(cam_key)] = smp['trans_world'].astype(joints.dtype)

        out_dict['{}_trans_image'.format(cam_key)] = trans

        out_dict['{}_mask'.format(cam_key)] = mask_patch / 255.0

        if self.rm_bg:
            out_dict['{}_img'.format(cam_key)] = out_dict['{}_img'.format(cam_key)] * out_dict['{}_mask'.format(cam_key)]

        out_dict['{}_geodesic_dis'.format(cam_key)], out_dict['{}_geodesic_center'.format(cam_key)] = \
            compute_geodesic_dis(out_dict['{}_mask'.format(cam_key)], smp['image'], self.geodesic_param_list, \
            centers=out_dict['{}_joints'.format(cam_key)][self.geodesic_pt_list] if len(self.geodesic_pt_list) else None)

    def generate_pseudo_smpl_data(self, out):
        for ii, cam_id in enumerate(self.cam_id_list):
            cam_key = 'cam_{}'.format(cam_id)

            if self.smpl_pseudo_img_type == 'no_texture':
                iter_num = random.randint(0, self.smpl_pseudo_img_info['max_iter_num'] - 1)
                batch_idx = random.randint(0, self.smpl_pseudo_img_info['batch_size'] - 1)
                id_idx = random.randint(0, len(self.smpl_pseudo_img_info['cam_id_list']) - 1)
                pseudo_cam_id = self.smpl_pseudo_img_info['cam_id_list'][id_idx]

                img_path = os.path.join(self.smpl_pseudo_img_path, 'image', '{}_cam_{}_{}.png'.format(iter_num, pseudo_cam_id, batch_idx))
                joint_path = os.path.join(self.smpl_pseudo_img_path, 'joints', '{}_cam_{}_{}.npy'.format(iter_num, pseudo_cam_id, batch_idx))

            elif self.smpl_pseudo_img_type == 'ori_surreal':
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

    def __getitem__(self, index):
        out = {}
        for cam_id in self.cam_id_list:
            cam_key = 'cam_{}'.format(cam_id)
            self.generate_item(self.db[index][cam_key], cam_key, out)

        if self.use_smpl_pseudo_img and self.is_train:
            self.generate_pseudo_smpl_data(out)

        out['act'] = self.db[index]['cam_0']['image'].split('/')[-1][5:21]  # e.g. 'act_04_subact_01'

        return out

    def __len__(self):
        return self.db_length


class hm36_Dataset(PatchDataset):
    def __init__(self, database, is_train, patch_width, patch_height, rect_3d_width, rect_3d_height,
                 batch_size, mean, std, aug_config, label_func, cam_id_list, geodesic_pt_list, geodesic_param_list, smpl_pseudo_img,\
                 rm_bg=True, convert_to_17kps=False):
        PatchDataset.__init__(self, database, is_train, patch_width, patch_height, rect_3d_width, rect_3d_height,
                              batch_size, mean, std, aug_config, label_func, cam_id_list, geodesic_pt_list, geodesic_param_list, smpl_pseudo_img,\
                              rm_bg, convert_to_17kps)

class mpi_inf_3dhp_Dataset(PatchDataset):
    def __init__(self, database, is_train, patch_width, patch_height, rect_3d_width, rect_3d_height,
                 batch_size, mean, std, aug_config, label_func, cam_id_list, geodesic_pt_list, geodesic_param_list, smpl_pseudo_img,\
                 rm_bg=True, convert_to_17kps=True):
        PatchDataset.__init__(self, database, is_train, patch_width, patch_height, rect_3d_width, rect_3d_height,
                              batch_size, mean, std, aug_config, label_func, cam_id_list, geodesic_pt_list, geodesic_param_list, smpl_pseudo_img,\
                              rm_bg, convert_to_17kps)

class mpi_inf_3dhp_hm36_Dataset(PatchDataset):
    def __init__(self, database, is_train, patch_width, patch_height, rect_3d_width, rect_3d_height,
                 batch_size, mean, std, aug_config, label_func, cam_id_list, geodesic_pt_list, geodesic_param_list, smpl_pseudo_img,\
                 rm_bg=True, convert_to_17kps=False):
        PatchDataset.__init__(self, database, is_train, patch_width, patch_height, rect_3d_width, rect_3d_height,
                              batch_size, mean, std, aug_config, label_func, cam_id_list, geodesic_pt_list, geodesic_param_list, smpl_pseudo_img,\
                              rm_bg, convert_to_17kps)
        self.db0 = database[0].gt_db()  # mpi_inf_3dhp
        self.db1 = database[1].gt_db()  # hm36
        self.joint_num = database[1].joint_num
        self.num_samples0 = len(self.db0)
        self.num_samples1 = len(self.db1)
        from_mpi_inf_3dhp_to_hm36(self.db0, use_hm_video_list=True)

        self.is_train = is_train
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.batch_size = batch_size
        self.mean = mean
        self.std = std
        self.aug_config = aug_config
        self.label_func = label_func

        self.cam_id_list = cam_id_list # in a list

        self.geodesic_pt_list = geodesic_pt_list
        self.geodesic_param_list = geodesic_param_list
        self.rm_bg = rm_bg

        self.smpl_flip = aug_config['smpl_flip'] if 'smpl_flip' in aug_config else 0

        if self.is_train:
            self.do_augment = True
        else:
            assert 0, "testing not supported"

        # padding samples to match input_batch_size
        extra_db = len(self.db0) % batch_size
        for i in range(0, batch_size - extra_db):
            self.db0.append(self.db0[i])
        self.db_length = len(self.db0) * 2
        assert self.db_length <= len(self.db0) + len(self.db1)

        self._count = None
        self._idx = None
        self.reset_hm36db()

    def reset_hm36db(self):
        self._count = 0
        self._idx = np.arange(self.num_samples1)
        np.random.shuffle(self._idx)

    def __getitem__(self, index):
        if index < self.num_samples0:
            select_db = self.db0[index]
        else:
            select_db = self.db1[self._idx[index - self.num_samples0]]

        out = {}
        for cam_id in self.cam_id_list:
            cam_key = 'cam_{}'.format(cam_id)
            self.generate_item(select_db[cam_key], cam_key, out)

        if self.use_smpl_pseudo_img and self.is_train:
            self.generate_pseudo_smpl_data(out)

        out['act'] = select_db['cam_0']['image'].split('/')[-1][5:21]  # e.g. 'act_04_subact_01'

        self._count = self._count + 1
        if self._count >= self.db_length:
            self.reset_hm36db()

        return out

    def __len__(self):
        return self.db_length