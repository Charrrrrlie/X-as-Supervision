import os
import numpy as np
import pickle as pk
import json
import cv2
from scipy.io import loadmat
from tqdm import tqdm

from .imdb import IMDB, patch_sample


class mpii(IMDB):

    def __init__(self, image_set_name, dataset_path, dataset_mask_path, patch_width, patch_height, extra_param, *args):
        super(mpii, self).__init__('MPII', image_set_name, dataset_path, patch_width, patch_height, dataset_path, extra_param)
        '''
        0-R_Ankle, 1-R_Knee, 2-R_Hip, 3-L_Hip, 4-L_Knee, 5-L_Ankle, 6-Pelvis, 7-Thorax,
        8-Neck, 9-Head, 10-R_Wrist, 11-R_Elbow, 12-R_Shoulder, 13-L_Shoulder, 14-L_Elbow, 15-L_Wrist
        '''
        self.joint_num = 16
        self.flip_pairs = np.array([[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]], dtype=np.int32)
        self.parent_ids = np.array([1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14], dtype=np.int32)

        self.pixel_std = 200
        self.aspect_ratio = self.patch_width * 1.0 / self.patch_height
        self.pelvis_id = 6
        self.thorax_id = 7

        self.lhip_id = 3
        self.rhip_id = 2
        self.lsho_id = 13
        self.rsho_id = 12

        self.y_move = 15
        self.scale_expand = 1.25

        self.dataset_mask_path = dataset_mask_path

    def center_and_size(self, a, jts_3d_vis):
        c = np.array(a['center'], dtype=np.float32)
        c_x = c[0]
        c_y = c[1]
        assert c_x >= 1
        if c_y < 1:
            assert sum(sum(jts_3d_vis)) == 0
        c_x = c_x - 1
        c_y = c_y - 1
        width = a['scale'] * self.pixel_std
        height = a['scale'] * self.pixel_std
        # Adjust center/scale slightly to avoid cropping limbs, this is the common practice on mpii dataset
        c_y = c_y + self.y_move * a['scale']
        width = width * self.scale_expand
        height = height * self.scale_expand
        # Adjust width to fit the required aspect_ratio, only support shrinking in width
        if width >= self.aspect_ratio * height:
            width = height * self.aspect_ratio
        else:
            assert 0, "Error. Invalid patch width and height"
        return c_x, c_y, width, height

    def remove_over_exposure(self, mask_path, ratio=0.7):
        mask = cv2.imread(mask_path)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1] / 255

        if np.sum(mask) > ratio * mask.shape[0] * mask.shape[1] or np.sum(mask) < 0.1 * mask.shape[0] * mask.shape[1]:
            return True
        return False

    def gt_db(self):

        cache_file = os.path.join(self.cache_path, self.name + '_new.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))
            return db

        # create train/val split
        with open(os.path.join(self.dataset_path, 'annot', 'mpii_'+ self.image_set_name+'.json')) as anno_file:
            anno = json.load(anno_file)

        SC_BIAS = 0.6

        gt_mat = loadmat(os.path.join(self.dataset_path, 'annot', f'mpii_gt_{self.image_set_name}.mat'))
        headboxes_src = gt_mat['headboxes_src']

        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :] 
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS

        gt_db = []
        for i, a in enumerate(tqdm(anno, total=len(anno), leave=True)):
            # joints and vis
            jts_3d = np.zeros((self.joint_num, 3), dtype=np.float32)
            jts_3d_vis = np.zeros((self.joint_num, 1), dtype=np.float32)
            if self.image_set_name != 'test':
                jts = np.array(a['joints'])
                jts[:, 0:2] = jts[:, 0:2] - 1
                jts_vis = np.array(a['joints_vis'])
                assert len(jts) == self.joint_num, 'joint num diff: {} vs {}'.format(len(jts), self.joint_num)
                jts_3d[:, 0:2] = jts[:, 0:2]
                jts_3d_vis[:, 0] = jts_vis[:]

            c_x, c_y, width, height = self.center_and_size(a, jts_3d_vis)
            rot = 0

            img_path = os.path.join(self.dataset_path, 'images', a['image'])

            mask_path = os.path.join(self.dataset_mask_path, a['image'])

            if len(jts_3d_vis) < np.sum(jts_3d_vis) or self.remove_over_exposure(mask_path) or jts_3d.min() < 0:
                continue

            smp = patch_sample(img_path, c_x, c_y, width, height, rot, jts_3d, jts_3d_vis, self.flip_pairs,
                               self.parent_ids)
            smp.head_size = headsizes[i]
            smp.mask = mask_path
            # keep same format
            gt_db.append({'cam_mono': smp})

        with open(cache_file, 'wb') as fid:
            pk.dump(gt_db, fid, pk.HIGHEST_PROTOCOL)
        print('{} samples ared wrote {}'.format(len(gt_db), cache_file))

        return gt_db