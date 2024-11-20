from __future__ import print_function

import os
import numpy as np
import pickle as pk

from human_utils.common.utility.camProject import CamProj, CamBackProj

from .imdb import IMDB, patch_sample

s_hm36_subject_num = 7
HM_subject_idx = [ 1, 5, 6, 7, 8, 9, 11 ]
HM_subject_idx_inv = [ -1, 0, -1, -1, -1, 1, 2, 3, 4, 5, -1, 6 ]

s_hm36_act_num = 15
HM_act_idx = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ]
HM_act_idx_inv = [ -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ]

s_hm36_subact_num = 2
HM_subact_idx = [ 1, 2 ]
HM_subact_idx_inv = [ -1, 0, 1 ]

s_hm36_camera_num = 4
HM_camera_idx = [ 1, 2, 3, 4 ]
HM_camera_idx_inv = [ -1, 0, 1, 2, 3 ]

# 17 joints of Human3.6M:
# 'root', 'Rleg0', 'Rleg1', 'Rleg2', 'Lleg0', 'Lleg1', 'Lleg2', 'torso', 'neck', 'nose', 'head', 'Larm0', 'Larm1', 'Larm2', 'Rarm0', 'Rarm1', 'Rarm2'
# 'root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'

# 18 joints with Thorax:
# 'root', 'Rleg0', 'Rleg1', 'Rleg2', 'Lleg0', 'Lleg1', 'Lleg2', 'torso', 'neck', 'nose', 'head', 'Larm0', 'Larm1', 'Larm2', 'Rarm0', 'Rarm1', 'Rarm2', 'Thorax'
# 'root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist, 'Thorax''
# 0       1       2        3         4       5        6         7        8       9       10      11           12        13        14           15        16       17
# [ 0,      0,      1,       2,        0,      4,       5,        0,      17,      17,      8,     17,           11,        12,       17,          14,       15,      0]
JointName = ['root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head',
             'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist', 'Thorax']

# 16 joints of MPII
# 0-R_Ankle, 1-R_Knee, 2-R_Hip, 3-L_Hip, 4-L_Knee, 5-L_Ankle, 6-Pelvis, 7-Thorax,
# 8-Neck, 9-Head, 10-R_Wrist, 11-R_Elbow, 12-R_Shoulder, 13-L_Shoulder, 14-L_Elbow, 15-L_Wrist

s_org_36_jt_num = 32
s_36_root_jt_idx = 0
s_36_lsh_jt_idx = 11
s_36_rsh_jt_idx = 14
s_36_jt_num = 18
s_36_flip_pairs = np.array([[1, 4], [2, 5], [3, 6], [14, 11], [15, 12], [16, 13]], dtype=np.int32)
s_36_parent_ids = np.array([0, 0, 1, 2, 0, 4, 5, 0, 17, 17, 8, 17, 11, 12, 17, 14, 15, 0], dtype=np.int32)
s_36_bone_jts = np.array([[0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                          [8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6]])
s_mpii_2_hm36_jt = [6, 2, 1, 0, 3, 4, 5, -1, 8, -1, 9, 13, 14, 15, 12, 11, 10, 7]
s_hm36_2_mpii_jt = [3, 2, 1, 4, 5, 6, 0, 17, 8, 10, 16, 15, 14, 11, 12, 13]

s_coco_2_hm36_jt = [-1, 12, 14, 16, 11, 13, 15, -1, -1, 0, -1, 5, 7, 9, 6, 8, 10, -1]

s_posetrack_2_hm36_jt = [-1, 2, 1, 0, 3, 4, 5, -1, 12, 13, 14, 9, 10, 11, 8, 7, 6, -1]


def parsing_hm36_gt_file(gt_file, ignore_jt_list=False):
    keypoints = []
    with open(gt_file, 'r') as f:
        content = f.read()
        content = content.split('\n')
        image_num = int(float(content[0]))
        img_width = content[1].split(' ')[1]
        img_height = content[1].split(' ')[2]
        rot = content[2].split(' ')[1:10]
        trans = content[3].split(' ')[1:4]
        fl = content[4].split(' ')[1:3]
        c_p = content[5].split(' ')[1:3]
        k_p = content[6].split(' ')[1:4]
        p_p = content[7].split(' ')[1:3]
        jt_list = content[8].split(' ')[1:18]
        for i in range(0, image_num):
            keypoints.append(content[9 + i].split(' ')[1:97])

    keypoints = np.asarray([[float(y) for y in x] for x in keypoints])
    keypoints = keypoints.reshape(keypoints.shape[0], keypoints.shape[1] // 3, 3)
    trans = np.asarray([float(y) for y in trans])
    jt_list = np.asarray([int(y) for y in jt_list])

    # load all landmarks
    if not ignore_jt_list:
        keypoints = keypoints[:, jt_list - 1, :]

        # add thorax
        thorax = (keypoints[:, s_36_lsh_jt_idx, :] + keypoints[:, s_36_rsh_jt_idx, :]) * 0.5
        thorax = thorax.reshape((thorax.shape[0], 1, thorax.shape[1]))
        keypoints = np.concatenate((keypoints, thorax), axis=1)

    rot = np.asarray([float(y) for y in rot]).reshape((3,3))
    rot = np.transpose(rot)
    fl = np.asarray([float(y) for y in fl])
    c_p = np.asarray([float(y) for y in c_p])
    img_width = np.asarray(float(img_width))
    img_height = np.asarray(float(img_height))
    return keypoints, trans, jt_list, rot, fl, c_p, img_width, img_height


def joint_to_bone_mat(parent_ids):
    joint_num = len(parent_ids)
    mat = np.zeros((joint_num, joint_num), dtype=int)
    for i in range(0, joint_num):
        p_i = parent_ids[i]
        if p_i != i:
            mat[i][p_i] = -1
            mat[i][i] = 1
        else:
            mat[i][i] = 1
    return np.transpose(mat)


def joint_to_full_pair_mat(joint_num):
    mat = np.zeros((joint_num * (joint_num - 1) / 2, joint_num), dtype=int)
    idx = 0
    for i in range(0, joint_num):
        for j in range(0, joint_num):
            if j > i:
                mat[idx][i] = 1
                mat[idx][j] = -1
                idx = idx + 1
    return np.transpose(mat)


def convert_joint(jts, vis, mat):
    cvt_jts = np.zeros((mat.shape[1]) * 3, dtype = float)
    cvt_jts[0::3] = np.dot(jts[0::3], mat)
    cvt_jts[1::3] = np.dot(jts[1::3], mat)
    cvt_jts[2::3] = np.dot(jts[2::3], mat)

    vis_mat = mat.copy()
    vis_mat[vis_mat!=0] = 1
    cvt_vis = np.zeros((mat.shape[1]) * 3, dtype = float)

    s = np.sum(vis_mat, axis=0)

    cvt_vis[0::3] = np.dot(vis[0::3], vis_mat) == s
    cvt_vis[1::3] = np.dot(vis[1::3], vis_mat) == s
    cvt_vis[2::3] = np.dot(vis[2::3], vis_mat) == s
    return cvt_jts, cvt_vis


def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t


def rigid_align(A, B):
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2


def from_worldjt_to_imagejt(n_img, joint_num, rot, keypoints, trans, fl, c_p, rect_3d_width, rect_3d_height):
    # project to image space
    pt_3d = np.zeros((joint_num, 3), dtype=np.float32)
    pt_2d = np.zeros((joint_num, 3), dtype=np.float32)
    for n_jt in range(0, joint_num):
        pt_3d[n_jt] = np.dot(rot, keypoints[n_img, n_jt] - trans)
        pt_2d[n_jt, 0], pt_2d[n_jt, 1] = CamProj(pt_3d[n_jt, 0], pt_3d[n_jt, 1], pt_3d[n_jt, 2], fl[0], fl[1],
                                                 c_p[0], c_p[1])
        pt_2d[n_jt, 2] = pt_3d[n_jt, 2]

    pelvis3d = pt_3d[s_36_root_jt_idx]
    # build 3D bounding box centered on pelvis, size 2000^2
    rect3d_lt = pelvis3d - [rect_3d_width / 2, rect_3d_height / 2, 0]
    rect3d_rb = pelvis3d + [rect_3d_width / 2, rect_3d_height / 2, 0]
    # back-project 3D BBox to 2D image
    rect2d_l, rect2d_t = CamProj(rect3d_lt[0], rect3d_lt[1], rect3d_lt[2], fl[0], fl[1], c_p[0], c_p[1])
    rect2d_r, rect2d_b = CamProj(rect3d_rb[0], rect3d_rb[1], rect3d_rb[2], fl[0], fl[1], c_p[0], c_p[1])

    # Subtract pelvis depth
    pt_2d[:, 2] = pt_2d[:, 2] - pelvis3d[2]
    pt_2d = pt_2d.reshape((joint_num, 3))
    vis = np.ones((joint_num, 1), dtype=np.float32)

    return rect2d_l, rect2d_r, rect2d_t, rect2d_b, pt_2d, pt_3d, vis, pelvis3d


def _H36FolderName(subject_id, act_id, subact_id):
    return "s_%02d_act_%02d_subact_%02d" % \
           (HM_subject_idx[subject_id], HM_act_idx[act_id], HM_subact_idx[subact_id])


def _H36ImageName(folder_name, frame_id):
    return "%s_%06d.jpg" % (folder_name, frame_id + 1)


def _AllHuman36Folders(subject_list_, cam_num=None):
    subject_list = subject_list_[:]
    if len(subject_list) == 0:
        for i in range(0, s_hm36_subject_num):
            subject_list.append(i)
    folders = []
    for i in range(0, len(subject_list)):
        for j in range(0, s_hm36_act_num):
            for m in range(0, s_hm36_subact_num):                
                folders.append(_H36FolderName(subject_list[i], j, m))
    return folders


def _sample_dataset(image_set_name):
    # divided by 4 cameras
    if image_set_name == 'train':
        sample_num = 200
        step = -1
        folder_start = 0
        folder_end = 150
        folders = _AllHuman36Folders([0, 1, 2, 3, 4])
    elif image_set_name == 'trainfull':
        sample_num = -1
        step = 1
        folder_start = 0
        folder_end = 150
        folders = _AllHuman36Folders([0, 1, 2, 3, 4])
    elif image_set_name == 'trainselect':
        sample_num = 0
        step = 10
        folder_start = 0
        folder_end = 150
        folders = _AllHuman36Folders([0, 1, 2, 3, 4])
    elif image_set_name == 'train_selected':
        sample_num = 200
        step = -1
        folder_start = 0
        folder_end = 150
        folders = _AllHuman36Folders([0, 1, 2, 3, 4])
    elif image_set_name == 'valid':
        sample_num = 40
        step = -1
        folder_start = 0
        folder_end = 60
        folders = _AllHuman36Folders([5, 6])
    elif image_set_name == 'validlarge':
        sample_num = 150
        step = 1
        folder_start = 0
        folder_end = 60
        folders = _AllHuman36Folders([5, 6])
    elif image_set_name == 'validfull':
        sample_num = -1
        step = 1
        folder_start = 0
        folder_end = 60
        folders = _AllHuman36Folders([5, 6])
    else:
        print("Error!!!!!!!!! Unknown hm36 sub set!")
        assert 0
    return folders, sample_num, step, folder_start, folder_end


class hm36(IMDB):
    def __init__(self, image_set_name, dataset_path, patch_width, patch_height, rect_3d_width,
                 rect_3d_height, extra_param, init_mode=False, *args):
        super(hm36, self).__init__('HM36', image_set_name, dataset_path, patch_width, patch_height,
                                   dataset_path, extra_param)
        """
        add camera to world tranformation infos
        remove center images pairs alignment, cropped data procedure
        """
        self.joint_num = s_36_jt_num if not init_mode else s_org_36_jt_num
        self.flip_pairs = s_36_flip_pairs
        self.parent_ids = s_36_parent_ids
        self.idx2name = ['root', 'R-hip', 'R-knee', 'R-ankle', 'L-hip', 'L-knee', 'L-ankle', 'torso', 'neck', 'nose',
                         'head', 'L-shoulder', 'L-elbow', 'L-wrist', 'R-shoulder', 'R-elbow', 'R-wrist', 'thorax']

        assert rect_3d_width * patch_height == rect_3d_height * patch_width
        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.aspect_ratio = 1.0 * patch_width / patch_height

        self.num_samples_single = 0
        self.num_samples_tracking = 0

    def load_gt_image(self, n_img, n_folder, rotation, keypoints, trans, fl, c_p):

        image_name = os.path.join(n_folder, _H36ImageName(n_folder, n_img))
        i_name = os.path.join(self.dataset_path, '', 'images', image_name)

        rect2d_l, rect2d_r, rect2d_t, rect2d_b, pt_2d, pt_3d, vis, pelvis3d = \
            from_worldjt_to_imagejt(n_img, self.joint_num, rotation, keypoints, trans, fl, c_p
                                    , self.rect_3d_width, self.rect_3d_height)
        rot = 0

        # 2d joints: smp.joints_3d
        smp = patch_sample(i_name, (rect2d_l + rect2d_r) * 0.5, (rect2d_t + rect2d_b) * 0.5,
                            (rect2d_r - rect2d_l), (rect2d_b - rect2d_t), rot, pt_2d, vis, self.flip_pairs,
                            self.parent_ids)
        smp.joints_3d_cam = pt_3d
        smp.pelvis = pelvis3d
        smp.fl = fl
        smp.c_p = c_p
        smp.rot_world = rotation
        smp.trans_world = - rotation @ trans
        return smp

    def gt_db(self):
        folders, sample_num, sample_step, folder_start, folder_end = _sample_dataset(self.image_set_name)

        c_name = 'kpt'
        cache_file = os.path.join(self.cache_path, self.name + '_' + c_name + '_smp_world' + str(sample_num) + '.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))
            self.num_samples_single = len(db)
            return db

        gt_db = []
        init_cam = 0

        for n_folder in range(folder_start, folder_end):
            print('Loading folder ', n_folder, ' in ', len(folders))

            # load ground truth
            keypoints, trans, jt_list, rotation, fl, c_p = {}, {}, {}, {}, {}, {}
            for cam in range(s_hm36_camera_num):
                keypoints[cam], trans[cam], jt_list[cam], rotation[cam], fl[cam], c_p[cam], img_width, img_height  = \
                    parsing_hm36_gt_file(os.path.join(self.dataset_path, "annot", "{}_ca_{:02d}".format(folders[n_folder], HM_camera_idx[cam]),\
                                                       'matlab_meta.txt'), ignore_jt_list=False)

                assert keypoints[cam].shape[1] == self.joint_num

            # random sample redundant video sequence
            if sample_num > 0:
                img_index = np.random.choice(keypoints[init_cam].shape[0], sample_num, replace=False)
            else:
                img_index = np.arange(keypoints[init_cam].shape[0])
                img_index = img_index[0:keypoints[init_cam].shape[0]:sample_step]

            for n_img in img_index:
                # select sub-dataset
                if self.image_set_name == 'train_selected':
                    if 'act_04' in folders[n_folder] or 'act_06' in folders[n_folder] or 'act_09' in folders[n_folder]\
                        or 'act_11' in folders[n_folder]:
                        continue

                smp_dict = {}
                for cam in range(s_hm36_camera_num):
                    smp = self.load_gt_image(n_img, "{}_ca_{:02d}".format(folders[n_folder], HM_camera_idx[cam]), rotation[cam], keypoints[cam], trans[cam], fl[cam], c_p[cam])
                    smp_dict['cam_{}'.format(cam)] = smp
                gt_db.append(smp_dict)

        with open(cache_file, 'wb') as fid:
            pk.dump(gt_db, fid, pk.HIGHEST_PROTOCOL)
        print('{} samples are wrote {}'.format(len(gt_db), cache_file))

        self.num_samples_single = len(gt_db)

        return gt_db