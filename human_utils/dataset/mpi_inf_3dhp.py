import os
import h5py
import glob
import cv2
import numpy as np
import pickle as pk
from scipy.io import loadmat

from tqdm import tqdm

from human_utils.common.utility.camProject import CamProj, CamBackProj

from .imdb import IMDB, patch_sample

s_mpi_subject_num = 8
mpi_subject_idx = [1, 2, 3, 4, 5, 6, 7, 8]

mpi_seq_num = 2
mpi_seq_idx = [1, 2]

total_mpi_video_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
total_mpi_video_num = 14
use_mpi_video_idx = [0, 2, 4, 7, 8]  # chest height

mpi_train_subject = [1,2,3,4,5,6]
mpi_valid_subject = [7,8]

train_joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',  # 0 ~ 4
                     'neck', 'head', 'head_top',  # 5 ~ 7
                     'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand',  # 8 ~ 12
                     'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand',  # 13 ~17
                     'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',  # 18 ~ 22
                     'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe'  # 23 ~ 27
                     ]

test_joint_names = ['head_top', 'neck',  # 0, 1
                    'right_shoulder', 'right_elbow', 'right_wrist',  # 2, 3, 4
                    'left_shoulder', 'left_elbow', 'left_wrist',  # 5, 6, 7
                    'right_hip', 'right_knee', 'right_ankle',  # 8, 9, 10
                    'left_hip', 'left_knee', 'left_ankle',  # 11, 12, 13
                    'pelvis', 'spine', 'head']  # 14, 15, 16

mpi_lsh_jt_idx = 9
mpi_rsh_jt_idx = 14
mpi_train_root_jt_idx = 4
mpi_jt_num = 28

mpi_flip_pairs = np.array(
    [[8, 13], [9, 14], [10, 15], [11, 16], [12, 17], [18, 23], [19, 24], [20, 25], [21, 26], [22, 27]], dtype=np.int32)
mpi_parent_ids = np.array(
    [0, 0, 0, 2, 3, 1, 5, 6, 5, 8, 9, 10, 11, 5, 13, 14, 15, 16, 4, 18, 19, 20, 21, 4, 23, 24, 25, 26], dtype=np.int32)

indoor_image_resolution = [2048, 2048]
outdoor_image_resolution = [1920, 1080]


def from_mpi_inf_3dhp_to_hm36(gt_db, use_hm_video_list=False):
    select_joints = [4, 23, 24, 25, 18, 19, 20, 2, 5, 6, 7, 9, 10, 11, 14, 15, 16, 1]
    for sample_dict in gt_db:
        for video_id in use_mpi_video_idx:
            sample_dict['cam_{}'.format(video_id)].joints_3d = sample_dict['cam_{}'.format(video_id)].joints_3d[select_joints]
            sample_dict['cam_{}'.format(video_id)].joints_3d_vis = sample_dict['cam_{}'.format(video_id)].joints_3d_vis[select_joints]
            sample_dict['cam_{}'.format(video_id)].joints_3d_cam = sample_dict['cam_{}'.format(video_id)].joints_3d_cam[select_joints]

    if use_hm_video_list:
        for sample_dict in gt_db:
            sample_dict['cam_1'] = sample_dict['cam_2']
            sample_dict['cam_2'] = sample_dict['cam_4']
            sample_dict['cam_3'] = sample_dict['cam_7']

            del sample_dict['cam_4'], sample_dict['cam_7'], sample_dict['cam_8']
 
def project2image(pose_3d, rect_3d_width, rect_3d_height, camera_in_param, im_shape):
    global mpi_train_root_jt_idx

    root_idx = mpi_train_root_jt_idx

    im_width, im_height = im_shape

    fx, fy, cx, cy = camera_in_param

    pt_3d = pose_3d.copy()
    pt_2d = np.zeros(pt_3d.shape, dtype=np.float32)

    num_jt = pose_3d.shape[0]

    for n_jt in range(num_jt):
        pt_2d[n_jt, 0], pt_2d[n_jt, 1] = CamProj(pose_3d[n_jt, 0], pose_3d[n_jt, 1], pose_3d[n_jt, 2],
                                                 fx, fy, cx, cy)
        pt_2d[n_jt, 2] = pt_3d[n_jt, 2]

    pelvis3d = pt_3d[root_idx]

    # build 3D bounding box centered on pelvis, size 2000^2
    rect3d_lt = pelvis3d - [rect_3d_width / 2, rect_3d_height / 2, 0]
    rect3d_rb = pelvis3d + [rect_3d_width / 2, rect_3d_height / 2, 0]
    # back-project 3D BBox to 2D image
    rect2d_l, rect2d_t = CamProj(rect3d_lt[0], rect3d_lt[1], rect3d_lt[2], fx, fy, cx, cy)
    rect2d_r, rect2d_b = CamProj(rect3d_rb[0], rect3d_rb[1], rect3d_rb[2], fx, fy, cx, cy)

    # Subtract pelvis depth
    pt_2d[:, 2] = pt_2d[:, 2] - pelvis3d[2]
    pt_2d = pt_2d.reshape((num_jt, 3))
    vis = np.ones((num_jt, 1), dtype=np.float32)

    # check visibility
    for n_jt in range(num_jt):
        x, y, _ = pt_2d[n_jt]
        if x < 0 or y < 0 or x >= im_width or y >= im_height:
            vis[n_jt] = 0

    return rect2d_l, rect2d_r, rect2d_t, rect2d_b, pt_2d, pt_3d, vis, pelvis3d


def sample_method(image_set_name):
    sequence = []
    if image_set_name == 'train':
        sample_num = -1
        s_step = -1
        sequence = mpi_train_subject
    elif image_set_name == 'train_s5':
        sample_num = -1
        s_step = 5
        sequence = mpi_train_subject
    elif image_set_name == 'train_s10':
        sample_num = -1
        s_step = 10
        sequence = mpi_train_subject
    elif image_set_name == 'valid':
        sample_num = -1
        s_step = -1
        sequence = mpi_valid_subject
    elif image_set_name == 'valid_s10':
        sample_num = -1
        s_step = 10
        sequence = mpi_valid_subject
    else:
        assert 0

    return sample_num, s_step, sequence


class mpi_inf_3dhp(IMDB):

    def __init__(self, image_set_name, dataset_path, patch_width, patch_height, rect_3d_width,
                 rect_3d_height, extra_param, init_mode=False, *args):
        super(mpi_inf_3dhp, self).__init__('MPI_INF_3DHP', image_set_name, dataset_path, patch_width, patch_height,
                                            dataset_path, extra_param)
        self.joint_num = mpi_jt_num
        self.flip_pairs = mpi_flip_pairs
        self.parent_ids = mpi_parent_ids

        self.rect_3d_width = rect_3d_width
        self.rect_3d_height = rect_3d_height
        self.aspect_ratio = 1.0 * patch_width / patch_height

    def parse_train_camera_info(self, filepath):
        fid = open(filepath, 'r')
        camera_intr_params = [0 for _ in range(total_mpi_video_num)]
        camera_extr_params = [0 for _ in range(total_mpi_video_num)]
        while True:
            line = fid.readline()
            if not line:
                break

            if line[:4] == 'name':
                cam_id = int(line.split()[-1])

                sensor = fid.readline().strip()
                size = fid.readline().strip()
                animated = fid.readline().strip()
                in_params = fid.readline().strip()
                ex_params = fid.readline().strip()

                assert in_params[:9] == 'intrinsic'
                in_params = in_params.split()[1:]
                fx = float(in_params[0])
                cx = float(in_params[2])
                fy = float(in_params[5])
                cy = float(in_params[6])

                assert ex_params[:9] == 'extrinsic'
                ex = ex_params.split()[1:]
                ex = np.array([float(x) for x in ex]).reshape(4, 4)

                camera_intr_params[cam_id] = [fx, fy, cx, cy]
                camera_extr_params[cam_id] = ex

        return camera_intr_params, camera_extr_params

    def parsing_train_gt_file(self, folder, annotation, video_id):
        # NOTE: keys of annotation
        # '__header__', '__version__', '__globals__',
        # 'annot2', 'annot3', 'cameras', 'frames', 'univ_annot3'

        image_path = glob.glob(folder + '/*.jpg')
        num_imgs = len(image_path)

        img_list = list()
        pose_2d_list = list()
        pose_3d_list = list()

        for idx in range(num_imgs):
            img_file_path = os.path.join(folder, 'frame_%06d.jpg' % (idx + 1))
            pose_2d = annotation['annot2'][video_id, 0][idx]
            pose_3d = annotation['annot3'][video_id, 0][idx]

            # re-organize order & filter needless
            pose_2d = pose_2d.reshape((-1, 2))
            pose_3d = pose_3d.reshape((-1, 3))

            img_list.append(img_file_path)
            pose_2d_list.append(pose_2d)
            pose_3d_list.append(pose_3d)

        return img_list, pose_2d_list, pose_3d_list
    
    def remove_foreground(self, image_path, points_2d):
        chair_mask_path = image_path.replace('images', 'chair_masks')

        chair_mask = cv2.imread(chair_mask_path)[..., [2]]

        chair_mask = cv2.threshold(chair_mask, 127, 255, cv2.THRESH_BINARY)[1]
        points_2d = points_2d.astype(np.int32)

        count = 0
        for pt_2d in points_2d:
            if chair_mask[pt_2d[1], pt_2d[0]] == 0:
                count +=1

        if count > 4:
            return True
        return False

    def remove_over_exposure(self, image_path, ratio=0.85):
        mask_path = image_path.replace('images', 'masks')
        mask = cv2.imread(mask_path)[..., [2]]
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1] / 255

        if np.sum(mask) > ratio * mask.shape[0] * mask.shape[1]:
            return True
        return False

    def gt_db(self):
        sample_num, d_step, subjects = sample_method(self.image_set_name)

        cache_file = os.path.join(self.cache_path, self.name + '_smp_world' + str(sample_num) + '.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))
            self.num_sample_single = len(db)
            return db

        gt_db = []
        init_cam = 0
        for subject_id in subjects:
            for seq_id in mpi_seq_idx:
                current_root = os.path.join(self.dataset_path, 'S%d' % subject_id, 'Seq%d' % seq_id)

                annotation = loadmat(os.path.join(current_root, 'annot.mat'))
                cam_intr_params, cam_extr_params = self.parse_train_camera_info(os.path.join(current_root, 'camera.calibration'))

                img_dict, pose_2d_dict, pose_3d_dict = {}, {}, {}
                for video_id in use_mpi_video_idx:
                    folder = os.path.join(current_root, 'images', 'video_%d' % video_id)
                    img_dict[video_id], pose_2d_dict[video_id], pose_3d_dict[video_id] = \
                        self.parsing_train_gt_file(folder, annotation, video_id)

                sample_img_idx = np.arange(len(img_dict[init_cam]))
                if sample_num > 0:
                    sample_img_idx = np.random.choice(sample_img_idx, sample_num, replace=False)
                elif d_step > 0:
                    sample_img_idx = np.arange(len(sample_img_idx), step=d_step)

                for n_img in tqdm(sample_img_idx, leave=True, \
                                  total=len(sample_img_idx), desc='{}-{}'.format(subject_id, seq_id)):
                    smp_dict = {}

                    vis_flag = True
                    for video_id in use_mpi_video_idx:
                        image_name = img_dict[video_id][n_img]

                        pose_3d = pose_3d_dict[video_id][n_img]
                        rect2d_l, rect2d_r, rect2d_t, rect2d_b, pt_2d, pt_3d, vis, pelvis3d = \
                            project2image(pose_3d, self.rect_3d_width, self.rect_3d_height, cam_intr_params[video_id],
                                            im_shape=indoor_image_resolution)

                        if not vis_flag or np.sum(vis) < len(vis) or self.remove_foreground(image_name, pt_2d)\
                            or self.remove_over_exposure(image_name):
                            vis_flag = False
                            break

                        fx, fy, cx, cy = cam_intr_params[video_id]

                        center_x = (rect2d_l + rect2d_r) * 0.5
                        center_y = (rect2d_t + rect2d_b) * 0.5
                        width = rect2d_r - rect2d_l
                        height = rect2d_b - rect2d_t
                        rot = 0

                        smp = patch_sample(image_name, center_x, center_y, width, height,\
                                           rot, pt_2d, vis, self.flip_pairs, self.parent_ids)
                        smp.joints_3d_cam = pt_3d
                        smp.pelvis = pelvis3d
                        smp.fl = np.array([fx, fy])
                        smp.c_p = np.array([cx, cy])
                        smp.rot_world = cam_extr_params[video_id][:3, :3]
                        smp.trans_world = cam_extr_params[video_id][:3, 3]
                        smp_dict['cam_{}'.format(video_id)] = smp

                    if vis_flag:
                        gt_db.append(smp_dict)

        with open(cache_file, 'wb') as fid:
            pk.dump(gt_db, fid, pk.HIGHEST_PROTOCOL)
        print('{} samples ared wrote {}'.format(len(gt_db), cache_file))

        self.num_sample_single = len(gt_db)

        return gt_db