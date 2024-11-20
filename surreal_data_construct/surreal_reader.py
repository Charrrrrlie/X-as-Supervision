# The code is partly from 
# https://github.com/gulvarol/surreal/blob/master/datageneration/misc/smpl_relations/smpl_relations.py

import os
import cv2
import random

import numpy as np
import scipy.io as sio

from threading import Thread
from queue import Queue

# NOTE(yyc): download from smplify repo
from smpl_webuser.serialization import load_model
from surreal_utils import get_intrinsic, get_frame, get_mask, filter_incorrect_cases
from reader_utils import load_info_npy, sample_time_idx, vis_check, construct_dataset, crop_and_resize

# NOTE(yyc): modify smpl kintree
HM36_KINTREE_TABLE = np.array(
    [
        [0, 0, 1, 2, 0, 4, 5, 0, 17, 8, 9, 17, 11, 12, 17, 14, 15, 7],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15, 16, 17]
    ]
)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def process(q, results, margin_x, sample_times, intrinsic, h36m_regressor, target_x_px, target_y_px, smpl_model_path, output_path, ignore_center_filter):
    while True:
        idx, info_path = q.get()
        if info_path is None:
            q.task_done()
            print('All tasks done')
            break
        try:
            info = sio.loadmat(info_path)
            cap = cv2.VideoCapture(info_path.replace('_info.mat', '.mp4'))
            mask_mat = sio.loadmat(info_path.replace('_info.mat', '_segm.mat'))
        except:
            q.task_done()
            continue

        time_list = sample_time_idx(info, margin_x, sample_times, ignore_center=ignore_center_filter)

        if info['gender'].ndim == 2:
            info['gender'] = info['gender'][0]

        for ii, t in enumerate(time_list):
            if info['gender'][0] == 0:  # f
                m = load_model(os.path.join(smpl_model_path, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl'))
            elif info['gender'][0] == 1:  # m
                m = load_model(os.path.join(smpl_model_path, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'))

            # image coords
            joints3D, vertices, box_lt, box_rb = construct_dataset(info, m, t, intrinsic, h36m_regressor)
            rgb = get_frame(cap, t)

            if 'segm_{}'.format(t + 1) not in mask_mat:
                continue

            mask = get_mask(mask_mat, t)

            rgb, mask, joints3D, vertices = crop_and_resize(rgb, mask, joints3D, vertices, box_lt, box_rb, target_x_px, target_y_px)

            joints3D[..., 0] = np.clip(joints3D[..., 0], 0, mask.shape[1] - 1)
            joints3D[..., 1] = np.clip(joints3D[..., 1], 0, mask.shape[0] - 1)

            if filter_incorrect_cases(mask, joints3D[..., :2]) == -1:
                continue

            cv2.imwrite(os.path.join(output_path, 'image', 'image_{:06d}.png'.format(idx * sample_times + ii)), rgb)
            cv2.imwrite(os.path.join(output_path, 'mask', 'mask_{:06d}.png'.format(idx * sample_times + ii)), mask)

            if (idx * sample_times + ii) % 1000 == 0:
                check_img_path = os.path.join(output_path, 'check_image', 'check_{:06d}.png'.format(idx * sample_times + ii))
                vis_check(rgb, mask, joints3D[..., :2], vertices[..., :2], check_img_path , HM36_KINTREE_TABLE)

            joints3D[..., 0] = joints3D[..., 0] / target_x_px * 2 - 1
            joints3D[..., 1] = joints3D[..., 1] / target_y_px * 2 - 1

            np.save(os.path.join(output_path, 'joints', 'joint_{:06d}.npy'.format(idx * sample_times + ii)), joints3D)

            results.append(idx * sample_times + ii)

        q.task_done()

if __name__ == '__main__':

    smpl_model_path = '../data/smpl_models'
    h36m_regressor = np.load(os.path.join(smpl_model_path, 'J_regressor_h36m.npy'))

    # NOTE: surreal to h36m
    # output_path = 'data/surreal_h36m_pose'
    # root_path = '../data/surreal/'
    # mode = 'train'

    # res_x_px = 320
    # res_y_px = 240

    # ignore_center_filter = False

    # NOTE: pseudo to h36m
    output_path = 'data/surreal_h36m_pose_pseudo'
    root_path = '../data/surreal_pseudo/'
    mode = 'train'
    ignore_center_filter = True

    res_x_px = 512
    res_y_px = 512

    ########################################
    intrinsic = get_intrinsic(res_x_px, res_y_px)

    margin_x = (res_x_px - res_y_px) // 2

    target_x_px = 256
    target_y_px = 256

    assert target_x_px == target_y_px

    info = load_info_npy(root_path, mode) # 29783 for training

    set_seed(42)
    sample_num = min(20000, len(info))
    sample_times = 5
    sampled_info = np.random.choice(info, sample_num, replace=False)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, 'image'))
        os.mkdir(os.path.join(output_path, 'mask'))
        os.mkdir(os.path.join(output_path, 'joints'))
        os.mkdir(os.path.join(output_path, 'check_image'))

    num_worker_threads = 12
    q = Queue()
    threads = []
    idx_list = []

    for i in range(num_worker_threads):
        t = Thread(target=process, args=(q, idx_list, margin_x, sample_times, intrinsic, h36m_regressor,
                                         target_x_px, target_y_px, smpl_model_path, output_path, ignore_center_filter))
        t.start()
        threads.append(t)

    for idx, info_path in enumerate(sampled_info):
        q.put((idx, info_path))

    q.join()

    # stop workers
    for i in range(num_worker_threads):
        q.put((-1, None))
    for t in threads:
        t.join()

    idx_list = sorted(idx_list)
    np.save(os.path.join(output_path, 'info.npy'), idx_list)
    print('Total {} items constructed under {} samples'.format(len(idx_list), sample_num * sample_times))