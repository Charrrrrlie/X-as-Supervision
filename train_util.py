import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from easydict import EasyDict as edict

from modules.base_losses.integral import get_label_func
from human_utils.common.visualization.pose import cv_draw_joints, cv_draw_joints_parent, plot_3d_skeleton

import human_utils.dataset as human_dataset
from human_utils.dataloader.dataloader import hm36_Dataset, mpi_inf_3dhp_Dataset, mpi_inf_3dhp_hm36_Dataset


def basic_data(config, eval_only=False):
    # load gt data but only for eval and check
    label_func = get_label_func()
    batch_size =  config['train_params']['batch_size']
    dataset_param = config['dataset_params']
    train_param = config['train_params']

    use_full_kp = dataset_param['use_full_kp'] if 'use_full_kp' in dataset_param else False
    cam_id_list = dataset_param['cam_id_list']
    geodesic_pt_list = dataset_param['geodesic_pt_list'] if 'geodesic_pt_list' in dataset_param else [0]
    geodesic_param_list = dataset_param['geodesic_param_list'] if 'geodesic_param_list' in dataset_param else [2.0, 1.0, 2.0, 1.0, 0.0]
    rm_bg= dataset_param['rm_bg'] if 'rm_bg' in dataset_param else False
    smpl_pseudo_img = dataset_param['smpl_pseudo_img'] if 'smpl_pseudo_img' in dataset_param else None

    convert_to_17kps = True if config['dataset_params']['dataset']['name'] == 'mpi_inf_3dhp' else False

    train_imdbs, valid_imdbs = [], []
    # TODO(yyc): change dict to EasyDict form
    train_param['aug'] = edict(train_param['aug'])

    if not eval_only:
        if '+' in dataset_param['dataset']['name']:
            dataset_name_list = dataset_param['dataset']['name'].split('+')
            dataset_param['dataset']['name'] = dataset_param['dataset']['name'].replace('+', '_')
            for dataset_name in dataset_name_list:
                train_imdbs.append(getattr(human_dataset,dataset_name)(
                                        dataset_param['dataset'][dataset_name]['train_image_set'],
                                        dataset_param['dataset'][dataset_name]['path'],
                                        train_param['patch_width'],
                                        train_param['patch_height'],
                                        train_param['rect_3d_width'],
                                        train_param['rect_3d_height'],
                                        dataset_param['dataset'][dataset_name]['extra_param'],
                                        init_mode=use_full_kp))
        else:
            train_imdbs.append(getattr(human_dataset,dataset_param['dataset']['name'])(
                                            dataset_param['dataset']['train_image_set'],
                                            dataset_param['dataset']['path'],
                                            train_param['patch_width'],
                                            train_param['patch_height'],
                                            train_param['rect_3d_width'],
                                            train_param['rect_3d_height'],
                                            dataset_param['dataset']['extra_param'],
                                            init_mode=use_full_kp))
        dataset_train = eval(dataset_param['dataset']['name'] + "_Dataset")(
                        train_imdbs,
                        True,
                        train_param['patch_width'],
                        train_param['patch_height'],
                        train_param['rect_3d_width'],
                        train_param['rect_3d_height'],
                        batch_size, dataset_param['dataiter']['mean'],
                        dataset_param['dataiter']['std'],
                        train_param['aug'],
                        label_func,cam_id_list,
                        geodesic_pt_list,
                        geodesic_param_list,
                        smpl_pseudo_img,
                        rm_bg)
        return dataset_train

    else:
        valid_imdbs.append(getattr(human_dataset,dataset_param['dataset']['name'])(
                                    dataset_param['dataset']['test_image_set'],
                                    dataset_param['dataset']['path'],
                                    train_param['patch_width'],
                                    train_param['patch_height'],
                                    train_param['rect_3d_width'],
                                    train_param['rect_3d_height'],
                                    dataset_param['dataset']['extra_param'],
                                    init_mode=use_full_kp))

        dataset_valid = eval(dataset_param['dataset']['name'] + "_Dataset")(
                        valid_imdbs,
                        False,
                        train_param['patch_width'],
                        train_param['patch_height'],
                        train_param['rect_3d_width'],
                        train_param['rect_3d_height'],
                        batch_size,
                        dataset_param['dataiter']['mean'],
                        dataset_param['dataiter']['std'],
                        train_param['aug'],
                        label_func,
                        cam_id_list,
                        geodesic_pt_list,
                        geodesic_param_list,
                        None,
                        rm_bg,
                        convert_to_17kps)
        return dataset_valid

def pose_vis(pose, size, flip_pairs, parent_ids=None, is_gt=False, img=None, mean=None, std=None):
    if not is_gt:
        pose = (pose + 1) / 2.0
        pose[:, 0] *= (size[0] - 1)
        pose[:, 1] *= (size[1] - 1)

    if img is None:
        img = np.ones([size[0], size[1], 3]) * 255.0
    else:
        if isinstance(img, torch.Tensor):
            if mean is not None and std is not None:
                for i in range(len(img)):
                    img[i, :, :] = img[i, :, :] * std[i] + mean[i]

            img = img.detach().cpu().numpy().copy().transpose([1, 2, 0])

        if np.max(img) < 128:
            img = img * 255.0
        img = np.ascontiguousarray(img, dtype=np.uint8)

    vis = np.ones(pose.shape)

    if parent_ids is not None:
        cv_draw_joints_parent(img, pose, vis, parent_ids)

    if np.max(flip_pairs) >= pose.shape[0]:
        flip_pairs = None
    cv_draw_joints(img, pose, vis, flip_pairs)

    img = np.uint8(img)
    return img.transpose([2, 0, 1]) # (C, H, W)

def pose_vis_3d(keypoints_3d, flip_pairs, parent_ids=None, ref_keypoints=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_skeleton(ax, keypoints_3d, parent_ids, flip_pairs, 256, 256)
    if ref_keypoints is not None:
        plot_3d_skeleton(ax, ref_keypoints, parent_ids, flip_pairs, 256, 256, c0='k')

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    width, height = fig.get_size_inches() * fig.get_dpi()
    plt.close()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 3)
    
    return img.transpose([2, 0, 1]) # (C, H, W)

def img_vis(img, mean=None, std=None):
    if isinstance(img, torch.Tensor):
        img = img.detach().data.cpu().numpy().copy()

    if mean and std:
        for i in range(len(img)):
            img[i, :, :] = img[i, :, :] * std[i] + mean[i]

    if np.max(img) < 128:
        img = img * 255

    return np.uint8(img)

def dis_vis(distance, centers):
    '''
    distance: (1, H, W)
    center: (2, )
    '''
    fig = plt.figure()
    plt.imshow(distance[0], interpolation='nearest')
    for center in centers:
        plt.scatter(center[0], center[1], c='r', s=5)

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    width, height = fig.get_size_inches() * fig.get_dpi()
    plt.close()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 3)

    return img.transpose([2, 0, 1])

def depth_heatmap_vis(depth_map, gt_pose_2d, depth_scale=256, heat_w=6, heat_h=6):
    K, H = depth_map.shape

    gt_depth = gt_pose_2d[:, [2]]
    gt_depth = ((gt_depth / depth_scale) + 1) / 2 # [0, 1]

    gt_depth = np.clip(gt_depth, 0, 1)
    gt_depth = gt_depth * H

    cmap = mcolors.ListedColormap(['white', 'red'])
    bounds = [0, 1, 2]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=heat_h, ncols=heat_w, figsize=(10, 4))

    for i in range(K):
        line_data = depth_map[[i]]
        location = gt_depth[i, 0]

        line_data = np.tile(line_data, (10, 1))

        mask = np.zeros_like(line_data)
        mask[:, int(location)] = 1.0
        mask[:, max(0, int(location) - 1)] = 1.0

        axes[i // heat_w * 2, i % heat_w].imshow(line_data, cmap='Reds')
        axes[i // heat_w * 2, i % heat_w].set_xticks([])
        axes[i // heat_w * 2, i % heat_w].set_yticks([])

        axes[(i // heat_w) * 2 + 1, i % heat_w].imshow(mask, cmap=cmap, norm=norm)
        axes[(i // heat_w) * 2 + 1, i % heat_w].set_xticks([])
        axes[(i // heat_w) * 2 + 1, i % heat_w].set_yticks([])

    plt.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    width, height = fig.get_size_inches() * fig.get_dpi()
    plt.close()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(int(height), int(width), 3)
    return img.transpose([2, 0, 1])

def tb_vis(tb_log, cur_step, tb_pair_ids, tb_parent_ids, total_loss, loss_kp, loss_disc, output, x, config, scheduler_detector,\
           simple_version=False):

    if not simple_version:
        if total_loss is not None:
            tb_log.add_scalar('training_loss/total_loss', total_loss, cur_step)
        for key, value in loss_kp.items():
            tb_log.add_scalar('training_loss/{}'.format(key), \
                                value.mean().detach().data.cpu().numpy(), cur_step)

        tb_log.add_scalar('meta/learning_rate/detector', float(scheduler_detector.get_last_lr()[0]), cur_step)
        for key in output.keys():
            if key.startswith('line_width'):
                    for i, val in enumerate(output[key]):
                        tb_log.add_scalar('training_line_width/{}_{}'.format(key, i), val.detach().data.cpu().numpy(), cur_step)

        if loss_disc is not None:
            tb_log.add_scalar('training_loss/smpl_disc', loss_disc.detach().data.cpu().numpy(), cur_step)

    if cur_step % 50 == 0:
        if 'dataiter' in config['dataset_params']:
            mean, std = config['dataset_params']['dataiter']['mean'], config['dataset_params']['dataiter']['std']
        else:
            mean, std = None, None

        if not simple_version:
            tb_log.add_text('training_img/file_name', '{}'.format(x['cam_0_img_path'][0]), cur_step)
        else:
            tb_log.add_text('training_img/file_name_2d', '{}'.format(x['cam_mono_img_path'][0]), cur_step)

        for key in x.keys():
            if 'pseudo' in key:
                continue
            if key.endswith('img'):
                tb_log.add_image('training_img/{}'.format(key), img_vis(x[key][0, :, :, :], mean=mean, std=std), cur_step)
            elif key.endswith('mask'):
                tb_log.add_image('training_mask/{}'.format(key), img_vis(x[key][0, :, :,:]), cur_step)
            elif key.endswith('joints'):
                cam_key = key.split('_joints')[0]
                tb_log.add_image('training_pose_2d/{}_gt_pose'.format(cam_key), pose_vis(x[key][0, :, :2]\
                            .detach().data.cpu().numpy(), x['{}_img'.format(cam_key)].shape[2:4], tb_pair_ids, tb_parent_ids,\
                            img=x['{}_img'.format(cam_key)][0, :, :, :].clone(), mean=mean, std=std, is_gt=True), cur_step)
            elif key.endswith('geodesic_dis'):
                cam_key = key.split('_geodesic_dis')[0]
                geo_center = x['{}_geodesic_center'.format(cam_key)][0, ...].detach().data.cpu().numpy()
                geo_dis = x[key][0, ...].detach().data.cpu().numpy()
                tb_log.add_image('training_weight/{}'.format(key), dis_vis(geo_dis, geo_center), cur_step)

        for key in output.keys():
            if key.startswith('mask'):
                tb_log.add_image('training_mask/{}'.format(key), img_vis(output[key][0, :, :,:]), cur_step)
            elif key.startswith('pose_2d'):
                mode = key.split('pose_2d_pred_')[1].split('_ori')[0]
                page = 'training_pose_2d' if not 'pseudo' in key else 'training_pseudo'
                tb_log.add_image('{}/{}'.format(page, key), pose_vis(output[key][0, :, :2]\
                            .detach().data.cpu().numpy(), x['{}_img'.format(mode)].shape[2:4], tb_pair_ids, tb_parent_ids,\
                            img=x['{}_img'.format(mode)][0, :, :, :].clone(), mean=mean, std=std), cur_step)
            elif key.startswith('pose_3d'):
                page = 'training_pose_3d' if not 'pseudo' in key else 'training_pseudo'
                tb_log.add_image('{}/{}'.format(page, key), pose_vis_3d(output[key][0]\
                            .detach().data.cpu().numpy(), tb_pair_ids, tb_parent_ids), cur_step)
            elif key.startswith('pose_smpl_2d') and not simple_version:
                tb_log.add_image('training_smpl/{}'.format(key), pose_vis(output[key][0, :, :2]\
                            .detach().data.cpu().numpy(), x['cam_0_img'].shape[2:4], tb_pair_ids, tb_parent_ids), cur_step)
            elif key.startswith('pose_smpl_3d') and not simple_version:
                tb_log.add_image('training_smpl/{}'.format(key), pose_vis_3d(output[key][0]\
                            .detach().data.cpu().numpy(), tb_pair_ids, tb_parent_ids), cur_step)
            elif key.startswith('depth_map') and not simple_version:
                mode = key.split('depth_map_')[1]
                tb_log.add_image('training_depth/{}'.format(key), depth_heatmap_vis(output[key]\
                            .detach().data.cpu().numpy(), x['{}_joints'.format(mode)][0].detach().data.cpu().numpy()), cur_step)
            elif 'logits' in key and not simple_version:
                tb_log.add_scalar('training_disc/{}'.format(key), output[key][0, ...].detach().cpu().numpy(), cur_step)

        if 'kp_gt_world' in output.keys():
            tb_log.add_image('training_pose_3d/src_gt_pose_3d', pose_vis_3d(output['kp_gt_world'][0]\
                            .detach().data.cpu().numpy(), tb_pair_ids, tb_parent_ids), cur_step)