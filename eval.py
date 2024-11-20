import copy
import os
import yaml
import torch

import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.tensorboard import SummaryWriter

from modules.keypoint_detector_integral import KPDetector3D as KPDetector3D_integral
from modules.keypoint_detector_integral_multi import KPDetector3DMulti as KPDetector3DMulti_integral

from train_util import basic_data, pose_vis, pose_vis_3d
from modules.util import convert_patch_to_world, triangulation

from metrics import keypoint_mpjpe, keypoint_3d_pck, keypoint_3d_auc
from eval_utils import switch_points, per_act_mse, cal_per_class_error

act = {'Directions':0.0, 'Discussion': 0.0, 'Eating': 0.0, \
        'Greeting':0.0, 'Phoning':0.0,\
         'Posing':0.0, 'Purchases':0.0, 'Sitting':0.0,\
         'SittingDown':0.0, 'Smoking':0.0, 'TakingPhoto':0.0, 'Waiting':0.0,\
         'Walking':0.0, 'WalkDog':0.0, 'WalkTogether':0.0}
act_idx_2_name = {2:'Directions', 3:'Discussion', 4:'Eating', \
        5:'Greeting', 6:'Phoning',\
        7:'Posing', 8:'Purchases', 9:'Sitting',\
        10:'SittingDown', 11:'Smoking', 12:'TakingPhoto', 13:'Waiting',\
        14:'Walking', 15:'WalkDog', 16:'WalkTogether'}

def update_dict(record_table, count_table, error, act):
    for i, act_item in enumerate(act):
        act_num = int(act_item[4:6])
        record_table[act_idx_2_name[act_num]] +=error[i]
        count_table[act_idx_2_name[act_num]] +=1

def update_dict_3d(kps_world_pred_list, kps_world_gt, vis_mask, record_table, count_table, act):
    for kps_world_pred in kps_world_pred_list:
        for metric, alignment in zip(['mpjpe', 'n-mpjpe', 'p-mpjpe'], ['none', 'scale', 'procrustes']):
            error_3d = np.mean(keypoint_mpjpe(kps_world_pred, kps_world_gt, vis_mask, alignment=alignment), axis=1)
            if cal_per_act:
                update_dict(record_table[metric], count_table[metric], error_3d, act)
            else:
                record_table[metric] += error_3d
                count_table[metric] += 1

        if not cal_per_act:
            record_table['pck'] += keypoint_3d_pck(kps_world_pred / 1000.0 , kps_world_gt / 1000.0, vis_mask).mean()
            record_table['auc'] += keypoint_3d_auc(kps_world_pred / 1000.0 , kps_world_gt / 1000.0, vis_mask)
            count_table['pck'] += 1
            count_table['auc'] += 1

    return record_table, count_table

def ddp_setup():
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

class Eval:
    def __init__(
        self,
        config: dict,
        detector: torch.nn.Module,
        eval_data: DataLoader,
        log_dir: str,
        img_size: float = 256.0,
    ):
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.config = config

        self.cam_id_list = config['model_params']['cam_id_list']
        if config['dataset_params']['dataset']['name'] == 'mpi_inf_3dhp':
            self.cal_per_act = False
        else:
            self.cal_per_act = True

        if 'dataiter' in config['dataset_params']:
            self.mean, self.std = config['dataset_params']['dataiter']['mean'], config['dataset_params']['dataiter']['std']
        else:
            self.mean, self.std = None, None

        self.tb_parent_ids = np.array(config['model_params']['parent_ids'])
        self.tb_pair_ids = np.array(config['model_params']['flip_pairs'])

        self.detector = detector.to(self.gpu_id)
        self.eval_data = eval_data

        self.detector = DDP(self.detector, device_ids=[self.gpu_id])
        self.detector.eval()

        self.log_dir = log_dir
        self.img_size = img_size

    def convert_data_to_device(self, x):
        for key in x:
            if isinstance(x[key], torch.Tensor):
                x[key] = x[key].to(self.gpu_id)
            elif isinstance(x[key], dict):
                x[key] = self.convert_data_to_device(x[key])
            elif isinstance(x[key], np.ndarray):
                x[key] = torch.tensor(x[key]).to(self.gpu_id)

        return x

    def eval(self, tb_log, record_table, count_table, record_3d_table, count_3d_table,
             record_3d_tri_table, count_3d_tri_table, ambiguity_ratio, mode='best'):

        for cur_step, x in enumerate(tqdm(self.eval_data, leave=False, disable=(self.gpu_id!=0))):
            x = self.convert_data_to_device(x)
            kp_pred_dict = {}
            trans_dict = {}
            for cam_id in self.cam_id_list:
                cam_key = 'cam_{}'.format(cam_id)
                kp_pred_dict[cam_key], _ = self.detector(x['{}_img'.format(cam_key)])

                kp_pred_2d = kp_pred_dict[cam_key].clone()[..., :2]

                kp_gt = x['{}_joints'.format(cam_key)].clone()

                kp_gt[..., :2] = kp_gt[..., :2] / (self.img_size - 1) * 2 - 1
                kp_gt[..., 2] = kp_gt[..., 2] / (self.img_size - 1)

                # process multi-hypothesis
                for hypo_id in range(kp_pred_dict[cam_key].shape[1]):
                    ignore_list = None # DEBUG [12,13,15,16]
                    if ignore_list is not None:
                        kp_pred_2d[:, hypo_id, ignore_list, :] = kp_gt[:, ignore_list, :2]

                    kp_pred_2d[:, hypo_id, ...], _ = switch_points(kp_pred_2d[:, hypo_id, ...], kp_gt[..., :2])
                    kp_pred_dict[cam_key][:, hypo_id, ...], trans_dict[cam_key] = switch_points(kp_pred_dict[cam_key][:, hypo_id, ...], kp_gt, switch_all=False)

                if mode == 'best' and kp_pred_dict[cam_key].shape[1] > 1:
                    # select min error with gt, best_idx is in [B, N]
                    best_idx = (kp_pred_dict[cam_key] - kp_gt[:, None, ...]).pow(2).sum(dim=-1).argmin(dim=1)
                    kp_pred_dict[cam_key] = torch.gather(kp_pred_dict[cam_key], 1, best_idx[:, None, :, None].expand(-1, -1, -1, kp_pred_dict[cam_key].shape[-1]))
                    kp_pred_dict[cam_key] = kp_pred_dict[cam_key].squeeze(1)
                    best_2d_idx = (kp_pred_2d - kp_gt[:, None, ..., :2]).pow(2).sum(dim=-1).argmin(dim=1)
                    kp_pred_2d = torch.gather(kp_pred_2d, 1, best_2d_idx[:, None, :, None].expand(-1, -1, -1, kp_pred_2d.shape[-1]))
                    kp_pred_2d = kp_pred_2d.squeeze(1)
                elif mode == 'confident' or kp_pred_dict[cam_key].shape[1] == 1:
                    kp_pred_dict[cam_key] = kp_pred_dict[cam_key][:, 0, ...]
                    kp_pred_2d = kp_pred_2d[:, 0, ...]
                else:
                    raise ValueError('Unknown mode: {}'.format(mode))

                if self.gpu_id == 0:
                    tb_log.add_image('testing_pred_pose/{}_pred_pose_v2'.format(cam_key), pose_vis(kp_pred_2d[0, :, :2]\
                                        .detach().data.cpu().numpy(), x['{}_img'.format(cam_key)].shape[2:4], self.tb_pair_ids, \
                                        parent_ids=self.tb_parent_ids, img=x['{}_img'.format(cam_key)][0, :, :, :].clone(), mean=self.mean, std=self.std), cur_step)
                    tb_log.add_image('testing_gt_pose/{}_gt_pose_v2'.format(cam_key), pose_vis(kp_gt[0, :, :2]\
                                        .detach().data.cpu().numpy(), x['{}_img'.format(cam_key)].shape[2:4], self.tb_pair_ids, \
                                        parent_ids=self.tb_parent_ids, img=x['{}_img'.format(cam_key)][0, :, :, :].clone(), mean=self.mean, std=self.std), cur_step)

                # 2D
                error_2d = per_act_mse(kp_pred_2d, kp_gt[..., :2])
                if cal_per_act:
                    update_dict(record_table, count_table, error_2d, x['act'])
                else:
                    record_table += error_2d
                    count_table += 1

            trans_val = torch.zeros_like(trans_dict['cam_0']).to(self.gpu_id).to(torch.float32)
            for cam_id in self.cam_id_list:
                cam_key = 'cam_{}'.format(cam_id)
                trans_val = trans_val + trans_dict[cam_key].to(self.gpu_id)

            ambiguity_ratio = ambiguity_ratio + torch.min(trans_val, len(self.cam_id_list) - trans_val).mean()

            # eval 3D
            kps_world_gt = convert_patch_to_world(x['cam_0_joints'], x, 'cam_0', is_norm=False)
            vis_mask = np.ones((kps_world_gt.shape[0], kps_world_gt.shape[1]), dtype=bool)
            if self.gpu_id == 0:
                tb_log.add_image('testing_pose_3D/gt', pose_vis_3d(kps_world_gt[0]\
                        .detach().data.cpu().numpy(), self.tb_pair_ids, self.tb_parent_ids), cur_step)

            # NOTE(yyc): comment out ref_keypoints input for removing gt overlapping in tb_log
            # triangulation 3D
            kps_world_pred_list = [triangulation(kp_pred_dict, x, self.cam_id_list)]
            record_3d_tri_table, count_3d_tri_table = update_dict_3d(kps_world_pred_list, kps_world_gt, vis_mask, record_3d_tri_table, count_3d_tri_table, x['act'])
            if self.gpu_id == 0:
                tb_log.add_image('testing_pose_3D/pred_tri', pose_vis_3d(kps_world_pred_list[0][0]\
                        .detach().data.cpu().numpy(), self.tb_pair_ids, self.tb_parent_ids, \
                        ref_keypoints=kps_world_gt[0].detach().data.cpu().numpy()), cur_step)

            # single-view 3D
            kps_world_pred_list = []
            for cam_id in self.cam_id_list:
                cam_key = 'cam_{}'.format(cam_id)
                kps_world_pred = convert_patch_to_world(kp_pred_dict[cam_key], x, cam_key, is_norm=True)
                if self.gpu_id == 0:
                    tb_log.add_image('testing_pose_3D/pred_{}'.format(cam_key), pose_vis_3d(kps_world_pred[0]\
                            .detach().data.cpu().numpy(), self.tb_pair_ids, self.tb_parent_ids, \
                            ref_keypoints=kps_world_gt[0].detach().data.cpu().numpy()), cur_step)
                kps_world_pred_list.append(kps_world_pred)

            record_3d_table, count_3d_table = update_dict_3d(kps_world_pred_list, kps_world_gt, vis_mask, record_3d_table, count_3d_table, x['act'])

        return [record_table, count_table, record_3d_table, count_3d_table, record_3d_tri_table, count_3d_tri_table, ambiguity_ratio]

    def record(self, record_table, count_table, record_3d_table, count_3d_table, record_3d_tri_table, count_3d_tri_table, ambiguity_ratio):
        if self.cal_per_act:
            full_err, select_err = cal_per_class_error(record_table, count_table)
            print("---2D-----")
            print(record_table)
            print("----------")
            print('2D MSE: {} %'.format(full_err.item()))
            print("----------")
            print('2D MSE: {} %'.format(select_err.item()))

            full_err_3d, select_err_3d = cal_per_class_error(record_3d_table, count_3d_table, multi=True)
            print("---3D----")
            print(record_3d_table)
            print("----------")
            print('MPJPE: {}'.format(full_err_3d['mpjpe'].item()))
            print('N-MPJPE: {}'.format(full_err_3d['n-mpjpe'].item()))
            print('P-MPJPE: {}'.format(full_err_3d['p-mpjpe'].item()))
            print("----------")
            print('MPJPE: {}'.format(select_err_3d['mpjpe'].item()))
            print('N-MPJPE: {}'.format(select_err_3d['n-mpjpe'].item()))
            print('P-MPJPE: {}'.format(select_err_3d['p-mpjpe'].item()))

            full_err_3d_tri, select_err_3d_tri = cal_per_class_error(record_3d_tri_table, count_3d_tri_table, multi=True)
            print("---Tri3D----")
            print(record_3d_tri_table)
            print("----------")
            print('MPJPE: {}'.format(full_err_3d_tri['mpjpe'].item()))
            print('N-MPJPE: {}'.format(full_err_3d_tri['n-mpjpe'].item()))
            print('P-MPJPE: {}'.format(full_err_3d_tri['p-mpjpe'].item()))
            print("----------")
            print('MPJPE: {}'.format(select_err_3d_tri['mpjpe'].item()))
            print('N-MPJPE: {}'.format(select_err_3d_tri['n-mpjpe'].item()))
            print('P-MPJPE: {}'.format(select_err_3d_tri['p-mpjpe'].item()))

            with open(os.path.join(self.log_dir, 'eval', 'eval_result.txt'), 'w') as f:
                f.write('2D MSE: {} %\n'.format(full_err.item()))

                f.write('MPJPE: {} %\n'.format(full_err_3d['mpjpe'].item()))
                f.write('N-MPJPE: {} %\n'.format(full_err_3d['n-mpjpe'].item()))
                f.write('P-MPJPE: {} %\n'.format(full_err_3d['p-mpjpe'].item()))

                f.write('TRI MPJPE: {} %\n'.format(full_err_3d_tri['mpjpe'].item()))
                f.write('TRI N-MPJPE: {} %\n'.format(full_err_3d_tri['n-mpjpe'].item()))
                f.write('TRI P-MPJPE: {} %\n'.format(full_err_3d_tri['p-mpjpe'].item()))

                f.write('--------select---------\n')
                f.write('2D MSE: {} %\n'.format(select_err.item()))

                f.write('MPJPE: {} %\n'.format(select_err_3d['mpjpe'].item()))
                f.write('N-MPJPE: {} %\n'.format(select_err_3d['n-mpjpe'].item()))
                f.write('P-MPJPE: {} %\n'.format(select_err_3d['p-mpjpe'].item()))

                f.write('TRI MPJPE: {} %\n'.format(select_err_3d_tri['mpjpe'].item()))
                f.write('TRI N-MPJPE: {} %\n'.format(select_err_3d_tri['n-mpjpe'].item()))
                f.write('TRI P-MPJPE: {} %\n'.format(select_err_3d_tri['p-mpjpe'].item()))
        else:
            print("---2D-----")
            print('2D MSE: {} %'.format((record_table.mean() / count_table).item()))
            print("----------")
            print("---3D-----")
            for key in record_3d_table.keys():
                if key == 'pck' or key == 'auc':
                    print('{}: {} %'.format(key, (record_3d_table[key] / count_3d_table[key]).item()))
                else:
                    print('{}: {}'.format(key, (record_3d_table[key].mean() / count_3d_table[key]).item()))
            print("----------")
            print("---Tri3D-----")
            for key in record_3d_tri_table.keys():
                if key == 'pck' or key == 'auc':
                    print('{}: {} %'.format(key, (record_3d_tri_table[key] / count_3d_tri_table[key]).item()))
                else:
                    print('{}: {}'.format(key, (record_3d_tri_table[key].mean() / count_3d_tri_table[key]).item()))

            with open(os.path.join(self.log_dir, 'eval', 'eval_result.txt'), 'w') as f:
                f.write('2D MSE: {} %\n'.format((record_table.mean() / count_table).item()))
                f.write('---3D-----\n')
                for key in record_3d_table.keys():
                    if key == 'pck' or key == 'auc':
                        f.write('{}: {} %\n'.format(key, (record_3d_table[key] / count_3d_table[key]).item()))
                    else:
                        f.write('{}: {}\n'.format(key, (record_3d_table[key].mean() / count_3d_table[key]).item()))

                f.write('---Tri3D-----\n')
                for key in record_3d_tri_table.keys():
                    if key == 'pck' or key == 'auc':
                        f.write('{}: {} %\n'.format(key, (record_3d_tri_table[key] / count_3d_table[key]).item()))
                    else:
                        f.write('{}: {}\n'.format(key, (record_3d_tri_table[key].mean() / count_3d_tri_table[key]).item()))

        print('Results saved in {}'.format(os.path.join(self.log_dir, 'eval', 'eval_result.txt')))
        print('Ambiguity Ratio:{}'.format(ambiguity_ratio / len(self.eval_data) / len(self.cam_id_list)))

        return

def prepare_model(config, opt):
    if config['model_params']['detector_params']['name'] == 'resnet_multi':
        detector = KPDetector3DMulti_integral(**config['model_params']['detector_params'])
    else:
        detector = KPDetector3D_integral(**config['model_params']['detector_params'])

    if torch.cuda.is_available():
        map_location = None
    else:
        map_location = 'cpu'
    checkpoint = torch.load(opt.checkpoint, map_location)

    detector_params = {k.replace('regressor.', ''):v for k, v in checkpoint['unsup_model'].items() if 'regressor.' in k}
    detector.load_state_dict(detector_params)

    return detector

def prepare_data(config, world_size, worker):
    eval_dataset = basic_data(config, eval_only=True)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=config['train_params']['batch_size'] // world_size,
                             shuffle=False,
                             num_workers=worker,
                             drop_last=False,
                             pin_memory=True,
                             sampler=DistributedSampler(eval_dataset, shuffle=False))
    return eval_loader

def create_logger(opt):
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        raise Exception("Must specify checkpoint path")

    if os.environ['LOCAL_RANK'] == '0':
        tb_logger = SummaryWriter(log_dir=os.path.join(log_dir, 'eval', 'tensorboard'))
    else:
        tb_logger = None

    return log_dir, tb_logger

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--worker", default=10, type=int)
    parser.add_argument("--extra_tag", default=' ')
    parser.add_argument("--multi_hypo", default='best', choices=['best', 'confident'], help='multi-hypothesis eval mode')
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model_params']['cam_id_list'] = config['dataset_params']['cam_id_list']

    if opt.batch_size:
        config['train_params']['batch_size'] = opt.batch_size

    # init statistics
    if config['dataset_params']['dataset']['name'] == 'mpi_inf_3dhp':
        cal_per_act = False
    else:
        cal_per_act = True

    if cal_per_act:
        record_table = copy.deepcopy(act)
        count_table = copy.deepcopy(act)

        record_3d_table = {'mpjpe': copy.deepcopy(act),
                            'n-mpjpe': copy.deepcopy(act),
                            'p-mpjpe': copy.deepcopy(act)}
        count_3d_table = {'mpjpe': copy.deepcopy(act),
                            'n-mpjpe': copy.deepcopy(act),
                            'p-mpjpe': copy.deepcopy(act)}
        record_3d_tri_table = {'mpjpe': copy.deepcopy(act),
                            'n-mpjpe': copy.deepcopy(act),
                            'p-mpjpe': copy.deepcopy(act)}
        count_3d_tri_table = {'mpjpe': copy.deepcopy(act),
                            'n-mpjpe': copy.deepcopy(act),
                            'p-mpjpe': copy.deepcopy(act)}
    else:
        record_table = 0.0
        count_table = 0.0
        record_3d_table = {'mpjpe': 0.0, 'n-mpjpe': 0.0, 'p-mpjpe': 0.0, 'pck': 0.0, 'auc':0.0}
        count_3d_table = {'mpjpe': 0.0, 'n-mpjpe': 0.0, 'p-mpjpe': 0.0, 'pck': 0.0, 'auc':0.0}
        record_3d_tri_table = {'mpjpe': 0.0, 'n-mpjpe': 0.0, 'p-mpjpe': 0.0, 'pck': 0.0, 'auc':0.0}
        count_3d_tri_table = {'mpjpe': 0.0, 'n-mpjpe': 0.0, 'p-mpjpe': 0.0, 'pck': 0.0, 'auc':0.0}

    ambiguity_ratio = 0.0

    ddp_setup()

    log_dir, tb_logger = create_logger(opt)
    detector = prepare_model(config, opt)
    eval_loader = prepare_data(config, int(os.environ['WORLD_SIZE']), opt.worker)

    eval = Eval(config, detector, eval_loader, log_dir)

    with torch.no_grad():
        record = [record_table, count_table, record_3d_table, count_3d_table, record_3d_tri_table, count_3d_tri_table, ambiguity_ratio, opt.multi_hypo]
        record = eval.eval(tb_logger, *record)

    destroy_process_group()

    if os.environ['LOCAL_RANK'] == '0':
        eval.record(*record)
        tb_logger.close()