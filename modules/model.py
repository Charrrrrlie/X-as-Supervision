import torch

from modules.util import convert_patch_to_world, draw_lines
from modules.util import random_rotation_3D, flip_3D
from modules.base_losses.loss_func import compute_mask_reconstruction_loss, compute_supervision, compute_disc_loss
from modules.base_losses.loss_func import compute_bone_sym_loss, compute_kp_sym_loss

def cal_links(parent_ids, line_select_ids=None, use_root=False, extension=True):
    if not use_root:
        child_ids =  list(range(1, len(parent_ids)))
        parent_ids = parent_ids[1:]
    else:
        child_ids = list(range(len(parent_ids)))

    parent_ids = [parent_ids[i] for i in line_select_ids]
    child_ids = [child_ids[i] for i in line_select_ids]

    if extension:
        parent_ids.extend([7,7,7,7,0,0,1,4])
        child_ids.extend([1,4,11,14,2,5,14,11])

    return parent_ids, child_ids

class Counter3DModel(torch.nn.Module):
    def __init__(self, cfg, regressor, smpl_layer, h36m_regressor, physique_network=None):
        super(Counter3DModel, self).__init__()
        self.regressor = regressor

        self.cam_id_list = cfg['cam_id_list']

        self.body_width = cfg['body_width'] if 'body_width' in cfg else 3.0
        self.body_width = float(self.body_width) * 1e-3

        line_select_ids = cfg['line_select_ids'] if 'line_select_ids' in cfg else None
        self.parent_ids, self.child_ids = cal_links(cfg['parent_ids'], line_select_ids=line_select_ids,
                                                       use_root=False, extension=True)

        self.loss_config = cfg['loss_config']

        self.use_learned_width = cfg['use_learned_width'] if 'use_learned_width' in cfg else False

        self.smpl_layer = smpl_layer
        self.h36m_regressor = h36m_regressor

        self.physique_network = physique_network

        self.DISC_SUP_DIMENSION = cfg['smpl_disc_params']['disc_sup_dim']  if 'disc_sup_dim' in cfg['smpl_disc_params'] else 3
        self.use_aug = cfg['smpl_disc_params']['use_aug'] if 'use_aug' in cfg['smpl_disc_params'] else False

    def forward(self, x, smpl_discriminator):
        if 'cam_mono_img' in x:
            # if use single view dataset
            cam_id_list = ['mono']
        else:
            cam_id_list = self.cam_id_list

        loss_values = {}
        output = {}

        kps_ori = {}
        kps_world_ori = {}
        for cam_id in cam_id_list:
            cam_key = 'cam_{}'.format(cam_id)
            kps_ori[cam_key], depth_map = self.regressor(x['{}_img'.format(cam_key)])

            # kps_ori[cam_key] in the shape [B, num_hypo, num_kp, 3]
            assert kps_ori[cam_key].dim() == 4, "use aligned multi-hypothesis settings"
            output['pose_2d_pred_{}_ori'.format(cam_key)] = kps_ori[cam_key].clone()[[0], 0, ...]
            output['depth_map_{}'.format(cam_key)] = depth_map

            kps_world_ori[cam_key] = []
            for i in range(kps_ori[cam_key].shape[1]):
                if cam_id == 'mono':
                    kps_world_ori[cam_key].append(
                        convert_patch_to_world(kps_ori[cam_key][:, i, ...], x, cam_key, is_norm=True, RECT_WIDTH=256, mono=True, patch=False))
                else:
                    kps_world_ori[cam_key].append(
                            convert_patch_to_world(kps_ori[cam_key][:, i, ...], x, cam_key, is_norm=True))
            kps_world_ori[cam_key] = torch.stack(kps_world_ori[cam_key], dim=1)
            output['pose_3d_depth_{}'.format(cam_key)] = kps_world_ori[cam_key][:, 0, ...].clone()

        # use gt to check the performance
        if 'mono' not in cam_id_list:
            output['kp_gt_world'] = convert_patch_to_world(x['cam_0_joints'], x, 'cam_0', is_norm=False)[[0], ...]

        reconstructed = {}

        for cam_id in cam_id_list:
            cam_key = 'cam_{}'.format(cam_id)
            # NOTE(yyc): multi-hypo only affects z dim therefore only one heatmap needs to be calculated
            heatmaps = draw_lines(kps_ori[cam_key][:, 0, :, :2], x['{}_img'.format(cam_key)].shape[-1],
                                  self.parent_ids, self.child_ids, self.body_width)

            reconstructed[cam_key] = torch.max(heatmaps.clone(), dim=1, keepdim=True)[0]

            output['mask_heatmap_line_{}'.format(cam_key)] = torch.max(heatmaps.clone(), dim=1, keepdim=True)[0]

        if 'symmetry_loss' in self.loss_config:
            loss_sym = 0
            for cam_id in cam_id_list:
                if cam_id == 'mono':
                    continue
                cam_key = 'cam_{}'.format(cam_id)

                _loss_sym = []
                for i in range(kps_world_ori[cam_key].shape[1]):
                    _loss_sym_temp  = 0
                    _loss_sym_temp += compute_bone_sym_loss(kps_world_ori[cam_key][:, i, ...]) * self.loss_config['symmetry_loss']['weight']['bone']
                    _loss_sym_temp += compute_kp_sym_loss(kps_world_ori[cam_key][:, i, ...]) * self.loss_config['symmetry_loss']['weight']['kp']

                    if 'kp_2d' in self.loss_config['symmetry_loss']['weight']:
                        _loss_sym_temp += compute_kp_sym_loss(kps_ori[cam_key][:, i, :, :2], is_3D=False) * 1e2 * self.loss_config['symmetry_loss']['weight']['kp_2d']
                    _loss_sym.append(_loss_sym_temp)
                loss_sym += torch.min(torch.stack(_loss_sym))

            loss_values['symmetry'] = loss_sym

        if 'smpl_gen_loss' in self.loss_config:
            loss_gen = 0
            for cam_id in cam_id_list:
                cam_key = 'cam_{}'.format(cam_id)
                # use normalized 3D pose in world coordinate
                pred_joints_world = kps_world_ori[cam_key].clone()
                pred_joints_world = (pred_joints_world - pred_joints_world[:, [0], :]) / 1000

                pred_logits = []
                for i in range(pred_joints_world.shape[1]):
                    pred_logits.append(smpl_discriminator(pred_joints_world[:, i, :, :self.DISC_SUP_DIMENSION].detach()))
                pred_logits = torch.stack(pred_logits, dim=1)

                if not self.use_aug:
                    loss_gen += compute_disc_loss(pred_logits, None)
                else:
                    loss_gen += compute_disc_loss(pred_logits, None) * 0.7

                    pred_logits_rot = []
                    for  i in range(pred_joints_world.shape[1]):
                        pred_logits_rot.append(smpl_discriminator(random_rotation_3D(pred_joints_world[:, i, ...])[..., :self.DISC_SUP_DIMENSION]))
                    pred_logits_rot = torch.stack(pred_logits_rot, dim=1)

                    loss_gen += compute_disc_loss(pred_logits_rot, None) * 0.3

            loss_values['smpl_gen'] = loss_gen * self.loss_config['smpl_gen_loss']['weight']

        if 'smpl_pseudo_img_loss' in self.loss_config:
            loss_pseudo = 0
            for cam_id in cam_id_list:
                cam_key = 'cam_{}'.format(cam_id)

                smpl_pseudo_pred, _ = self.regressor(x['{}_pseudo_img'.format(cam_key)])
                smpl_pseudo_gt = x['{}_pseudo_joints'.format(cam_key)]

                output['pose_2d_pred_{}_pseudo'.format(cam_key)] = smpl_pseudo_pred.clone()[[0], 0, ...]
                output['pose_3d_pred_{}_pseudo'.format(cam_key)] = convert_patch_to_world(smpl_pseudo_pred[:, 0, ...], x, cam_key, is_norm=True, RECT_WIDTH=256, mono=True, patch=False)[[0], ...]

                output['pose_3d_gt_{}_pseudo'.format(cam_key)] = convert_patch_to_world(smpl_pseudo_gt, x, cam_key, is_norm=True, RECT_WIDTH=256, mono=True, patch=False)[[0], ...]

                _loss_pseudo = []

                for i in range(smpl_pseudo_pred.shape[1]):
                    _loss_pseudo.append(compute_supervision(smpl_pseudo_pred[:, i, ...], smpl_pseudo_gt))
                loss_pseudo += torch.min(torch.stack(_loss_pseudo))

            loss_values['smpl_pseudo_img'] = loss_pseudo * self.loss_config['smpl_pseudo_img_loss']['weight']

        if 'physique_recons_loss' in self.loss_config and self.physique_network is not None:
            loss_phy_recons = 0
            use_dis_map = self.loss_config['physique_recons_loss']['use_dis_map']
            for cam_id in cam_id_list:
                cam_key = 'cam_{}'.format(cam_id)
                phy_reconstructed = self.physique_network(reconstructed[cam_key])

                output['mask_physique_{}'.format(cam_key)] = phy_reconstructed[[0], ...]

                loss_phy_recons = loss_phy_recons + \
                    compute_mask_reconstruction_loss(phy_reconstructed, x['{}_mask'.format(cam_key)], \
                    weight=x['{}_geodesic_dis'.format(cam_key)] if use_dis_map else None)

            loss_values['physique_recons'] = loss_phy_recons * self.loss_config['physique_recons_loss']['weight']

        if 'recons_loss' in self.loss_config:
            loss_values['reconstruction'] = 0
            use_dis_map = self.loss_config['recons_loss']['use_dis_map']
            for cam_id in cam_id_list:
                cam_key = 'cam_{}'.format(cam_id)
                loss_values['reconstruction'] = loss_values['reconstruction'] +\
                compute_mask_reconstruction_loss(reconstructed[cam_key], x['{}_mask'.format(cam_key)], \
                    weight=x['{}_geodesic_dis'.format(cam_key)] if use_dis_map else None, use_clip=True)

            loss_values['reconstruction'] = loss_values['reconstruction'] * self.loss_config['recons_loss']['weight']

        return loss_values, output

class Counter3DDisc(torch.nn.Module):
    def __init__(self, cfg, smpl_discriminator, smpl_layer, h36m_regressor):
        super(Counter3DDisc, self).__init__()

        self.smpl_discriminator = smpl_discriminator

        self.cam_id_list = cfg['cam_id_list']

        line_select_ids = cfg['line_select_ids'] if 'line_select_ids' in cfg else None
        self.parent_ids, self.child_ids = cal_links(cfg['parent_ids'], line_select_ids=line_select_ids,
                                                       use_root=False, extension=False)

        self.loss_config = cfg['loss_config']

        if 'GCN' in self.smpl_discriminator.name:
            self.smpl_discriminator.parent_ids = self.parent_ids
            self.smpl_discriminator.child_ids = self.child_ids

        self.smpl_layer = smpl_layer
        self.h36m_regressor = h36m_regressor

        self.DISC_SUP_DIMENSION = cfg['smpl_disc_params']['disc_sup_dim']  if 'disc_sup_dim' in cfg['smpl_disc_params'] else 3
        self.use_aug = cfg['smpl_disc_params']['use_aug'] if 'use_aug' in cfg['smpl_disc_params'] else False

    def forward(self, x, regressor):
        loss_disc = 0
        output = {}

        if 'cam_mono_img' in x:
            # if use single view dataset
            cam_id_list = ['mono']
        else:
            cam_id_list = self.cam_id_list

        for cam_id in cam_id_list:
            cam_key = 'cam_{}'.format(cam_id)

            pred_joints, _ = regressor(x['{}_img'.format(cam_key)])

            # NOTE: use pseudo data
            smpl_joints = x['{}_pseudo_joints'.format(cam_key)]

            smpl_joints_world = convert_patch_to_world(smpl_joints, x, cam_key, is_norm=True, RECT_WIDTH=256, mono=True, patch=False)

            output['pose_smpl_2d_{}'.format(cam_key)] = smpl_joints[[0], ...]
            output['pose_smpl_3d_{}'.format(cam_key)] = smpl_joints_world[[0], ...].clone()

            pred_logits = []
            for i in range(pred_joints.shape[1]):
                pred_logits.append(self.smpl_discriminator(pred_joints[:, i, :, :self.DISC_SUP_DIMENSION].detach()))
            pred_logits = torch.stack(pred_logits, dim=1)
            smpl_logits = self.smpl_discriminator(smpl_joints[..., :self.DISC_SUP_DIMENSION])

            output['smpl_logits_{}'.format(cam_key)] = smpl_logits[[0], ...]
            output['pred_logits_{}'.format(cam_key)] = pred_logits[[0], 0, ...]

            if self.use_aug:
                smpl_joints_world_rot = random_rotation_3D(smpl_joints_world)
                # smpl_joints_world_flip = flip_3D(smpl_joints_world)
                output['pose_smpl_3d_{}_rot'.format(cam_key)] = smpl_joints_world_rot[[0], ...]
                # output['pose_smpl_3d_{}_flip'.format(cam_key)] = smpl_joints_world_flip[[0], ...]
                smpl_logits_rot = self.smpl_discriminator(smpl_joints_world_rot[..., :self.DISC_SUP_DIMENSION])
                # smpl_logits_flip = self.smpl_discriminator(smpl_joints_world_flip[..., :self.DISC_SUP_DIMENSION])
                loss_disc += compute_disc_loss(pred_logits, smpl_logits) * 0.6
                loss_disc += compute_disc_loss(smpl_logits_rot, None) * 0.4
            else:
                loss_disc += compute_disc_loss(pred_logits, smpl_logits)

        loss_disc = loss_disc * self.loss_config['smpl_disc_loss']['weight']

        return loss_disc, output