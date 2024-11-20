import os
import yaml

import torch
import random
import numpy as np

from argparse import ArgumentParser
from shutil import copy
from time import gmtime, strftime
from tqdm import trange, tqdm

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from train_util import basic_data, tb_vis

from modules.model import Counter3DModel, Counter3DDisc
from modules.keypoint_detector_integral import KPDetector3D
from modules.keypoint_detector_integral_multi import KPDetector3DMulti
from modules.discriminator import GCNDiscriminator, GCNDiscriminatorDecouple, GCNSAGEDiscriminator
from modules.physique_network import PhysiqueMaskGenerator

from modules.smplpytorch.pytorch.smpl_layer import SMPL_Layer

def setup_seed(seed):
    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return

def ddp_setup():
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

class Trainer:
    def __init__(
        self,
        config: dict,
        unsup_model: torch.nn.Module,
        unsup_disc: torch.nn.Module,
        train_data: DataLoader,
        optimizer_detector: torch.optim.Optimizer,
        save_dir: str,
        checkpoint_path: str = None,
        optimizer_discriminator: torch.optim.Optimizer = None,
        mode: str = 'train',
    ) -> None:

        self.gpu_id = int(os.environ['LOCAL_RANK'])

        self.unsup_model = unsup_model.to(self.gpu_id)
        self.unsup_disc = unsup_disc.to(self.gpu_id)
        self.train_data = train_data

        self.optimizer_detector = optimizer_detector
        self.optimizer_discriminator = optimizer_discriminator

        self.epochs_run = 0
        self.config = config

        self.save_dir = save_dir
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, mode)

        # init scheduler
        self.scheduler_detector = MultiStepLR(optimizer_detector, config['train_params']['epoch_milestones'], gamma=0.1,
                                        last_epoch= -1 + self.epochs_run * (config['train_params']['lr_kp_detector'] != 0))
        if self.optimizer_discriminator is not None:
            self.scheduler_discriminator = MultiStepLR(optimizer_discriminator, config['train_params']['epoch_milestones'], gamma=0.1,\
                                            last_epoch=-1 + self.epochs_run * (config['train_params']['lr_discriminator'] != 0))
        else:
            self.scheduler_discriminator = None

        # wrap model
        self.unsup_model = DDP(self.unsup_model, device_ids=[self.gpu_id])
        self.unsup_disc = DDP(self.unsup_disc, device_ids=[self.gpu_id])

        # inti GAN training params
        self.tb_parent_ids = np.array(config['model_params']['parent_ids'])
        self.tb_pair_ids = np.array(config['model_params']['flip_pairs'])

        if config['model_params']['loss_config']['smpl_disc_loss']['update_interval'] >= 1:
            self.disc_update_interval = config['model_params']['loss_config']['smpl_disc_loss']['update_interval']
            self.gen_update_interval = 1
        else:
            self.disc_update_interval = 1
            self.gen_update_interval = int(1.0 / config['model_params']['loss_config']['smpl_disc_loss']['update_interval'])

    def _load_checkpoint(self, checkpoint_path, mode):
        loc = f'cuda:{self.gpu_id}'
        checkpoint = torch.load(checkpoint_path, map_location=loc)
        self.unsup_model.load_state_dict(checkpoint['unsup_model'])
        self.optimizer_detector.load_state_dict(checkpoint['optimizer_detector'])

        try:
            self.unsup_disc.load_state_dict(checkpoint['unsup_disc'])
            if self.optimizer_discriminator is not None:
                self.optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])

        except:
            print('Load new discriminator for ablation')

        # NOTE(yyc): do not load optimizer during finetune
        if mode == 'train':
            self.epochs_run = checkpoint['epochs']
            print(f'Resuming training from checkpoint at Epoch {self.epochs_run}')

        elif mode == 'finetune':
            print(f'Finetuning from checkpoint at Epoch {self.epochs_run}')
        else:
            raise NotImplementedError

    def _save_checkpoint(self, epoch):
        checkpoint = {
            'unsup_model': self.unsup_model.module.state_dict(),
            'unsup_disc': self.unsup_disc.module.state_dict(),
            'epochs': epoch,
            'optimizer_detector': self.optimizer_detector.state_dict(),
            'optimizer_discriminator': self.optimizer_discriminator.state_dict()
        }

        torch.save(checkpoint, os.path.join(self.save_dir, '{:05d}_ckpt.pth.tar'.format(epoch)))

    def convert_data_to_device(self, x):
        for key in x:
            if isinstance(x[key], torch.Tensor):
                x[key] = x[key].to(self.gpu_id)
            elif isinstance(x[key], dict):
                x[key] = self.convert_data_to_device(x[key])
            elif isinstance(x[key], np.ndarray):
                x[key] = torch.tensor(x[key]).to(self.gpu_id)

        return x

    def train(self, tb_logger):

        num_epochs = self.config['train_params']['num_epochs']
        ckpt_save_freq = self.config['train_params']['checkpoint_freq']

        for epoch in trange(self.epochs_run, self.config['train_params']['num_epochs'], disable=(self.gpu_id != 0)):
            self.train_data.sampler.set_epoch(epoch)
            for iter_num, x in enumerate(tqdm(self.train_data, leave=False, disable=(self.gpu_id != 0))):
                cur_step = epoch * len(self.train_data) + iter_num

                output = {}
                x = self.convert_data_to_device(x)

                # NOTE(yyc): update discriminator first
                if self.optimizer_discriminator is not None and \
                    cur_step % self.disc_update_interval == 0:

                    loss_disc, disc_info = self.unsup_disc(x, self.unsup_model.module.regressor)

                    output.update(disc_info)

                    loss_disc = loss_disc.mean()
                    loss_disc.backward()

                    self.optimizer_discriminator.step()
                    self.optimizer_discriminator.zero_grad()

                else:
                    loss_disc = None

                if cur_step % self.gen_update_interval == 0:
                    loss_kp, kp_info = self.unsup_model(x, self.unsup_disc.module.smpl_discriminator)

                    output.update(kp_info)

                    loss_values = [val.mean() for val in loss_kp.values()]
                    loss = sum(loss_values)

                    tb_log_total_loss = loss.item()

                    loss.backward()

                    self.optimizer_detector.step()
                    self.optimizer_detector.zero_grad()

                else:
                    tb_log_total_loss = None
                    loss_kp = {}

                if self.gpu_id == 0:
                    tb_vis(tb_logger, cur_step, self.tb_pair_ids, self.tb_parent_ids,
                            tb_log_total_loss, loss_kp, loss_disc,
                            output, x, config, self.scheduler_detector)

                    # NOTE(yyc): DEBUG use
                    # if cur_step >= 10:
                    #     tb_log.close()
                    #     raise ValueError

            self.scheduler_detector.step()
            self.scheduler_discriminator.step()

            if self.gpu_id == 0 and (epoch % ckpt_save_freq == 0 or epoch == num_epochs - 1):
                self._save_checkpoint(epoch)

def prepare_model(config):

    if config['model_params']['detector_params']['name'] == 'resnet_multi':
        regressor = KPDetector3DMulti(**config['model_params']['detector_params'])
    else:
        regressor = KPDetector3D(**config['model_params']['detector_params'])

    if 'smpl_disc_params' in config['model_params']:
        if 'gcn' in config['model_params']['smpl_disc_params']['name']:
            if 'decouple' in config['model_params']['smpl_disc_params']['name']:
                smpl_discriminator = GCNDiscriminatorDecouple(config['model_params']['smpl_disc_params'])
            elif 'sage' in config['model_params']['smpl_disc_params']['name']:
                smpl_discriminator = GCNSAGEDiscriminator(config['model_params']['smpl_disc_params'])
            else:
                smpl_discriminator = GCNDiscriminator(config['model_params']['smpl_disc_params'])
        else:
            raise NotImplementedError

        smpl_layer = SMPL_Layer(
            center_idx=0,
            gender='neutral',
            model_root=config['model_params']['smpl_layer_params']['model_path']
        )

        h36m_regressor = np.load(os.path.join(config['model_params']['smpl_layer_params']['model_path'], \
                                              'J_regressor_h36m.npy'))
        h36m_regressor = torch.tensor(h36m_regressor, dtype=torch.float32)

    else:
        smpl_discriminator = None
        smpl_layer = None
        h36m_regressor = None

    if 'physique_mask_generator_params' in config['model_params']:
        physique_mask_generator = PhysiqueMaskGenerator(
            config['model_params']['physique_mask_generator_params']['layers']
        )
    else:
        physique_mask_generator = None

    net_params = list(regressor.parameters())

    if physique_mask_generator is not None:
        net_params = net_params + list(physique_mask_generator.parameters())

    optimizer_detector = torch.optim.Adam(net_params,\
                        lr=config['train_params']['lr_kp_detector'], betas=(0.5, 0.999))

    if smpl_discriminator is not None:
        optimizer_discriminator = torch.optim.Adam(smpl_discriminator.parameters(),\
                            lr=config['train_params']['lr_discriminator'], betas=(0.5, 0.999))
    else:
        optimizer_discriminator = None

    unsup_model = Counter3DModel(config['model_params'], regressor, smpl_layer, h36m_regressor, physique_mask_generator)
    unsup_disc = Counter3DDisc(config['model_params'], smpl_discriminator, smpl_layer, h36m_regressor)
    
    return unsup_model, unsup_disc, optimizer_detector, optimizer_discriminator

def prepare_data(config, world_size, worker):
    train_dataset = basic_data(config)
    train_loader = DataLoader(train_dataset,
                              batch_size=config['train_params']['batch_size'] // world_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=worker,
                              sampler=DistributedSampler(train_dataset))

    return train_loader

def create_logger(opt):
    if opt.checkpoint is not None and not opt.finetune:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        seed = 'seed{}_'.format(opt.seed if opt.seed !=-1 else '_rand')
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        if opt.finetune:
            log_dir += '_FINETUNE'
        log_dir += '_' + seed + opt.extra_tag + strftime('%d_%m_%y_%H.%M.%S', gmtime())

    if os.environ['LOCAL_RANK'] == '0':
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
            copy(opt.config, log_dir)

        tb_logger = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))

    else:
        tb_logger = None
    return log_dir, tb_logger

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config')
    parser.add_argument('--log_dir', default='log', help='path to log into')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint to restore')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--epoch', default=None, type=int)
    parser.add_argument('--worker', default=10, type=int)
    parser.add_argument('--extra_tag', default='')
    parser.add_argument('--finetune', default=False, action='store_true', help='finetune the model')
    parser.add_argument('--seed', default=-1, type=int)
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model_params']['cam_id_list'] = config['dataset_params']['cam_id_list']

    # running on server
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    if opt.batch_size:
        config['train_params']['batch_size'] = opt.batch_size
    if opt.epoch:
        config['train_params']['num_epochs'] = opt.epoch

    ddp_setup()
    setup_seed(opt.seed)

    save_dir, tb_logger = create_logger(opt)
    unsup_model, unsup_disc, optimizer_detector, optimizer_discriminator = prepare_model(config)
    train_loader = prepare_data(config, world_size=int(os.environ['WORLD_SIZE']), worker=opt.worker)

    trainer = Trainer(config, unsup_model, unsup_disc, train_loader, optimizer_detector, save_dir,
                      checkpoint_path=opt.checkpoint,
                      optimizer_discriminator=optimizer_discriminator,
                      mode='finetune' if opt.finetune else 'train')
    trainer.train(tb_logger)
    if os.environ['LOCAL_RANK'] == '0':
        tb_logger.close()
    destroy_process_group()