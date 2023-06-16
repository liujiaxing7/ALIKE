import argparse
import os
import sys
import time
import logging
import functools
from pathlib import Path

import yaml

sys.path.append('/media/xin/work1/github_pro/ALIKE/') # 添加模型训练根目录

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from ALIKE_code.datasets.hpatches import HPatchesDataset
from ALIKE_code.datasets.megadepth import MegaDepthDataset
from ALIKE_code.datasets.cat_datasets import ConcatDatasets
from ALIKE_code.training.train_wrapper import TrainWrapper
from ALIKE_code.training.scheduler import WarmupConstantSchedule

from pytorch_lightning.callbacks.base import Callback


class RebuildDatasetCallback(Callback):
    def __init__(self):
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        train_loader_dataset = trainer.train_dataloader.dataset
        train_loader_dataset.datasets.datasets[0].build_dataset()
        train_loader_dataset.datasets.datasets[1].build_dataset()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    args = parser.parse_args()

    config_file = args.config
    assert (os.path.exists(config_file))
    # 读取配置文件
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)

    debug = config['data']['data_set']['debug']
    num_workers = config['data']['data_set']['num_workers']
    log_dir = 'log_' + Path(__file__).stem

    # 加载日志文件
    log_name = 'debug' if debug else 'train'
    # version = time.strftime("Version-%m%d-%H%M%S", time.localtime())
    version = config['solver']['version']
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=log_dir, name=log_name, version=version, default_hp_metric=False, sub_dir='loss_tensor')
    logging.info(f'>>>>>>>>>>>>>>>>> log dir: {logger.log_dir}')
    # 训练集
    mega_dataset = [MegaDepthDataset(root=config['data']['image_train_path'], train=True, using_cache=debug, pairs_per_scene=config['data']['data_set']['pairs_per_scene'],
                                     image_size=config['data']['image_size'][0], gray=False, colorjit=True, crop_or_scale=t,img_type=config['data']['image_type']) for t in ['crop','scale']]
    train_datasets = ConcatDatasets(*mega_dataset)
    train_loader = DataLoader(train_datasets, batch_size=config['solver']['batch_size'], shuffle=True, pin_memory=not debug,
                              num_workers=num_workers)

    # 验证集
    val_dataloaders = []
    hpatch_imw_path = config['data']['image_val_path'].get('hpatch_imw_path')
    other_path = config['data']['image_val_path'].get('other_path')
    if hpatch_imw_path:
        for alter in ['i','v']:
            hpatch_dataset = HPatchesDataset(root=hpatch_imw_path[0], alteration=alter)
            hpatch_dataloader = DataLoader(hpatch_dataset, batch_size=config['solver']['batch_size'], pin_memory=not debug, num_workers=num_workers)
            val_dataloaders.append(hpatch_dataloader)
        imw2020val = MegaDepthDataset(root=hpatch_imw_path[1], train=False, using_cache=True, colorjit=False,gray=False)
        imw2020val_dataloader = DataLoader(imw2020val, batch_size=config['solver']['batch_size'], pin_memory=not debug, num_workers=num_workers)
        val_dataloaders.append(imw2020val_dataloader)
    elif other_path:
        for val_path in other_path:
            dataset = MegaDepthDataset(root=val_path, train=False, using_cache=True, colorjit=False, gray=False)
            dataloaders = DataLoader(val_path, batch_size=config['solver']['batch_size'], pin_memory=not debug, num_workers=num_workers)
            val_dataloaders.append(dataloaders)
    # 加载模型
    pretrained_model = config['model']['pretrained_model']
    lr_scheduler = functools.partial(WarmupConstantSchedule, warmup_steps=config['model']['training parameters']['warmup_steps'])
    model = TrainWrapper(
        # ================================== feature encoder
        *config['model']['c1_4_dim_single_head'],
        agg_mode=config['model']['agg_mode'],  # sum, cat, fpn
        single_head=config['model']['single_head'],
        pe=config['model']['pe'],
        # ================================== detect parameters
        radius=config['model']['detect_parameters']['radius'],top_k=config['model']['detect_parameters']['top_k'], scores_th=0, n_limit=0,
        scores_th_eval=config['model']['detect_parameters']['scores_th_eval'], n_limit_eval=config['model']['detect_parameters']['n_limit_eval'],
        # ================================== gt reprojection th
        train_gt_th=config['model']['gt_reprojection_th']['train_gt_th'], eval_gt_th=config['model']['gt_reprojection_th']['eval_gt_th'],
        # ================================== loss weight
        w_pk=config['model']['loss_weight']['w_pk'],  # weight of peaky loss
        w_rp=config['model']['loss_weight']['w_rp'],  # weight of reprojection loss
        w_sp=config['model']['loss_weight']['w_sp'],  # weight of score map rep loss
        w_ds=config['model']['loss_weight']['w_ds'],  # weight of descriptor loss
        w_triplet=config['model']['loss_weight']['w_triplet'],
        sc_th=config['model']['loss_weight']['sc_th'],  # score threshold in peaky and  reprojection loss
        norm=config['model']['loss_weight']['norm'],  # distance norm
        temp_sp=config['model']['loss_weight']['temp_sp'],  # temperature in ScoreMapRepLoss
        temp_ds=config['model']['loss_weight']['temp_ds'],  # temperature in DescReprojectionLoss
        # ================================== learning rate
        lr=float(config['solver']['lr']),
        log_freq_img=config['model']['training parameters']['log_freq_img'],
        # ================================== pretrained_model
        pretrained_model=pretrained_model,
        lr_scheduler=lr_scheduler,
        debug=debug
    )
    # 断点续训
    model_name = config['model']['name']
    dir_path = os.path.join(os.path.dirname(logger.log_dir),model_name+'_checkpoints')
    last_path = os.path.join(dir_path,'last.ckpt')
    if os.path.exists(last_path):
        resume_from_checkpoint = last_path
    else:
        resume_from_checkpoint = None

    # 模型训练
    trainer = pl.Trainer(gpus=config['model']['training parameters']['gpus'],
                         resume_from_checkpoint=resume_from_checkpoint,
                         fast_dev_run=False,
                         accumulate_grad_batches=config['data']['data_set']['accumulate_grad_batches'], # 多少批进行一次梯度累积
                         num_sanity_val_steps=config['data']['data_set']['num_sanity_val_steps'], # 训练前检查多少批验证数据
                         limit_train_batches= config['solver']['limit_train_batches'], # 训练数据集 如果是小数则表示百分比
                         limit_val_batches=config['solver']['limit_val_batches'], # 验证数据集
                         max_epochs=config['solver']['max_epochs'], # 最多训练轮数
                         logger=logger, # 日志
                         reload_dataloaders_every_epoch=config['solver']['reload_dataloaders_every_epoch'], # 每一轮是否重新载入数据
                         callbacks=[
                             ModelCheckpoint(monitor='val_metrics/mean',
                                             save_top_k=-1, # 保存前n个最好的模型
                                             mode='max',
                                             save_last=True,
                                             dirpath=dir_path ,
                                             auto_insert_metric_name=False,
                                             filename='epoch={epoch}-mean_metric={val_metrics/mean:.4f}',
                                             ),
                             LearningRateMonitor(logging_interval='step'),
                             RebuildDatasetCallback()
                         ]
                         )
    trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_dataloaders)






