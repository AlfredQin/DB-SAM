import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import hydra
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from accelerate.utils import DistributedDataParallelKwargs

from super_gradients import init_trainer
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.common.environment import is_distributed

from super_gradients.common.factories.losses_factory import LossesFactory
from super_gradients.common.factories.datasets_factory import DatasetsFactory
from super_gradients.common.factories.callbacks_factory import CallbacksFactory
from super_gradients.common.factories.metrics_factory import MetricsFactory

from src.training.trainer.accelerate_sg_trainer import Trainer
from src.training.datasets.npz_dataset import get_medsam_test_dataset
from projects.db_sam.builder import build_adapted_sam_vit_b

sample_loader = lambda x: np.load(x)['img']


@hydra.main(config_path="configs/", config_name="samed_db_sam_plus_cfg.yaml")
def _main(cfg):
    cfg = hydra.utils.instantiate(cfg)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir="./checkpoints",
                      mixed_precision=cfg.training_hyperparams.mixed_precision,
                      kwargs_handlers=[ddp_kwargs])
    model = build_adapted_sam_vit_b(cfg.sam_ckpt_path, cfg.model_params)

    train_val_dataset = DatasetsFactory().get(conf=cfg.dataset_params.train_dataset_params)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        train_val_dataset,
        [int(len(train_val_dataset) * 0.99), len(train_val_dataset) - int(len(train_val_dataset) * 0.99)],
        generator=torch.Generator().manual_seed(42)
    )

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.dataset_params.train_dataloader_params.batch_size,
                                  # shuffle=(train_sampler is None),
                                  num_workers=cfg.dataset_params.train_dataloader_params.num_workers,
                                  # sampler=train_sampler,
                                  drop_last=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset, shuffle=False) if is_distributed() else None
    val_dataloader = DataLoader(validation_dataset,
                                batch_size=cfg.dataset_params.train_dataloader_params.batch_size,
                                shuffle=(val_sampler is None),
                                num_workers=cfg.dataset_params.train_dataloader_params.num_workers,
                                sampler=val_sampler, drop_last=False)

    test_datasets = get_medsam_test_dataset(cfg.dataset_params.test_dataset_params.root,
                                            add_box_perturbation=cfg.dataset_params.test_dataset_params.add_box_perturbation,
                                            perturbation_value=cfg.dataset_params.test_dataset_params.perturbation_value
                                            )  # dict of datasets
    # sort test datasets according to the key
    test_datasets = dict(sorted(test_datasets.items(), key=lambda x: x[0]))
    test_datasets = {dataset_name: dataset for dataset_name, dataset in test_datasets.items()}
    test_samplers = {
        dataset_name: torch.utils.data.distributed.DistributedSampler(test_dataset) if is_distributed() else None
        for dataset_name, test_dataset in test_datasets.items()
    }
    test_dataloaders = {
        dataset_name: DataLoader(test_dataset,
                                 batch_size=cfg.dataset_params.test_dataloader_params.batch_size,
                                 shuffle=False,
                                 num_workers=cfg.dataset_params.test_dataloader_params.num_workers,
                                 sampler=test_samplers[dataset_name],
                                 collate_fn=test_dataset.collate_fn,
                                 drop_last=False)
        for dataset_name, test_dataset in test_datasets.items()
    }

    trainer.train(
        model=model,
        training_params=cfg.training_hyperparams,
        train_loader=train_dataloader,
        valid_loader=val_dataloader,
        test_loaders=test_dataloaders,
    )


def main():
    init_trainer()
    _main()


if __name__ == "__main__":
    main()