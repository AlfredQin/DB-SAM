import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'super-gradients/src'))

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from super_gradients import Trainer
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.common.environment import is_distributed

from projects.inference.build_sam import build_sam_vit_b
from projects.sam_ms_deform_attn_adapter.build_sam_adapter import build_adapter_sam_vit_b
from super_gradients.common.factories.losses_factory import LossesFactory
from super_gradients.common.factories.metrics_factory import MetricsFactory

from training.datasets.npz_dataset import get_medsam_test_dataset


def main(args):
    setup_device(num_gpus=args.num_gpus)
    if args.model_name == "sam":
        model = build_sam_vit_b(checkpoint=args.sam_checkpoint)
    elif args.model_name == "med_sam":
        model = build_sam_vit_b(checkpoint=args.med_sam_checkpoint)
    elif args.model_name == "adapted_sam":
        model = build_adapter_sam_vit_b(args)
        ckpt = torch.load(args.adapted_sam_checkpoint)["net"]
        model.load_state_dict(ckpt)
    loss = LossesFactory().get(conf={args.loss_type: {"include_background": args.include_background}})

    test_datasets = get_medsam_test_dataset(args.test_data_dir, add_box_perturbation=args.add_box_perturbation)  # dict of datasets
    # sort test datasets according to the key
    test_datasets = dict(sorted(test_datasets.items(), key=lambda x: x[0]))
    test_datasets = {dataset_name: dataset for dataset_name, dataset in test_datasets.items()}
    test_samplers = {
        dataset_name: torch.utils.data.distributed.DistributedSampler(test_dataset) if is_distributed() else None
        for dataset_name, test_dataset in test_datasets.items()
    }
    test_dataloaders = {
        dataset_name: DataLoader(test_dataset, batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers, sampler=test_samplers[dataset_name],
                                 collate_fn=test_dataset.collate_fn, drop_last=False)
        for dataset_name, test_dataset in test_datasets.items()
    }

    val_metrics = [MetricsFactory().get(conf={"DICEScore": {}}), MetricsFactory().get(conf={"SurfaceDICEScore": {}})]

    trainer = Trainer(experiment_name=args.experiment_name)

    model.training_state = "testing"
    results = {}
    for dataset_name, test_dataloader in test_dataloaders.items():
        test_results = trainer.test(model=model, test_loader=test_dataloader, loss=loss, test_metrics_list=val_metrics)
        results[dataset_name] = test_results
        print(f"Test results for {dataset_name}: {test_results}")

    print(results)
    # save results to txt file
    # get the current time
    import datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H:%M:%S")
    file_name = f"{args.model_name}_add_box_perturbation_{args.add_box_perturbation}_{now}.txt"
    with open(os.path.join("./results", file_name), "w") as f:
        f.write(str(results))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='finetune_sam_mask_decoder')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--train_dataset_type', type=str, default='SAMEncEmbeddingDataset')
    parser.add_argument('--train_data_root', type=str, default='/home/qinc/Code/Segmentation/MedSAM/data/Tr_npy')
    parser.add_argument('--samples_sub_directory', type=str, default='npy_images')
    parser.add_argument('--sample_extensions', type=str, default='.npz')
    parser.add_argument('--targets_sub_directory', type=str, default='npy_gts')
    parser.add_argument('--target_extension', type=str, default='.npy')
    parser.add_argument('--test_data_dir', type=str, default='/home/qinc/Downloads/Test')
    parser.add_argument('--add_box_perturbation', action='store_true')

    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--model_config_path', type=str, default='./configs/conv_vit_adapter_cfg.py')
    parser.add_argument('--sam_checkpoint', type=str,
                        default='/home/qinc/pretrained_weights/sam_vit_b_01ec64.pth')
    parser.add_argument('--frozen_prompt_encoder', action='store_true')
    parser.add_argument('--frozen_mask_decoder', action='store_true')

    parser.add_argument('--global_attn_indexes', type=list, default=[2, 5, 8, 11])
    parser.add_argument('--prompt_embed_dim', type=int, default=256)
    parser.add_argument('--interaction_indexes', type=list, default=[[0, 2], [3, 5], [6, 8], [9, 11]])
    parser.add_argument('--pixel_mean', type=list, default=[123.675, 116.28, 103.53])
    parser.add_argument('--pixel_std', type=list, default=[58.395, 57.12, 57.375])
    parser.add_argument('--loss_type', type=str, default='SegLoss')
    parser.add_argument('--include_background', action='store_true')

    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_mode', type=str, default='PolyLRScheduler')
    parser.add_argument('--step_lr_update_freq', type=int, default=5)
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_mode', type=str, default='LinearBatchLRWarmup')
    parser.add_argument('--lr_warmup_epochs', type=int, default=1)
    parser.add_argument('--warmup_initial_lr', type=float, default=1e-7)
    parser.add_argument('--lr_warmup_steps', type=int, default=1500)
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--model_name', type=str, default='sam')
    parser.add_argument('--adapted_sam_checkpoint', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
