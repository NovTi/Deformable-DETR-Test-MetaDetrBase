import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pdb

import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.dataset_support import build_support_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from util.lr_scheduler import WarmupMultiStepLR
from util.load_cfg import load_cfg_from_cfg_file, merge_cfg_from_list


torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Testing Deformable DETR Idea')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    cfg.config_path = args.config
    cfg.debug_flag = args.debug
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main(args):
    # utils.init_distributed_mode(args)
    print(args)

    # pdb.set_trace()
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    image_set = 'finetune' if args.is_finetune else 'train'
    dataset_train = build_dataset(image_set=image_set, args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=False)

    loader_train = DataLoader(dataset_train,
                              batch_sampler=batch_sampler_train,
                              collate_fn=utils.collate_fn,
                              num_workers=args.num_workers,
                              pin_memory=True)

    loader_val = DataLoader(dataset_val,
                            batch_size=args.batch_size,
                            sampler=sampler_val,
                            drop_last=False,
                            collate_fn=utils.collate_fn,
                            num_workers=args.num_workers,
                            pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    # pdb.set_trace()

    if args.is_finetune:
        # load weight
        # change to the cosine classifer
        pass

    if args.is_finetune:
        # For few-shot finetune stage, just train the class_embed and bbox_embed modules
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                     if match_name_keywords(n, args.finetune_module_name) and p.requires_grad],
                "lr": args.lr,
                "initial_lr": args.lr,
            },
            # {
            #     "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            #     "lr": args.lr_backbone,
            #     "initial_lr": args.lr_backbone,
            # },
        ]  
    else:
        param_dicts = [
            {
                "params":
                    [p for n, p in model_without_ddp.named_parameters()
                     if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
                "initial_lr": args.lr,
            },
            # also train backbone?
            # {
            #     "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            #     "lr": args.lr_backbone,
            #     "initial_lr": args.lr_backbone,
            # },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
                "initial_lr": args.lr * args.lr_linear_proj_mult,
            }
        ]

    optimizer = torch.optim.AdamW(param_dicts, weight_decay=args.weight_decay)
    # change to the original Deformable-DETR scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)  

    # lr_scheduler = WarmupMultiStepLR(optimizer,
    #                                  args.lr_drop_milestones,
    #                                  gamma=0.1,
    #                                  warmup_epochs=args.warmup_epochs,
    #                                  warmup_factor=args.warmup_factor,
    #                                  warmup_method='linear',
    #                                  last_epoch=args.start_epoch - 1)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)
    if args.resume:  # load weight for finetune on novel set
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_dict = model_without_ddp.state_dict()
        for index, key in enumerate(model_dict.keys()):
            if 'class_embed.0' not in key:
                if model_dict[key].shape == checkpoint['model'][key].shape:
                    model_dict[key] = checkpoint['model'][key]
                else:
                    print(f"Weight Does Not Match for \'{key}\'!")

    print("Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args, model, criterion, loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        lr_scheduler.step()

        # Saving Checkpoints after each epoch
        if args.output_dir and (not args.is_finetune):
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # Saving Checkpoints every args.save_every_epoch epoch(s)
        if args.output_dir:
            checkpoint_paths = []
            if (epoch + 1) % args.save_every_epoch == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # Evaluation and Logging
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "results.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Meta-DETR', parents=[get_args_parser()])
    # args = parser.parse_args()
    args = parse_args()
    assert args.max_pos_support <= args.total_num_support
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
