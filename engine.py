# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from torchmetrics.classification import MulticlassF1Score
import pandas as pd
import os
from tqdm import tqdm

import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, args, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # if data_iter_step == 0 and utils.is_main_process():
        #     # args.imagenet_default_mean_and_std 에 따른 값
        #     if args.imagenet_default_mean_and_std:
        #         mean = [0.485, 0.456, 0.406]
        #         std  = [0.229, 0.224, 0.225]
        #     else:        # inception mean/std
        #         mean = [0.5, 0.5, 0.5]
        #         std  = [0.5, 0.5, 0.5]

        #     utils.visualize_first_batch(
        #         samples, mean, std,
        #         save_path=os.path.join(args.output_dir, f"first_batch_epoch{epoch}.png"),
        #         nrow=min(8, samples.size(0))
        #     )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

            # if data_iter_step == 0 and utils.is_main_process():
            #     # args.imagenet_default_mean_and_std 에 따른 값
            #     if args.imagenet_default_mean_and_std:
            #         mean = [0.485, 0.456, 0.406]
            #         std  = [0.229, 0.224, 0.225]
            #     else:        # inception mean/std
            #         mean = [0.5, 0.5, 0.5]
            #         std  = [0.5, 0.5, 0.5]

            #     utils.visualize_first_batch(
            #         samples, mean, std,
            #         save_path=os.path.join(args.output_dir, f"first_batch_epoch{epoch}_mixup.png"),
            #         nrow=min(8, samples.size(0))
            #     )


        if use_amp:
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    f1 = MulticlassF1Score(num_classes=args.nb_classes, top_k=1, average='macro', dist_sync_on_step=True).to(device)

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        f1.update(output, target)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    f1_score = f1.compute().item()
    metric_logger.update(f1=f1_score)
    print('* F1 score {f1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(f1=metric_logger.f1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test(data_loader, model, device, args, output_csv_path, sample_submission_path, idx_to_cls: dict):
    model.eval()
    preds = []
    paths = []

    for imgs, img_paths in tqdm(data_loader, desc="Prediction on test dataset"):
        imgs = imgs.to(device, non_blocking=True)

        if args.use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(imgs)
        else:
            output = model(imgs)
        
        pred = output.argmax(dim=1)
        preds.append(pred.cpu())
        paths.extend(img_paths)
    
    preds = torch.cat(preds).numpy()
    pred_classes = [idx_to_cls[idx] for idx in preds]

    submission = pd.read_csv(sample_submission_path)
    path_to_pred = {}

    for path, pred_class in zip(paths, pred_classes):
        f_name = os.path.splitext(os.path.basename(path))[0]
        path_to_pred[f_name] = pred_class
    
    submission['rock_type'] = submission['ID'].map(path_to_pred)

    submission.to_csv(output_csv_path, index=False)
    print(f"Test finished. Submission saved to {output_csv_path}")