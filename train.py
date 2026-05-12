import os
import argparse
import datetime
import time
import json
import random
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import utils
from dataloaders import get_train_dataloader
from models import get_model
from schedulers import get_scheduler
from optimizers import get_optimizer
from losses import get_loss
from engine import train_MTD_GAN_Ours, valid_MTD_GAN_Ours
from module.weight_methods import WeightMethods


def get_args_parser():
    parser = argparse.ArgumentParser('MTD-GAN Deep-Learning Train script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',               default="amc", type=str, help='dataset name')
    parser.add_argument('--dataset-type-train',    default="window_patch", type=str)
    parser.add_argument('--dataset-type-valid',    default="window_patch", type=str)
    parser.add_argument('--batch-size',            default=72, type=int)
    parser.add_argument('--train-num-workers',     default=10, type=int)
    parser.add_argument('--valid-num-workers',     default=10, type=int)

    # Model parameters
    parser.add_argument('--model',                 default='MTD_GAN_Method', type=str, help='model name (only MTD_GAN_Method is supported)')
    parser.add_argument('--loss',                  default='L1 Loss', type=str, help='loss name for validation L1 monitoring')
    parser.add_argument('--method',                default='', help='multi-task weighting name (e.g. pcgrad, cagrad, mgda, nashmtl). Empty = vanilla sum.')

    # Optimizer / Scheduler / Epoch
    parser.add_argument('--optimizer',             default='adamw', type=str)
    parser.add_argument('--scheduler',             default='poly_lr', type=str)
    parser.add_argument('--epochs',                default=1000, type=int)
    parser.add_argument('--warmup-epochs',         default=10, type=int)
    parser.add_argument('--lr',                    default=5e-4, type=float)
    parser.add_argument('--min-lr',                default=1e-5, type=float)

    # GPU
    parser.add_argument('--multi-gpu-mode',        default='DataParallel', choices=['Single', 'DataParallel'], type=str)
    parser.add_argument('--device',                default='cuda', help='device to use for training / testing')

    # Logging
    parser.add_argument('--print-freq',            default=10, type=int)
    parser.add_argument('--save-checkpoint-every', default=1,  type=int)

    # Output
    parser.add_argument('--checkpoint-dir',        default='', help='path where to save checkpoint or output')
    parser.add_argument('--save-dir',              default='', help='path where to prediction PNG save')

    # Resume / Pretrained
    parser.add_argument('--from-pretrained',       default='', help='pre-trained from checkpoint')
    parser.add_argument('--resume',                default='', help='resume from checkpoint')

    # Memo
    parser.add_argument('--memo',                  default='', help='memo for script')
    return parser


# fix random seeds for reproducibility
random_seed = 2024
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def main(args):
    start_epoch = 0
    utils.print_args(args)
    device = torch.device(args.device)
    print("cpu == ", os.cpu_count())

    # Dataloader
    data_loader_train, data_loader_valid = get_train_dataloader(name=args.dataset, args=args)

    # Model
    model = get_model(name=args.model)

    # Multi-GPU (MTD-GAN has Generator + Discriminator)
    if args.multi_gpu_mode == 'DataParallel':
        model.Generator     = torch.nn.DataParallel(model.Generator)
        model.Discriminator = torch.nn.DataParallel(model.Discriminator)
        model.Generator.to(device)
        model.Discriminator.to(device)
    else:
        model.to(device)

    # Loss (validation monitoring only — actual training losses live inside the model)
    loss = get_loss(name=args.loss)

    # Optimizer & Scheduler
    if args.method and not args.resume:
        # Weight Method such as PCGrad, CAGrad, MGDA, NashMTL
        # Ref: https://github.com/AvivNavon/nash-mtl
        weight_methods_parameters = defaultdict(dict)
        weight_methods_parameters.update(dict(
            nashmtl=dict(update_weights_every=1, optim_niter=20),
            stl=dict(main_task=0),
            cagrad=dict(c=0.4),
            dwa=dict(temp=2.0),
        ))
        weight_method_D = WeightMethods(method=args.method, n_tasks=3, device=device, **weight_methods_parameters[args.method])
        optimizer_D = torch.optim.AdamW([
            dict(params=model.Discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False),
            dict(params=weight_method_D.parameters(),     lr=0.025,   betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False),
        ])
        scheduler_D = get_scheduler(name=args.scheduler, optimizer=optimizer_D, args=args)
        optimizer_G = get_optimizer(name=args.optimizer, model=model.Generator, lr=args.lr)
        scheduler_G = get_scheduler(name=args.scheduler, optimizer=optimizer_G, args=args)
    else:
        weight_method_D = None
        optimizer_D = get_optimizer(name=args.optimizer, model=model.Discriminator, lr=args.lr)
        scheduler_D = get_scheduler(name=args.scheduler, optimizer=optimizer_D, args=args)
        optimizer_G = get_optimizer(name=args.optimizer, model=model.Generator, lr=args.lr)
        scheduler_G = get_scheduler(name=args.scheduler, optimizer=optimizer_G, args=args)

    # Resume
    if args.resume:
        print(f"Loading... Resume from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        # 1. Model
        checkpoint['model_state_dict'] = {k.replace('.module', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 2. Weight Method (MTD-GAN Multi-task weights)
        if weight_method_D is not None and 'weight_method_D_params' in checkpoint:
            for current_p, saved_p in zip(weight_method_D.parameters(), checkpoint['weight_method_D_params']):
                current_p.data.copy_(saved_p.to(current_p.device))

        # 3. Optimizers & Schedulers
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        scheduler_D.load_state_dict(checkpoint['scheduler_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        scheduler_G.load_state_dict(checkpoint['scheduler_G'])
        utils.fix_optimizer(optimizer_D)
        utils.fix_optimizer(optimizer_G)

        # 4. Random States (중단된 시점의 배치 순서 및 노이즈 상태 정확히 복원)
        if 'random_state' in checkpoint:
            random.setstate(checkpoint['random_state'])
            np.random.set_state(checkpoint['np_random_state'])
            torch.set_rng_state(checkpoint['torch_rng_state'])
            torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_state'])

        start_epoch = checkpoint['epoch'] + 1
    # Tensorboard
    tensorboard = SummaryWriter(args.checkpoint_dir + '/runs')
    print('Writing Tensorboard logs to ', args.checkpoint_dir + '/runs')

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # Train & Valid loop
    for epoch in range(start_epoch, args.epochs):

        train_stats = train_MTD_GAN_Ours(
            model, data_loader_train, optimizer_G, optimizer_D,
            device, epoch, args.print_freq, args.batch_size, weight_method_D,
        )
        print("Averaged train_stats: ", train_stats)
        for key, value in train_stats.items():
            tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)

        valid_stats = valid_MTD_GAN_Ours(
            model, loss, data_loader_valid, device, epoch, args.save_dir, args.print_freq,
        )
        print("Averaged valid_stats: ", valid_stats)
        for key, value in valid_stats.items():
            tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)

        # LR scheduler update
        scheduler_G.step()
        scheduler_D.step()



# Save checkpoint
        if epoch % args.save_checkpoint_every == 0:
            checkpoint_path = args.checkpoint_dir + '/epoch_' + str(epoch) + '_checkpoint.pth'
            
            save_dict = {
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'scheduler_D': scheduler_D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'scheduler_G': scheduler_G.state_dict(),
                'epoch': epoch,
                'args': args,
                # 랜덤 상태 저장
                'random_state': random.getstate(),
                'np_random_state': np.random.get_state(),
                'torch_rng_state': torch.get_rng_state(),
                'torch_cuda_rng_state': torch.cuda.get_rng_state_all(),
            }
            
            # Weight Method 파라미터가 있다면 함께 저장
            if weight_method_D is not None:
                save_dict['weight_method_D_params'] = [p.data.cpu().clone() for p in weight_method_D.parameters()]

            torch.save(save_dict, checkpoint_path)



        # Log
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'valid_{k}': v for k, v in valid_stats.items()},
            'epoch': epoch,
        }
        with open(args.checkpoint_dir + "/log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # Finish
    tensorboard.close()
    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MTD-GAN training script', parents=[get_args_parser()])
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir + "/args", mode=0o777, exist_ok=True)
    os.makedirs(args.save_dir, mode=0o777, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    args_path = args.checkpoint_dir + "/args/args_" + stamp + ".json"
    if not os.path.isfile(args_path):
        with open(args_path, "w") as f:
            json.dump(args.__dict__, f, indent=2)

    main(args)