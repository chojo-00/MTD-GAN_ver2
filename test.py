import os
import argparse
import datetime
import time
import json
import random
import torch
import numpy as np

import utils
from dataloaders import get_test_dataloader
from models import get_model
from losses import get_loss
from engine import test_MTD_GAN_Ours


def get_args_parser():
    parser = argparse.ArgumentParser('MTD-GAN Deep-Learning Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',               default="amc", type=str, help='dataset name')
    parser.add_argument('--dataset-type-test',     default="window_patch", type=str)
    parser.add_argument('--test-batch-size',       default=72, type=int)
    parser.add_argument('--test-num-workers',      default=10, type=int)

    # Model parameters
    parser.add_argument('--model',                 default='MTD_GAN_Method', type=str, help='model name (only MTD_GAN_Method is supported)')
    parser.add_argument('--loss',                  default='L1 Loss', type=str, help='loss name for L1 monitoring')
    parser.add_argument('--method',                default='', help='multi-task weighting name (kept for arg-compat; unused at test time)')

    # GPU
    parser.add_argument('--multi-gpu-mode',        default='DataParallel', choices=['Single', 'DataParallel'], type=str)
    parser.add_argument('--device',                default='cuda', help='device to use for training / testing')

    # Resume
    parser.add_argument('--resume',                default='', help='resume from checkpoint')

    # Logging
    parser.add_argument('--print-freq',            default=10, type=int)

    # Output
    parser.add_argument('--checkpoint-dir',        default='', help='path where to save checkpoint or output')
    parser.add_argument('--save-dir',              default='', help='path where to prediction PNG save')
    parser.add_argument('--epoch',                 default=10, type=int)

    # Memo
    parser.add_argument('--memo',                  default='', help='memo for script')
    return parser


# fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    print('Available CPUs: ', os.cpu_count())
    utils.print_args_test(args)
    device = torch.device(args.device)

    # Dataloader
    data_loader_test = get_test_dataloader(name=args.dataset, args=args)

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

    # Loss
    loss = get_loss(name=args.loss)

    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint['model_state_dict'] = {k.replace('.module', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(checkpoint['model_state_dict'])

    start_time = time.time()

    test_stats = test_MTD_GAN_Ours(model, loss, data_loader_test, device, args.save_dir)
    print("Averaged test stats: ", test_stats)

    # Log
    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': args.epoch}
    with open(args.checkpoint_dir + "/test_log.txt", "a") as f:
        f.write(json.dumps(log_stats) + "\n")

    print('***********************************************')
    print("Finish...!")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('TEST time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MTD-GAN evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir + "/args", exist_ok=True)
    os.makedirs(args.save_dir, mode=0o777, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    args_path = args.checkpoint_dir + "/args/test_args_" + stamp + ".json"
    if not os.path.isfile(args_path):
        with open(args_path, "w") as f:
            json.dump(args.__dict__, f, indent=2)

    main(args)