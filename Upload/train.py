from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import logging
import argparse
import sys
import cv2
import numpy as np
from torch import optim
from PIL import Image
from tqdm import tqdm
from model import AMD_HookNet_plus
from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
from valid_dataset import BasDataset
from eval_complete import eval_net
from torch.utils.data import DataLoader
from config import get_config, cfg
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss, PixelContrastLoss

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='./checkpoints', help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=130, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=170, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='./swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE",
                    help='path to config file', )
parser.add_argument('--resume', default='./swin_tiny_patch4_window7_224.pth', help='resume from checkpoint')
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
args = parser.parse_args()
config = get_config(args)

dir_checkpoint = cfg.output_path

scratch_train = os.path.join(os.environ['TMPDIR'], 'Training')
dir_img_target = os.path.join(scratch_train, 'target_images')
dir_img_context = os.path.join(scratch_train, 'context_images')
dir_mask_target = os.path.join(scratch_train, 'target_masks')
dir_mask_context = os.path.join(scratch_train, 'context_masks')

scratch_valid = os.path.join(os.environ['TMPDIR'], 'Validation')
valid_img_target = os.path.join(scratch_valid, 'target_images')
valid_img_context = os.path.join(scratch_valid, 'context_images')
valid_mask_target = os.path.join(scratch_valid, 'target_masks')
valid_mask_context = os.path.join(scratch_valid, 'context_masks')


def seed_torch(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cudnn.benchmark = True
    cudnn.deterministic = False

    cfg.in_channels = 3
    cfg.n_filters = 32
    cfg.batch_size = 170
    args.base_lr = 0.01
    args.num_classes = 4
    args.max_epochs = 130

    db_train = BasicDataset(dir_img_target=dir_img_target, dir_mask_target=dir_mask_target,
                            dir_img_context=dir_img_context, dir_mask_context=dir_mask_context, scale=1.0)
    n_dataset = len(db_train)
    print("The length of train set is: {}".format(len(db_train)))
    train_loader = DataLoader(db_train, batch_size=cfg.batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=False)
    valid_dataset = BasDataset(dir_img_target=valid_img_target, dir_mask_target=valid_mask_target,
                               dir_img_context=valid_img_context, dir_mask_context=valid_mask_context, scale=1.0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True,
                              drop_last=False)
    model = AMD_HookNet_plus(cfg.in_channels, cfg.filter_size, cfg.n_filters, config, img_size=args.img_size,
                             num_classes=args.num_classes)
    model.to(device=device)
    model.load_from(config)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    max_iterations = 150 * len(train_loader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    contrastive_deep_supervision = PixelContrastLoss(temperature=1.0, base_temperature=6.0, max_samples=1024, max_views=8)

    load_path = None

    if load_path:
        checkpoints = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoints['model_state_dict'])
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        epoch = checkpoints['epoch']
        iter_num = checkpoints['iter_num']
        iterator = tqdm(range(args.max_epochs - epoch), ncols=70)
        print(f'Model loaded from {load_path}')
    else:
        iterator = tqdm(range(args.max_epochs), ncols=70)
        epoch = 0
        iter_num = 0

    for epoch_num in iterator:
        epoch_loss = 0
        print('\n' + 'lr:', optimizer.param_groups[0]['lr'])
        for i_batch, batch in enumerate(train_loader):
            image_target, image_context, label_target, label_context = (batch['image_target'].cuda(non_blocking=True),
                                                                        batch['image_context'].cuda(non_blocking=True),
                                                                        batch['mask_target'].cuda(non_blocking=True),
                                                                        batch['mask_context'].cuda(non_blocking=True))
            outputs = model(image_context, image_target)
            c_loss1 = ce_loss(outputs[0], label_context)
            t_loss1 = ce_loss(outputs[1], label_target)
            c_loss2 = dice_loss(outputs[0], label_context, softmax=True)
            t_loss2 = dice_loss(outputs[1], label_target, softmax=True)
            c_loss = c_loss1 + c_loss2
            t_loss = t_loss1 + t_loss2
            loss_cds = 0
            masks = [label_target.cpu()]
            for i in range(4):
                big_mask = masks[-1]
                small_mask = F.avg_pool2d(big_mask, 2)
                masks.append(small_mask)
            small = masks[2:]
            prd_target = F.log_softmax(outputs[1], dim=1)
            prd_target = torch.argmax(prd_target, dim=1)
            t_masks = [prd_target.cpu()]
            for i in range(4):
                big_mask = t_masks[-1]
                small_mask = F.avg_pool2d(big_mask, 2)
                t_masks.append(small_mask)
            t_small = t_masks[2:]
            for i in range(len(outputs[2])):
                predict = t_small[i].cuda().type(torch.int64)
                loss_cds += contrastive_deep_supervision(outputs[2][1 - i], small[i].cuda().type(torch.int64), predict)
            loss = t_loss + c_loss + 0.5 * loss_cds
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1

        print('Train average loss:', epoch_loss / n_dataset)
        piou, valid_iou_ratio, iou = eval_net(model, valid_loader, device)
        print('Valid IoU ratio: {}, {}'.format(valid_iou_ratio, iou))

        if load_path:
            save_mode_path = os.path.join(args.output_dir,
                                          'AMD_HookNet_plus_epoch{:03d}.pth'.format(epoch_num + epoch + 1))
            print("save model to {}".format(save_mode_path))

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch_num + epoch + 1,
                        'iter_num': iter_num,
                        }, save_mode_path)

        else:
            save_mode_path = os.path.join(args.output_dir, 'AMD_HookNet_plus_epoch{:03d}.pth'.format(epoch_num + 1))
            print("save model to {}".format(save_mode_path))

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch_num + 1,
                        'iter_num': iter_num,
                        }, save_mode_path)
    print("Training Finished!")
