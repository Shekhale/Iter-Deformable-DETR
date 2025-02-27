# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import torch
from common import *
# import datasets
# import misc as utils
# from torch.utils.data import DataLoader
# from datasets.coco import construct_dataset
# from datasets import get_coco_api_from_dataset
from detr import build_model
import pdb

import numpy as np
from datasets.coco import make_coco_transforms
from util.misc import nested_tensor_from_tensor_list


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # Input
    parser.add_argument('--input_image_path', type=str, default=None,
                        help="Path to the input image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--use_checkpoint', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--end_epoch', default=120, type=int, metavar='N',
                        help='end epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--num_gpus', default=4, type=int)

    parser.add_argument('--dense_query', default=0, type=int)
    parser.add_argument('--rectified_attention', default=0, type=int)

    parser.add_argument('--aps', default=0, type=int)

    return parser


def deplicate(record, thr):
    assert 'scores' in record
    names = [k for (k, v) in record.items()]
    flag = record['scores'] >= thr
    for name in names:
        record[name] = record[name][flag]
    return record


# import datasets.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


# def make_coco_transforms():
#     normalize = F.Compose([
#         F.ToTensor(),
#         F.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     return F.Compose([
#         F.RandomResize([800], max_size=1333),
#         normalize,
#     ])


def load_image(args):
    # img = Image.open(os.path.join(self.root, path)).convert('RGB')
    img = Image.open(args.input_image_path).convert('RGB')
    return img


def deplicate(record, thr):
    assert 'scores' in record
    names = [k for (k, v) in record.items()]
    flag = record['scores'] >= thr
    for name in names:
        record[name] = record[name][flag]
    return record


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # model_file = osp.join(model_dir, 'checkpoint-{}.pth'.format(epoch))

    model_file = args.frozen_weights
    # results = multi_process(processor, record, args.num_gpus, model_file, args)

    # torch.cuda.set_device(device)
    device = "cuda"
    model, _, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    # dataset_val = construct_dataset(fpath, 'val', args)
    checkpoint = torch.load(model_file, map_location='cpu')
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    # if utils.is_main_process():
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    model.eval()
    model_without_ddp.to(args.device)
    model_without_ddp.eval()
    transform_fn = make_coco_transforms("val")
    counter, thr = 0, 0.05
    sample = load_image(args)
    origin_size = [sample.size[1], sample.size[0]]
    # sample = torch.tensor(np.asarray(load_image(args)))
    # sample = np.asarray(load_image(args))
    print(type(sample))
    sample = transform_fn(sample)[0]

    print(sample.shape)
    sample = sample.to(args.device)
    print(sample.shape)
    # sample = torch.nested.nested_tensor([sample])
    sample = nested_tensor_from_tensor_list([sample])

    with torch.no_grad():
        outputs = model_without_ddp(sample)  # .unsqueeze(0))
    print(type(outputs))

    orig_target_sizes = torch.stack([torch.tensor(origin_size)], dim=0).to(args.device)
    # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    results = postprocessors['bbox'](outputs, orig_target_sizes)
    results = [deplicate(r, thr) for r in results]
    print(results)

    # targets = [{k:v.cpu().numpy() for k, v in t.items()} for t in targets]
    results = [{k: v.cpu().numpy() for k, v in r.items()} for r in results]
    # dtboxes = [np.hstack([r['boxes'], r['scores'][:, np.newaxis]]) for r in results]
    # dtboxes = [boxes_dump(db) for db in dtboxes]
    # res = [{'ID':name, 'dtboxes':db} for name, db in zip(filenames, dtboxes)]

    #     targets = [{k: v.cpu().numpy() for k, v in t.items()} for t in targets]
    #     results = [{k: v.cpu().numpy() for k, v in r.items()} for r in results]
    #     dtboxes = [np.hstack([r['boxes'], r['scores'][:, np.newaxis]]) for r in results]
    #     dtboxes = [boxes_dump(db) for db in dtboxes]
    #     res = [{'ID': name, 'dtboxes': db} for name, db in zip(filenames, dtboxes)]
    #     assert len(res) == 1
    #     result_queue.put_nowait(res[0])
    # os.remove(fpath)
    # return result_list


if __name__ == '__main__':
    main()
