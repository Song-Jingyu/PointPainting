"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import sys
import time
import torch

# from apex import amp
from runx.logx import logx
from config import assert_and_infer_cfg, update_epoch, cfg
from utils.misc import AverageMeter, prep_experiment, eval_metrics
from utils.misc import ImageDumper
from utils.trnval_utils import validate_topn, flip_tensor, resize_tensor, fmt_scale
from loss.utils import get_loss
from loss.optimizer import get_optimizer, restore_opt, restore_net

import datasets
import network

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Import autoresume module
sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
AutoResume = None
try:
    from userlib.auto_resume import AutoResume
except ImportError:
    print(AutoResume)

DATA_PATH = "../../detector/data/kitti/training/"

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--arch', type=str, default='ocrnet.HRNet_Mscale',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, mapillary, camvid, kitti')
parser.add_argument('--dataset_inst', default=None,
                    help='placeholder for dataset instance')
parser.add_argument('--num_workers', type=int, default=4,
                    help='cpu worker threads per dataloader instance')

parser.add_argument('--cv', type=int, default=0,
                    help=('Cross-validation split id to use. Default # of splits set'
                          ' to 3 in config'))

parser.add_argument('--class_uniform_pct', type=float, default=0.5,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='Use coarse annotations for specific classes')

parser.add_argument('--custom_coarse_dropout_classes', type=str, default=None,
                    help='Drop some classes from auto-labelling')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--rmi_loss', action='store_true', default=False,
                    help='use RMI loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help=('Batch weighting for class (use nll class weighting using '
                          'batch stats'))

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_epoch', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new lr ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--apex', action='store_true', default=False,
                    help='Use Nvidia Apex Distributed Data Parallel')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')

parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
parser.add_argument('--global_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--amsgrad', action='store_true', help='amsgrad for adam')

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help=('0 means no aug, 1 means hard negative mining '
                          'iter 1, 2 means hard negative mining iter 2'))

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=150,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--brt_aug', action='store_true', default=False,
                    help='Use brightness augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--poly_step', type=int, default=110,
                    help='polynomial epoch step')
parser.add_argument('--bs_trn', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=str, default='896',
                    help=('training crop size: either scalar or h,w'))
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default="./assets/seg_weights/cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth")
parser.add_argument('--resume', type=str, default=None,
                    help=('continue training from a checkpoint. weights, '
                          'optimizer, schedule are restored'))
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--restore_net', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--result_dir', type=str, default=DATA_PATH+"log/",
                    help='where to write log output')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help=('Minimum testing to verify nothing failed, '
                          'Runs code for 1 epoch of train and val'))
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
# Full Crop Training
parser.add_argument('--full_crop_training', action='store_true', default=False,
                    help='Full Crop Training')

# Multi Scale Inference
parser.add_argument('--multi_scale_inference', action='store_true',
                    help='Run multi scale inference')

parser.add_argument('--default_scale', type=float, default=1.0,
                    help='default scale to run validation')

parser.add_argument('--log_msinf_to_tb', action='store_true', default=False,
                    help='Log multi-scale Inference to Tensorboard')

parser.add_argument('--eval', type=str, default='folder',
                    help=('just run evaluation, can be set to val or trn or '
                          'folder'))
parser.add_argument('--eval_folder', type=str, default=DATA_PATH,
                    help='path to frames to evaluate')
parser.add_argument('--three_scale', action='store_true', default=False)
parser.add_argument('--alt_two_scale', action='store_true', default=False)
parser.add_argument('--do_flip', action='store_true', default=False)
parser.add_argument('--extra_scales', type=str, default='0.5,2.0')
parser.add_argument('--n_scales', type=str, default="0.5,1.0,2.0")
parser.add_argument('--align_corners', action='store_true',
                    default=False)
parser.add_argument('--translate_aug_fix', action='store_true', default=False)
parser.add_argument('--mscale_lo_scale', type=float, default=0.5,
                    help='low resolution training scale')
parser.add_argument('--pre_size', type=int, default=None,
                    help=('resize long edge of images to this before'
                          ' augmentation'))
parser.add_argument('--amp_opt_level', default='O1', type=str,
                    help=('amp optimization level'))
parser.add_argument('--rand_augment', default=None,
                    help='RandAugment setting: set to \'N,M\'')
parser.add_argument('--init_decoder', default=False, action='store_true',
                    help='initialize decoder with kaiming normal')
parser.add_argument('--dump_topn', type=int, default=0,
                    help='Dump worst val images')
parser.add_argument('--dump_assets', action='store_true',
                    help='Dump interesting assets')
parser.add_argument('--dump_all_images', action='store_true',
                    help='Dump all images, not just a subset')
parser.add_argument('--dump_for_submission', action='store_true',
                    help='Dump assets for submission')
parser.add_argument('--dump_for_auto_labelling', action='store_true',
                    help='Dump assets for autolabelling')
parser.add_argument('--dump_topn_all', action='store_true', default=False,
                    help='dump topN worst failures')
parser.add_argument('--custom_coarse_prob', type=float, default=None,
                    help='Custom Coarse Prob')
parser.add_argument('--only_coarse', action='store_true', default=False)
parser.add_argument('--mask_out_cityscapes', action='store_true',
                    default=False)
parser.add_argument('--ocr_aspp', action='store_true', default=False)
parser.add_argument('--map_crop_val', action='store_true', default=False)
parser.add_argument('--aspp_bot_ch', type=int, default=None)
parser.add_argument('--trial', type=int, default=None)
parser.add_argument('--mscale_cat_scale_flt', action='store_true',
                    default=False)
parser.add_argument('--mscale_dropout', action='store_true',
                    default=False)
parser.add_argument('--mscale_no3x3', action='store_true',
                    default=False, help='no inner 3x3')
parser.add_argument('--mscale_old_arch', action='store_true',
                    default=False, help='use old attention head')
parser.add_argument('--mscale_init', type=float, default=None,
                    help='default attention initialization')
parser.add_argument('--attnscale_bn_head', action='store_true',
                    default=False)
parser.add_argument('--set_cityscapes_root', type=str, default=None,
                    help='override cityscapes default root dir')
parser.add_argument('--ocr_alpha', type=float, default=None,
                    help='set HRNet OCR auxiliary loss weight')
parser.add_argument('--val_freq', type=int, default=1,
                    help='how often (in epochs) to run validation')
parser.add_argument('--deterministic', action='store_true',
                    default=False)
parser.add_argument('--summary', action='store_true',
                    default=False)
parser.add_argument('--segattn_bot_ch', type=int, default=None,
                    help='bottleneck channels for seg and attn heads')
parser.add_argument('--grad_ckpt', action='store_true',
                    default=False)
parser.add_argument('--no_metrics', action='store_true', default=False,
                    help='prevent calculation of metrics')
parser.add_argument('--supervised_mscale_loss_wt', type=float, default=None,
                    help='weighting for the supervised loss')
parser.add_argument('--ocr_aux_loss_rmi', action='store_true', default=False,
                    help='allow rmi for aux loss')


args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                    'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}


# Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """
    Main Function
    """
    if AutoResume:
        AutoResume.init()

    assert args.result_dir is not None, 'need to define result_dir arg'
    logx.initialize(logdir=args.result_dir,
                    tensorboard=False, hparams=vars(args),
                    global_rank=args.global_rank)

    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    prep_experiment(args)
    train_loader, val_loader, train_obj = datasets.setup_loaders(args)
    criterion, criterion_val = get_loss(args)

    auto_resume_details = None
    if AutoResume:
        auto_resume_details = AutoResume.get_resume_details()

    if auto_resume_details:
        checkpoint_fn = auto_resume_details.get("RESUME_FILE", None)
        checkpoint = torch.load(checkpoint_fn,
                                map_location=torch.device('cpu'))
        args.result_dir = auto_resume_details.get("TENSORBOARD_DIR", None)
        args.start_epoch = int(auto_resume_details.get("EPOCH", None)) + 1
        args.restore_net = True
        args.restore_optimizer = True
        msg = ("Found details of a requested auto-resume: checkpoint={}"
               " tensorboard={} at epoch {}")
        logx.msg(msg.format(checkpoint_fn, args.result_dir,
                            args.start_epoch))
    elif args.resume:
        checkpoint = torch.load(args.resume,
                                map_location=torch.device('cpu'))
        args.arch = checkpoint['arch']
        args.start_epoch = int(checkpoint['epoch']) + 1
        args.restore_net = True
        args.restore_optimizer = True
        msg = "Resuming from: checkpoint={}, epoch {}, arch {}"
        logx.msg(msg.format(args.resume, args.start_epoch, args.arch))
    elif args.snapshot:
        if 'ASSETS_PATH' in args.snapshot:
            args.snapshot = args.snapshot.replace('ASSETS_PATH', cfg.ASSETS_PATH)
        checkpoint = torch.load(args.snapshot,
                                map_location=torch.device('cpu'))
        args.restore_net = True
        msg = "Loading weights from: checkpoint={}".format(args.snapshot)
        logx.msg(msg)

    net = network.get_net(args, criterion)
    optim, scheduler = get_optimizer(args, net)

    net = network.wrap_network_in_dataparallel(net, args.apex)

    if args.restore_optimizer:
        restore_opt(optim, checkpoint)
    if args.restore_net:
        restore_net(net, checkpoint)

    if args.init_decoder:
        net.module.init_mods()

    torch.cuda.empty_cache()

    if args.start_epoch != 0:
        scheduler.step(args.start_epoch)

    if args.eval == 'folder':
        # Using a folder for evaluation means to not calculate metrics
        # validate(val_loader, net, criterion=None, optim=None, epoch=0,
        #          calc_metrics=False, dump_assets=args.dump_assets,
        #          dump_all_images=True)
        if not os.path.exists(args.result_dir+'image_2/'):
            os.mkdir(args.result_dir+'image_2/')
        if not os.path.exists(args.result_dir+'image_3/'):
            os.mkdir(args.result_dir+'image_3/')

        num_image = 7481
        for idx in tqdm(range(num_image)):
            sample_idx = "%06d" % idx
            eval_minibatch(sample_idx, "image_2/", net, args)
            eval_minibatch(sample_idx, "image_3/", net, args)
        
        return 0
    elif args.eval is not None:
        raise 'unknown eval option {}'.format(args.eval)

def eval_minibatch(idx, left, net, args):
    """
    Evaluate a single minibatch of images.
     * calculate metrics
     * dump images

    There are two primary multi-scale inference types:
      1. 'MSCALE', or in-model multi-scale: where the multi-scale iteration loop is
         handled within the model itself (see networks/mscale.py -> nscale_forward())
      2. 'multi_scale_inference', where we use Averaging to combine scales
    """
    filename = args.eval_folder + left + ('%s.png' % idx)
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    images = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    net.eval()
    torch.cuda.empty_cache()

    scales = [args.default_scale]
    if args.multi_scale_inference:
        scales.extend([float(x) for x in args.extra_scales.split(',')])

    # input    = torch.Size([1, 3, h, w])
    # gt_image = torch.Size([1, h, w])
    # images, gt_image, img_names, scale_float = data
    # images = data
    # assert len(images.size()) == 4 and len(gt_image.size()) == 3
    # assert images.size()[2:] == gt_image.size()[1:]
    input_size = images.size(2), images.size(3)

    if args.do_flip:
        # By ending with flip=0, we insure that the images that are dumped
        # out correspond to the unflipped versions. A bit hacky.
        flips = [1, 0]
    else:
        flips = [0]

    with torch.no_grad():
        output = 0.0

        for flip in flips:
            for scale in scales:
                if flip == 1:
                    inputs = flip_tensor(images, 3)
                else:
                    inputs = images

                infer_size = [round(sz * scale) for sz in input_size]

                if scale != 1.0:
                    inputs = resize_tensor(inputs, infer_size)

                inputs = {'images': inputs}
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Expected Model outputs:
                #   required:
                #     'pred'  the network prediction, shape (1, 19, h, w)
                #
                #   optional:
                #     'pred_*' - multi-scale predictions from mscale model
                #     'attn_*' - multi-scale attentions from mscale model
                output_dict = net(inputs)

                _pred = output_dict['pred']

                # save AVGPOOL style multi-scale output for visualizing
                if not cfg.MODEL.MSCALE:
                    scale_name = fmt_scale('pred', scale)
                    output_dict[scale_name] = _pred

                # resize tensor down to 1.0x scale in order to combine
                # with other scales of prediction
                if scale != 1.0:
                    _pred = resize_tensor(_pred, input_size)

                if flip == 1:
                    output = output + flip_tensor(_pred, 3)
                else:
                    output = output + _pred

    output = output / len(scales) / len(flips)
    # assert_msg = 'output_size {} gt_cuda size {}'
    # gt_cuda = gt_image.cuda()
    # assert_msg = assert_msg.format(
    #     output.size()[2:], gt_cuda.size()[1:])
    # assert output.size()[2:] == gt_cuda.size()[1:], assert_msg
    assert output.size()[1] == cfg.DATASET.NUM_CLASSES#, assert_msg

    output_data = torch.nn.functional.softmax(output, dim=1).cpu().data
    # max_probs, predictions = output_data.max(1)

    # # Assemble assets to visualize
    # assets = {}
    # for item in output_dict:
    #     if 'attn_' in item:
    #         assets[item] = output_dict[item]
    #     if 'pred_' in item:
    #         smax = torch.nn.functional.softmax(output_dict[item], dim=1)
    #         _, pred = smax.data.max(1)
    #         assets[item] = pred.cpu().numpy()

    # predictions = predictions.numpy()
    # assets['predictions'] = predictions
    # assets['prob_mask'] = max_probs
    # assets['scores'] = output_data

    output_permute = torch.tensor(output_data[0]).permute(1,2,0) # H, W, 19


    output_reassign = torch.zeros(output_permute.size(0),output_permute.size(1), 5)
    output_reassign[:,:,0], _ = torch.max(output_permute[:,:,:11], dim=2) # background
    output_reassign[:,:,1], _ = torch.max(output_permute[:,:,[17, 18]], dim=2) # bicycle
    output_reassign[:,:,2], _ = torch.max(output_permute[:,:,[13, 14, 15, 16]], dim=2) # car
    output_reassign[:,:,3] = output_permute[:,:,11] #person
    output_reassign[:,:,4] = output_permute[:,:,12] #rider
    sf = torch.nn.Softmax(dim=2)
    output_reassign_softmax = sf(output_reassign).cpu().numpy()

    np.save(args.result_dir + left + ("%s.npy" % idx), output_reassign_softmax)

    return output_reassign_softmax
    # return assets

if __name__ == '__main__':
    main()
