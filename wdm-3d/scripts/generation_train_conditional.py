"""
A script for training a diffusion model to conditional image generation.
The condition is the segmentation label. 
In this case we consider 3 channels to the condition: 
    -> CT head and neck cancer: 0,1 contrast, non contrast / 2 segmentation  
    -> BraTS: 3 channel label from MONAI
"""

import nibabel as nib
import warnings
import argparse
import numpy as np
import random
import sys
import torch as th

sys.path.append(".")
sys.path.append("..")

from utils import data_inpaint_utils
from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.lidcloader import LIDCVolumes
from guided_diffusion.hnnloader import HnNVolumes
from guided_diffusion.c_bratsloader import c_BraTSVolumes
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.c_script_util import (model_and_diffusion_defaults,
                                          create_model_and_diffusion,
                                          args_to_dict,
                                          add_dict_to_argparser)
from guided_diffusion.c_train_util import TrainLoop
from torch.utils.tensorboard import SummaryWriter
import datetime

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        day = now.day
        hour = str(now.hour).zfill(2)
        minute = str(now.minute).zfill(2)
        second = str(now.second).zfill(2)
        args.use_dilation = True if (args.use_dilation=="True" or args.use_dilation==True) else False
        args.use_data_augmentation = True if (args.use_data_augmentation=="True" or args.use_data_augmentation==True) else False # Use for tumour inpainting 
        args.ROI_DataAug = True if (args.ROI_DataAug=="True" or args.ROI_DataAug==True) else False # Used for the CT generator

        name_completetion=""
        if args.use_dilation:
            name_completetion = name_completetion+"_dilated_always_known"
        if args.use_data_augmentation or args.ROI_DataAug:
            name_completetion = name_completetion+"_DA"
            
        logdir = f"./runs/{args.dataset}_{args.modality}_{args.train_mode}_{name_completetion}_tumorW_{args.tumour_weight}_{day}_{month}_{year}_{hour}:{minute}:{second}"
        
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()
    
    args.devices = [th.cuda.current_device()]
    args.use_label_cond = True if (args.use_label_cond=="True" or args.use_label_cond_conv==True) else False
    args.use_label_cond_conv = True if (args.use_label_cond_conv=="True" or args.use_label_cond_conv==True) else False
    args.remove_tumour_from_loss = True if (args.remove_tumour_from_loss=="True" or args.remove_tumour_from_loss==True) else False
    args.use_wavelet = True if (args.use_wavelet=="True" or args.use_wavelet==True) else False
    
    dist_util.setup_dist(devices=args.devices)
    print(f"Devices: {args.devices}")

    logger.log("Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    print(f"arguments: {arguments}")
    model, diffusion = create_model_and_diffusion(**arguments)
    

    # logger.log("Number of trainable parameters: {}".format(np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    
    
    if args.dataset == 'hnn':
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        dl, ds = HnNVolumes(
            args=args,
            directory=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize=(lambda x: 2*x - 1) if args.renormalize else None,
            mode='train',
            img_size=args.image_size,
            no_seg=args.no_seg,
            full_background=args.full_background,
            clip_min=args.clip_min,
            clip_max=args.clip_max).get_dl_ds()
    
    elif args.dataset == 'c_brats':
        assert args.image_size in [128, 256], "We currently just support image sizes: 128, 256"
        dl, ds = c_BraTSVolumes(
            directory=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_key=args.modality,
            normalize=(lambda x: 2*x - 1) if args.renormalize else None,
            mode='train',
            img_size=args.image_size,
            no_seg=args.no_seg).get_dl_ds()
        if args.full_background:
            warnings.warn("full_background set to True but it is not used in the BraTS dataset", UserWarning)
    
    elif args.dataset == 'hnn_tumour_inpainting':
        args.csv_path = args.data_dir
        dl = data_inpaint_utils.get_loader(args)

    else:
        print("We currently just support the datasets: brats, lidc-idri, hnn, c_brats")
    
    if args.dataset == 'hnn' or args.dataset == 'c_brats' or args.dataset == 'hnn_tumour_inpainting':
        datal = dl
    else:
        datal = th.utils.data.DataLoader(ds,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=True,
                                        )
    logger.log("Settings")        
    logger.log(f"Settings: {args}")
    logger.log("Start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode=args.train_mode,
        tumour_weight=args.tumour_weight,
        label_cond_noise=args.label_cond_noise,
        modality=args.modality,
        data_seg_augment=args.data_seg_augment,
        use_label_cond=args.use_label_cond,
        use_wavelet=args.use_wavelet,
        use_dilation=args.use_dilation,
        use_label_cond_conv=args.use_label_cond_conv,
        remove_tumour_from_loss=args.remove_tumour_from_loss
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        use_tensorboard=True,
        tensorboard_path='',  # set path to existing logdir for resuming
        devices=None,
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode=None,
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        tumour_weight=0,
        label_cond_noise=False,
        no_seg=False,
        full_background=False,
        modality=None,
        data_seg_augment=None,
        clip_min=None,
        clip_max=None,
        use_label_cond=None,
        use_label_cond_conv=None,
        train_mode=None,
        use_wavelet=None,
        use_dilation=None,
        use_data_augmentation=None,
        ROI_DataAug=None,
        remove_tumour_from_loss=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
