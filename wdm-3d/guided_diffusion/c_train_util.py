import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.utils.tensorboard
from torch.optim import AdamW
import torch.cuda.amp as amp

import itertools

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

from scipy import ndimage
import math

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        in_channels,
        image_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset='brats',
        summary_writer=None,
        mode='default',
        loss_level='image',
        tumour_weight=0,
        label_cond_noise=False,
        modality=None,
        data_seg_augment=None,
        use_label_cond=None,
        use_wavelet=None,
        use_dilation=None,
        use_label_cond_conv=None,
        remove_tumour_from_loss=False,
    ):
        self.summary_writer = summary_writer
        self.mode = mode
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.dataset = dataset
        self.iterdatal = iter(data)
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.modality = modality
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.grad_scaler = th.amp.GradScaler('cuda')
        else:
            self.grad_scaler = th.amp.GradScaler('cuda',enabled=False)

        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D('haar')

        self.loss_level = loss_level
        self.tumour_weight = tumour_weight
        self.label_cond_noise = label_cond_noise
        self.data_seg_augment = data_seg_augment 
        self.use_label_cond = use_label_cond
        self.use_wavelet = use_wavelet
        self.use_dilation = use_dilation
        self.use_label_cond_conv = use_label_cond_conv
        self.remove_tumour_from_loss = remove_tumour_from_loss

        self.step = 1
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        load_optimizer = self._load_and_sync_parameters()

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step and load_optimizer:
            print("Resume Step: " + str(self.resume_step))
            self._load_optimizer_state()

        if not th.cuda.is_available():
            logger.warn(
                "Training requires CUDA. "
            )

    def _load_and_sync_parameters(self):
        load_optimizer = True
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            print('resume model ...')
            mismatched_keys = []
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                load_checkpoint = dist_util.load_state_dict(resume_checkpoint, map_location="cpu") 
                # Get the state dictionary of the new model
                model_dict = self.model.state_dict()
                # Filter out unnecessary keys based on shape mismatch and collect mismatched keys
                for k, v in load_checkpoint.items():
                    if k in model_dict:
                        if v.shape != model_dict[k].shape:
                            mismatched_keys.append(k)
                            mismatched_keys.append(f"{k.split('.weight')[0]}.bias")
                    else:
                        mismatched_keys.append(k)
                        mismatched_keys.append(f"{k.split('.weight')[0]}.bias")
                if len(mismatched_keys)>0:
                    import warnings
                    warnings.warn(f"The saved weights {mismatched_keys} do not match the shape of the network. These will be ignored and set un-trained.\nThe optimizer will not be loaded")
                    logger.log(f"The saved weights {mismatched_keys} do not match the shape of the network. These will be ignored and set un-trained.\The optimizer will not be loaded")
             
                    load_optimizer = False
                else:
                    load_optimizer = True
               # Remove mismatched keys from the pretrained dictionary
                pretrained_dict = {k: v for k, v in load_checkpoint.items() if k not in mismatched_keys}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(pretrained_dict, strict=False)

        dist_util.sync_params(self.model.parameters())
        return load_optimizer



    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
            print(f"Optimizer resumed")
        
        else:
            print('no optimizer checkpoint exists')

    def run_loop(self):
        import time
        t = time.time()
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            t_total = time.time() - t
            t = time.time()
            if self.dataset in ['brats', 'hnn', 'c_brats', 'hnn_tumour_inpainting']:
                try:
                    batch = next(self.iterdatal)
                    cond = {}
                except StopIteration:
                    self.iterdatal = iter(self.datal)
                    batch = next(self.iterdatal)
                    cond = {}
            
            if self.dataset=='hnn':
                image = batch["image"].to(dist_util.dev())
                if self.use_label_cond:
                    label_cond = batch["seg"].to(dist_util.dev())
                    # When computing the loss function, if segmentation!=None it will remove the tumour from the loss computation
                    if self.remove_tumour_from_loss:
                        segmentation = label_cond[:,2:3,:,:,:]
                    else:
                        segmentation = None 
                else:
                    label_cond = None
                channel_of_interest = 2
            elif self.dataset=='c_brats':
                image = batch[self.modality].to(dist_util.dev())
                if self.use_label_cond:
                    label_cond = batch["seg"].to(dist_util.dev())
                    if self.remove_tumour_from_loss:
                        raise ValueError(f'remove_tumour_from_loss not implemented for  c_brats dataset')
                    else:
                        segmentation = None
                else:
                    label_cond = None
                channel_of_interest = 1
            elif self.dataset=='hnn_tumour_inpainting':
                image = batch["scan_volume_crop_pad"].to(dist_util.dev())
                segmentation = None
                if self.use_label_cond:
                    no_contrast_tensor = batch["no_contrast_tensor"].to(dist_util.dev())
                    contrast_tensor = batch["contrast_tensor"].to(dist_util.dev())
                    label_crop_pad = batch["label_crop_pad"].to(dist_util.dev())
                    if self.use_dilation:
                        label_crop_pad_dilated = batch["label_crop_pad_dilated"].to(dist_util.dev())
                        label_cond = torch.cat((no_contrast_tensor, contrast_tensor, label_crop_pad, label_crop_pad_dilated), dim=1)
                    else:
                        label_cond = torch.cat((no_contrast_tensor, contrast_tensor, label_crop_pad), dim=1)
                else:
                    label_cond = None
            else:
                batch = batch.to(dist_util.dev())
                channel_of_interest = 2

            t_fwd = time.time()
            t_load = t_fwd-t

            lossmse, sample, sample_idwt, label_cond = self.run_step(batch=image, cond=cond, segmentation=segmentation, label_cond=label_cond)

            t_fwd = time.time()-t_fwd


            if self.summary_writer is not None:
                self.summary_writer.add_scalar('time/load', t_load, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/forward', t_fwd, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/total', t_total, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/MSE', lossmse.item(), global_step=self.step + self.resume_step)

            if self.step % 200 == 0: 
                print(f"Saving ")
                # Copmputing the center of mass of the label of the channel_of_interest (seg in case of hnn and WT in case of Brats)
                x_center, y_center, z_center = ndimage.measurements.center_of_mass(label_cond.cpu().numpy()[0][-1]) 
                # Saving the image used generated    
                """
                    # Uncomment if want coronal and sagittal
                    if math.isnan(x_center):
                        x_center = sample_idwt[0].size()[0]//2
                    if math.isnan(y_center):
                        y_center = sample_idwt[0].size()[1]//2
                """
                if math.isnan(z_center):
                    z_center = label_cond.shape[-1]//2

                midplane = image[0, 0, :, :, int(z_center)]
                self.summary_writer.add_image(f'sample/real', midplane.unsqueeze(0),
                                                global_step=self.step + self.resume_step)

                midplane = sample_idwt[0, 0, :, :, int(z_center)]
                self.summary_writer.add_image(f'sample/x_{self.modality}_0_a', midplane.unsqueeze(0),
                                                global_step=self.step + self.resume_step)
                """
                    # Uncomment if want coronal and sagittal
                    midplane = sample_idwt[0, 0, :, int(y_center), :]
                    self.summary_writer.add_image('sample/x_{modal}_0_c', midplane.unsqueeze(0),
                                                    global_step=self.step + self.resume_step)

                    midplane = sample_idwt[0, 0, int(x_center), :, :]
                    self.summary_writer.add_image('sample/x_{modal}_0_s', midplane.unsqueeze(0),
                                                    global_step=self.step + self.resume_step)
                """
                #######################################################################
                if self.use_label_cond:
                    # Saving all channels of labels used as input
                    for ch_idx, ch_label in enumerate(label_cond[0]):
                        midplane = ch_label[:, :, int(z_center)]
                        self.summary_writer.add_image(f'sample/l_{ch_idx}_a', midplane.unsqueeze(0),
                                                    global_step=self.step + self.resume_step)
                        """
                            # Uncomment if want coronal and sagittal
                            midplane = ch_label[:, int(y_center), :]
                            self.summary_writer.add_image(f'sample/l_{ch_idx}_s', midplane.unsqueeze(0),
                                                        global_step=self.step + self.resume_step)

                            midplane = ch_label[int(x_center), :, :]
                            self.summary_writer.add_image(f'sample/l_{ch_idx}_c', midplane.unsqueeze(0),
                                                        global_step=self.step + self.resume_step)
                        """
                else:
                    pass
                #######################################################################

                image_size = sample.size()[2]
                if self.use_wavelet:
                    names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                    for ch in range(len(names)):
                        midplane = sample[0, ch, :, :, image_size // 2]
                        self.summary_writer.add_image('sample/{}'.format(names[ch]), midplane.unsqueeze(0),
                                                    global_step=self.step + self.resume_step)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()


    def run_loop_OLD(self):
        import time
        t = time.time()
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            t_total = time.time() - t
            t = time.time()
            if self.dataset in ['brats', 'hnn', 'c_brats', 'hnn_tumour_inpainting']:
                try:
                    batch = next(self.iterdatal)
                    cond = {}
                except StopIteration:
                    self.iterdatal = iter(self.datal)
                    batch = next(self.iterdatal)
                    cond = {}
            
            if self.dataset=='hnn':
                image = batch["image"].to(dist_util.dev())
                if self.use_label_cond:
                    label_cond = batch["seg"].to(dist_util.dev())
                else:
                    label_cond = None
                channel_of_interest = 2
            elif self.dataset=='c_brats':
                if len(self.modality.split("_")) > 1:
                    image = []
                    for modal in self.modality.split("_"):
                        image.append(batch[modal].to(dist_util.dev()))
                else:
                    image = batch[self.modality].to(dist_util.dev())
                if self.use_label_cond:
                    label_cond = batch["seg"].to(dist_util.dev())
                else:
                    label_cond = None
                channel_of_interest = 1
            else:
                batch = batch.to(dist_util.dev())
                channel_of_interest = 2

            t_fwd = time.time()
            t_load = t_fwd-t    
            

            lossmse, sample, sample_idwt, label_cond = self.run_step(batch=image, cond=cond, label_cond=label_cond)

            t_fwd = time.time()-t_fwd

            names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"] * len(self.modality.split("_"))

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('time/load', t_load, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/forward', t_fwd, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/total', t_total, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/MSE', lossmse.item(), global_step=self.step + self.resume_step)

            if self.step % 200 == 0: 
                print(f"Saving ")
                # Copmputing the center of mass of the label of the channel_of_interest (seg in case of hnn and WT in case of Brats)
                x_center, y_center, z_center = ndimage.measurements.center_of_mass(label_cond.cpu().numpy()[0][1])
                # Saving the image used generated 
                for modal_idx, modal in enumerate(self.modality.split("_")):
                    """
                    # Uncomment if want coronal and sagittal
                    if math.isnan(x_center):
                        x_center = sample_idwt[0].size()[0]//2
                    if math.isnan(y_center):
                        y_center = sample_idwt[0].size()[1]//2
                    """
                    
                    midplane = sample_idwt[modal_idx][0, 0, :, :, int(z_center)]
                    self.summary_writer.add_image(f'sample/x_{modal}_0_a', midplane.unsqueeze(0),
                                                  global_step=self.step + self.resume_step)
                    """
                    # Uncomment if want coronal and sagittal
                    midplane = sample_idwt[0, 0, :, int(y_center), :]
                    self.summary_writer.add_image('sample/x_{modal}_0_c', midplane.unsqueeze(0),
                                                  global_step=self.step + self.resume_step)

                    midplane = sample_idwt[0, 0, int(x_center), :, :]
                    self.summary_writer.add_image('sample/x_{modal}_0_s', midplane.unsqueeze(0),
                                                  global_step=self.step + self.resume_step)
                    """
                #######################################################################
                if self.use_label_cond:
                    # Saving all channels of labels used as input
                    for ch_idx, ch_label in enumerate(label_cond[0]):
                        midplane = ch_label[:, :, int(z_center)]
                        self.summary_writer.add_image(f'sample/l_{ch_idx}_a', midplane.unsqueeze(0),
                                                    global_step=self.step + self.resume_step)
                        """
                        # Uncomment if want coronal and sagittal
                        midplane = ch_label[:, int(y_center), :]
                        self.summary_writer.add_image(f'sample/l_{ch_idx}_s', midplane.unsqueeze(0),
                                                    global_step=self.step + self.resume_step)

                        midplane = ch_label[int(x_center), :, :]
                        self.summary_writer.add_image(f'sample/l_{ch_idx}_c', midplane.unsqueeze(0),
                                                    global_step=self.step + self.resume_step)
                        """
                else:
                    pass
                #######################################################################

                image_size = sample.size()[2]
                for ch in range(8*len(self.modality.split("_"))):
                    midplane = sample[0, ch, :, :, image_size // 2]
                    self.summary_writer.add_image('sample/{}'.format(names[ch]), midplane.unsqueeze(0),
                                                  global_step=self.step + self.resume_step)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, segmentation=None, label_cond=None, label=None, info=dict()):
        # label_cond added
        lossmse, sample, sample_idwt, label_cond = self.forward_backward(
            batch=batch, 
            cond=cond, 
            label_cond=label_cond, 
            segmentation=segmentation, 
            label=label
            )

        if self.use_fp16:
            self.grad_scaler.unscale_(self.opt)  # check self.grad_scaler._per_optimizer_states

        # compute norms
        with torch.no_grad():
            param_max_norm = max([p.abs().max().item() for p in self.model.parameters()])
            grad_max_norm = max([p.grad.abs().max().item() for p in self.model.parameters()])
            info['norm/param_max'] = param_max_norm
            info['norm/grad_max'] = grad_max_norm

        if not torch.isfinite(lossmse): #infinite
            if not torch.isfinite(torch.tensor(param_max_norm)):
                logger.log(f"Model parameters contain non-finite value {param_max_norm}, entering breakpoint", level=logger.ERROR)
                breakpoint()
            else:
                logger.log(f"Model parameters are finite, but loss is not: {lossmse}"
                           "\n -> update will be skipped in grad_scaler.step()", level=logger.WARN)

        if self.use_fp16:
            print("Use fp16 ...")
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            info['scale'] = self.grad_scaler.get_scale()
        else:
            self.opt.step()
        self._anneal_lr()
        self.log_step()
        return lossmse, sample, sample_idwt, label_cond

    def forward_backward(self, batch, cond, label_cond, segmentation, label=None):
        for p in self.model.parameters():  # Zero out gradient
            p.grad = None

        t, weights = self.schedule_sampler.sample(self.batch_size, dist_util.dev())
        
        # The label used as condition is fed separately from the image.
        # A convolution is used to reduce the dimention of the label to half of the original shape, 
        #   as the number of channels is increased to 8.
        compute_losses = functools.partial(self.diffusion.training_losses,
                                            self.model,
                                            #x_start=micro,
                                            x_start=batch,
                                            t=t,
                                            label_cond=label_cond,
                                            segmentation=segmentation,
                                            use_wavelet=self.use_wavelet,
                                            model_kwargs=None,
                                            mode=self.mode,
                                            tumour_weight=self.tumour_weight
                                            )
        losses1 = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses1["loss"].detach()
            )

        losses = losses1[0]         # Loss values
        sample = losses1[1]         # Denoised subbands at t=0
        sample_idwt = losses1[2]    # Inverse wavelet transformed denoised subbands at t=0
        label_cond = losses1[3]     # Condition used for the generation process

        if 'mse_wav' in losses:
            # Log wavelet level loss
            self.summary_writer.add_scalar('loss/mse_wav_lll', losses["mse_wav"][0].item(),
                                            global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_llh', losses["mse_wav"][1].item(),
                                            global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_lhl', losses["mse_wav"][2].item(),
                                            global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_lhh', losses["mse_wav"][3].item(),
                                            global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hll', losses["mse_wav"][4].item(),
                                            global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hlh', losses["mse_wav"][5].item(),
                                            global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hhl', losses["mse_wav"][6].item(),
                                            global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hhh', losses["mse_wav"][7].item(),
                                            global_step=self.step + self.resume_step)

            weights = th.ones(len(losses["mse_wav"])).cuda()  # Equally weight all wavelet channel losses
            if self.tumour_weight!=0:
                # Write the tumour loss 
                self.summary_writer.add_scalar('loss/mse_tumour', losses["mse_tumour"].item(),
                                                    global_step=self.step + self.resume_step)
                loss = (losses["mse_wav"] * weights).mean() + (losses["mse_tumour"] * self.tumour_weight).mean()
            else:
                loss = (losses["mse_wav"] * weights).mean()
        else:
            # Log mse loss
            self.summary_writer.add_scalar('loss/mse', losses["mse"].item(),
                                            global_step=self.step + self.resume_step)
            if self.tumour_weight!=0:
                # Write the tumour loss 
                self.summary_writer.add_scalar('loss/mse_tumour', losses["mse_tumour"].item(),
                                                    global_step=self.step + self.resume_step)
                loss = (losses["mse"] * weights).mean() + (losses["mse_tumour"] * self.tumour_weight).mean()
            else:
                loss = (losses["mse"] * weights).mean()
            
        lossmse = loss.detach()

        log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

        # perform some finiteness checks
        if not torch.isfinite(loss):
            logger.log(f"Encountered non-finite loss {loss}")
        if self.use_fp16:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        return lossmse.detach(), sample, sample_idwt, label_cond

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, state_dict):
            if dist.get_rank() == 0:
                logger.log("Saving model...")
                if self.dataset == 'brats':
                    filename = f"brats_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'lidc-idri':
                    filename = f"lidc-idri_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'hnn':
                    filename = f"hnn_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'c_brats':
                    filename = f"c_brats_{(self.step+self.resume_step):06d}.pt"
                elif self.dataset == 'hnn_tumour_inpainting':
                    filename = f"hnn_tumour_inpainting_{(self.step+self.resume_step):06d}.pt"
                else:
                    raise ValueError(f'dataset {self.dataset} not implemented')

                with bf.BlobFile(bf.join(get_blob_logdir(), 'checkpoints', filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.model.state_dict())

        if dist.get_rank() == 0:
            checkpoint_dir = os.path.join(logger.get_dir(), 'checkpoints')
            with bf.BlobFile(
                bf.join(checkpoint_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """

    split = os.path.basename(filename)
    split = split.split(".")[-2]  # remove extension
    split = split.split("_")[-1]  # remove possible underscores, keep only last word
    # extract trailing number
    reversed_split = []
    for c in reversed(split):
        if not c.isdigit():
            break
        reversed_split.append(c)
    split = ''.join(reversed(reversed_split))
    split = ''.join(c for c in split if c.isdigit())  # remove non-digits
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
