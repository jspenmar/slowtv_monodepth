from functools import partial

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ChainedScheduler
from torch.utils.data import DataLoader

import src.typing as ty
from src import LOGGER
from src.tools import T_from_AAt, ViewSynth, ops, parsers, resize_K, to_inv, to_scaled
from src.utils import MultiLevelTimer, flatten_dict
from . import aspect_ratio_aug, handlers as h

__all__ = ['MonoDepthModule']


class MonoDepthModule(pl.LightningModule):
    """A trainer class for monocular depth estimation.

    Self-supervised monocular depth estimation usually consists of the following:
        - Depth estimation network: Produces a multi-scale sigmoid disparity in the range [0, 1]
        - Pose estimation network: Produces the relative transform between two input images (i.e. support frames)
        - Reconstruction loss: Given depth and pose we can reconstruct the target view from a support frame.

    :param cfg: (MonoDepthConfig) Trainer configuration (see `src.typing.MonoDepthConfig` or example cfg)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        torch.set_float32_matmul_precision(self.cfg['trainer'].get('matmul', 'high'))

        self.batch_size = self.cfg['loader']['batch_size']
        self.lr = self.cfg['optimizer']['lr']
        self.save_hyperparameters()

        self.nets = parsers.get_net(self.cfg['net'])
        self.losses, self.weights = parsers.get_loss(self.cfg['loss'])
        self.metrics = parsers.get_metrics()
        self.synth = None

        self.current_batch: dict[str, ty.S] = {'train': [], 'val': []}  # For image logging.

        self.scales = self.nets['depth'].out_scales
        self.n_scales = len(self.scales)

        self.min_depth = cfg['trainer'].get('min_depth', None)
        self.max_depth = cfg['trainer'].get('max_depth', None)
        self.should_scale = self.min_depth or self.max_depth
        self.to_depth = (lambda x: to_scaled(x, self.min_depth, self.max_depth)[1]) if self.should_scale else to_inv

        self.always_fwd_pose = cfg['trainer'].get('always_fwd_pose', True)
        self.auto_scale_lr = cfg['trainer'].get('auto_scale_lr', False)

        self.ar_aug = partial(
            aspect_ratio_aug,
            p=cfg['trainer'].get('aspect_ratio_aug_prob', 0.0),
            crop_min=cfg['trainer'].get('aspect_ratio_min', 0.5),
            crop_max=cfg['trainer'].get('aspect_ratio_max', 1),
            ref_shape=cfg['trainer'].get('aspect_ratio_ref_shape', None)
        )

        if mode := self.cfg['trainer'].get('compile', False):
            # FIXME: Not tested thoroughly. Models seem to not optimize sometimes.
            LOGGER.info('Compiling models and losses...')
            if mode is True: mode = 'default'
            for k in self.nets: self.nets[k].encoder = torch.compile(self.nets[k].encoder, mode=mode)
            for k in self.losses: self.losses[k] = torch.compile(self.losses[k], mode=mode)

        self.timer = MultiLevelTimer(name=self.__class__.__qualname__, as_ms=True, precision=4, sync_gpu=True)

    def train_dataloader(self) -> DataLoader:
        """Return the dataloader for the training dataset."""
        self.cfg['loader']['batch_size'] = self.batch_size
        dl = parsers.get_dl('train', self.cfg['dataset'], self.cfg['loader'])
        LOGGER.info(f'-> Train dataloader: {len(dl)}')
        return dl

    def val_dataloader(self) -> DataLoader:
        """Return the dataloader for the validation dataset."""
        self.cfg['loader']['batch_size'] = self.batch_size
        dl = parsers.get_dl('val', self.cfg['dataset'], self.cfg['loader'])
        LOGGER.info(f'-> Val dataloader: {len(dl)}')
        return dl

    def configure_optimizers(self):
        """Create optimizer & scheduler."""
        self.cfg['optimizer']['lr'] = self.lr
        out = {'optimizer': parsers.get_opt(self.nets, self.cfg['optimizer'])}

        if cfg := self.cfg.get('scheduler'):
            sch = parsers.get_sched(out['optimizer'], cfg)
            out['lr_scheduler'] = ChainedScheduler(list(sch.values()))

        return out

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer) -> None:
        """Speed up zero grad by setting parameters to `None`."""
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: ty.T, *args, **kwargs) -> None:
        """Time backwards pass."""
        with self.timer('Backward'): super().backward(loss, *args, **kwargs)

    def training_step(self, batch: ty.BatchData, batch_idx: int) -> ty.T:
        """Run forward training step & cache batch."""
        with self.timer('Aug'): batch = self.ar_aug(batch)
        self.current_batch['train'] = batch
        return self.step(batch, mode='train')[0]

    def validation_step(self, batch: ty.BatchData, batch_idx: int) -> ty.T:
        """Run forward validation step & cache batch."""
        self.current_batch['val'] = batch
        return self.step(batch, mode='val')[0]

    def step(self, batch: ty.BatchData, mode: str = 'train') -> tuple[ty.T, ty.TensorDict, ty.TensorDict]:
        """Run a single training step.

        - Compute network forward pass
        - Post-process network outputs
        - Compute loss forward pass
        - Compute depth metrics
        - Log scalars

        :param batch: A single training batch consisting of (
            x: (TensorDict) {
                images: (Tensor) (b, c, h, w) Augmented target images to predict depth.
                supp_imgs: (Tensor) (n, b, c, h, w) Augmented support frames to compute relative pose.
                supp_idxs: (Tensor) (n,) Index of each support frame w.r.t. the target frame.

                (Optional)
                (If using stereo support frame)
                idx_stereo: (Tensor) (n,) Index of the stereo pair within `supp_idxs`. Added by `self.forward_postprocces`.
            }

            y: (TensorDict) {
                images: (Tensor) (b, c, h, w) Non-augmented (or standardized) target images.
                supp_imgs: (Tensor) (n, b, c, h, w) Non-augmented (or standardized) support frames.
                K: (Tensor) (b, 4, 4) Camera intrinsic parameters.

                (Optional)
                (If using stereo support frame)
                T_stereo: (Tensor) (b, 4, 4) Transform to the stereo pair.

                (If using depth validation)
                depth: (Tensor)(b, 1, h, w) Ground-truth LiDAR depth.

                (If using proxy depth supervision)
                depth_hints: (Tensor) (b, 1, h, w) Proxy stereo depth map.
            }

            m: (dict) {
                items: (str) Loaded dataset item.
                aug (list[str]): Augmentations applied to current item.
                errors: (list[str]): List of errors when loading previous items.
                data_timer (list[MultiLevelTimer]): Timing information for each item in the batch.
            }
        )

        :param mode: (str) Training phase {core, val}.
        :return: (
            loss: (Tensor) Total loss for optimization.
            loss_dict: (TensorDict) Intermediate outputs produced by the loss. (See `self.forward_loss`)
            fwd: (TensorDict) Network forward pass. (See `self.forward` & `self.forward_postprocess`)
        )
        """
        try:
            x, y, m = batch
            self.synth = ViewSynth(x['imgs'].shape[-2:]).to(x['imgs'].device)

            with self.timer('Total'):
                with self.timer('Forward'): fwd = self.forward(x)
                with self.timer('Post-Process'): fwd = self.forward_postprocess(fwd, x, y)
                with self.timer('Loss'): loss, loss_dict = self.forward_loss(fwd, x, y)
                with self.timer('Metrics'): metrics = self.compute_metrics(fwd['depth_up'][0].detach(), y['depth']) \
                    if 'depth' in y else {}

            self.log_dict(flatten_dict({
                f'{mode}_losses/loss': loss,
                f'{mode}_losses': {k: v for k, v in loss_dict.items() if 'loss_' in k},
                **({f'{mode}_timer/Data': m['timer_data'][0].mean_elapsed(m['timer_data'])} if 'timer_data' in m else {}),
                f'{mode}_timer/Module': self.timer.to_dict(),
                f'{mode}_metrics': metrics,
                f'{mode}_monitor_depth': self.summarize_depth(fwd),
                f'{mode}_monitor_pose': self.summarize_pose(fwd),
                f'{mode}_monitor_K': self.summarize_K(fwd),
            }), rank_zero_only=True)
        finally:
            self.timer.reset()

        return loss, loss_dict, fwd

    def forward(self, x: ty.TensorDict) -> ty.TensorDict:
        """Run networks forward pass.

        NOTE: The `virtual stereo` prediction has two channels, not one.
        This is because we assume that the target frame is the central image in a trinocular setting.
        The first channel corresponds to the virtual left prediction, while the second channel is the right prediction.

        :param x: (TensorDict) Batch inputs required for network forward pass. (See `self._step`)
        :return: fwd: (TensorDict) {
            depth_feats: (list[ty.T]) List of intermediate depth encoder features.
            disp: (TensorDict) {s: (b, 1, h/2**s, w/2**s)} Predicted sigmoid disparity at each scale.

            (Optional)
            (If using `learn_K`)
            K: (Tensor) (b, 4, 4) Predicted camera intrinsics for each image.
            fs: (Tensor) (b, 2) Predicted normalized focal lengths (x, y).
            cs: (Tensor) (b, 2) Predicted normalized principal points (x, y).

            (Optional)
            (If using `mask` prediction)
            mask: (TensorDict) {s: (b, n, h/2**s, w/2**s)} Predicted photometric mask at each scale.

            (If using `virtual stereo` prediction)
            disp_stereo: (TensorDict) {s: (b, 2, h/2**s, w/2**s)} Predicted disparity at each scale for the STEREO pair.

            (If using `mask` & `virtual stereo` prediction)
            mask_stereo: (TensorDict) {s: (b, n, h/2**s, w/2**s)} Predicted photometric mask for the stereo pair.

            (If using `pose` prediction network)
            T_{idx}: (b, 4, 4) Transform from each target frame to each support frame (excluding stereo)

            (If using `autoencoder` network)
            autoenc_imgs: (TensorDict) {s: (b, 3, h/2**s, w/2**s)} Autoencoder target image predictions.
            supp_autoenc_imgs: (TensorDict) {s: (n, b, 3, h/2**s, w/2**s)} Autoencoder support image predictions.
            autoenc_feats: (list[ty.T]) List of intermediate autoencoder target features.
            supp_autoenc_feats: (list[ty.T]) List of intermediate autoencoder support features.
        }
        """
        fwd = {}
        for key, net in self.nets.items():
            # DEPTH ESTIMATION
            # Multi-scale depth prediction: `disp_{0-3}`.
            if key == 'depth':
                fwd |= net(x['imgs'])

            # POSE ESTIMATION
            # Relative poses wrt each support frame `T_{idx}`.
            # We always predict a forward pose, so image order is reversed and the pose inverted when `idx < 0`.
            # NOTE: Stereo pose is added during loss computation if required.
            elif key == 'pose':
                should_inv = lambda i: self.always_fwd_pose and i < 0
                imgs = torch.stack([
                    torch.cat([supp, x['imgs']] if should_inv(i) else [x['imgs'], supp], dim=1)
                    for i, supp in zip(x['supp_idxs'], x['supp_imgs']) if i != 0
                ])  # (n, b, 3*2, h, w)
                sh = imgs.shape[:2]

                fwd_pose = net(imgs.flatten(0, 1))  # R: (n*b, 1, 3) - T: (n*b, 1, 3)
                Ts = T_from_AAt(aa=fwd_pose['R'][:, 0], t=fwd_pose['t'][:, 0]).unflatten(0, sh)  # (n, b, 4, 4)

                idxs = [i for i in x['supp_idxs'] if i != 0]
                fwd |= {f'T_{i}': T.inverse() if should_inv(i) else T for i, T in zip(idxs, Ts)}

                if 'fs' in fwd_pose and 'fs' not in fwd:
                    fwd['fs'] = fwd_pose['fs'].unflatten(0, sh)  # For consistency losses
                    fwd['cs'] = fwd_pose['cs'].unflatten(0, sh)

                    # Use only the Ks predicted for the first support image.
                    K = net.build_K(fwd_pose['fs'], fwd_pose['cs']).unflatten(0, sh)[0]
                    fwd['K'] = resize_K(K, x['imgs'].shape[-2:])
                    del K

                del imgs, Ts, idxs, fwd_pose

            # AUTOENCODER IMAGE RECONSTRUCTION
            elif key == 'autoencoder':
                fwd.update(net(x['imgs']))

                fwd_supp = net(x['supp_imgs'].flatten(0, 1))
                fwd_supp = ops.op(fwd_supp, fn='unflatten', dim=0, sizes=(len(x['supp_idxs']), -1))
                fwd |= {f'supp_{k}': v for k, v in fwd_supp.items()}
                del fwd_supp

            else:
                raise KeyError(f'Unrecognized key: {key}.')

        return fwd

    def forward_postprocess(self, fwd: ty.TensorDict, x: ty.TensorDict, y: ty.TensorDict) -> ty.TensorDict:
        """Run network forward postprocessing.

        - Upsample (stereo) disparity & mask predictions.
        - Convert upsampled disparity to scaled depth.
        - Index the correct cam for the virtual stereo prediction.
        - Concatenate predicted & stereo support motion.

        :param fwd: (TensorDict) Network forward pass. (See `self.forward`)
        :param x: (TensorDict) Batch inputs required for network forward pass. (See `self._step`)
        :param y: (TensorDict) Batch inputs required for loss forward pass. (See `self._step`)
        :return: fwd: (TensorDict) Updated `fwd` with {
            disp_up: (TensorDict) {s: (b, 1, h, w)} Upsampled sigmoid disparity predictions.
            depth_up: (TensorDict) {s: (b, 1, h, w)} Upsampled scaled depth predictions.
            Ts: (TensorDict) (n, b, 4, 4) Predicted and/or stereo motion w.r.t. the target frame.

            (Optional)
            (If using `mask` prediction)
            mask_up: (TensorDict) {s: (b, 1, h, w)} Upsampled photometric mask predictions.

            (If using `virtual stereo` prediction)
            idx_stereo: (int) Index of the support frame corresponding to the support frame.
            disp_stereo: (TensorDict) {s: (b, 1, h/2**s, w/2**s)} Sigmoid stereo disparity predictions.
            disp_stereo_up: (TensorDict) {s: (b, 1, h, w)} Upsampled sigmoid stereo disparity predictions.
            depth_stereo_up: (TensorDict) {s: (b, 1, h, w)} Upsampled stereo scaled depth predictions.

            (If using `mask` and `virtual stereo` prediction)
            mask_stereo_up: (TensorDict) {s: (b, 1, h, w)} Upsampled photometric mask predictions for the STEREO pair.

            (If using `autoencoder` network)
            autoenc_imgs_up: (TensorDict) {s: (b, 3, h, w)} Upsampled autoencoder target image predictions.
            supp_autoenc_imgs_up: (TensorDict) {s: (n, b, 3, h, w)} Upsampled autoencoder support image predictions.
        }
        """
        # UPSAMPLE & CONVERT TO DEPTH
        fwd_new = {}
        for k, v in fwd.items():
            k_new = f'{k}_up'

            if 'disp' in k:  # {s: (b, 1, h/2**s, w/2**s)}
                    fwd_new[k_new] = ops.op(v, fn=ops.interpolate_like, other=x['imgs'], mode='bilinear')
                    fwd_new[k_new.replace('disp', 'depth')] = ops.op(fwd_new[k_new], fn=self.to_depth)

            elif 'mask' in k:  # {s: (b, 1, h/2**s, w/2**s)}
                fwd_new[k_new] = ops.op(v, fn=ops.interpolate_like, other=x['imgs'], mode='bilinear')

            elif k == 'autoenc_imgs':  # {s: (b, 1, h/2**s, w/2**s)}
                fwd_new[k_new] = ops.op(v, fn=ops.interpolate_like, other=x['imgs'], mode='bilinear')

            elif k == 'supp_autoenc_imgs':  # {s: (n, b, 1, h/2**s, w/2**s)} (n=supp_imgs)
                v = ops.op(v, fn='flatten', start_dim=0, end_dim=1)
                fwd_new[k_new] = ops.op(v, fn=ops.interpolate_like, other=x['imgs'], mode='bilinear')
                fwd_new[k_new] = ops.op(fwd_new[k_new], fn='unflatten', dim=0, sizes=(len(x['supp_idxs']), -1))

        fwd.update(fwd_new)

        # VIRTUAL STEREO
        if 'disp_stereo' in fwd:
            assert 'T_stereo' in y, 'Missing stereo transform.'

            x['idx_stereo'] = next(i for i in x['supp_idxs'] if i == 0)  # Index of stereo images in support
            idx = (y['T_stereo'][:, 0, 3] > 0).long()  # 0 if target=l virtual=r, 1 if target=r virtual=l

            for k in {'disp_stereo', 'disp_stereo_up', 'depth_stereo_up'}:
                fwd[k] = {s: torch.stack([d[i] for i, d in zip(idx, depth)])[:, None] for s, depth in fwd[k].items()}

        # CONCATENATE POSES
        fwd['Ts'] = torch.stack([(y['T_stereo'] if i == 0 else fwd[f'T_{i}']) for i in x['supp_idxs']])
        return fwd

    def forward_loss(self, fwd: ty.TensorDict, x: ty.TensorDict, y: ty.TensorDict) -> tuple[ty.T, ty.TensorDict]:
        """Run loss forward pass.

        :param fwd: (TensorDict) Network forward pass. (See `self.forward` & `self.forward_postprocess`)
        :param x: (TensorDict) Batch inputs required for network forward pass. (See `self._step`)
        :param y: (TensorDict) Batch inputs required for loss forward pass. (See `self._step`)
        :return: (
            loss: (Tensor) (,) Total loss for optimization.
            loss_dict: {
                supp_imgs_warp: (Tensor) (n, b, 3, h, w) Support frames warped to match the target frame.

                (Optional)
                (If using automasking in `reconstruction` loss)
                automask: (Tensor) (b, 1, h, w) Boolean mask indicating pixels NOT removed by the automasking procedure.

                (If using `feature_reconstruction` loss)
                supp_feats_warp: (Tensor) (n, b, c, h, w) The warped support features to match the target frame.

                (If using `stereo_consistency` loss)
                disps_warp: (Tensor) (b*2, c, h, w) The warped disparities (first half corresponds to the virtual stereo).

                (If using proxy depth `regression` loss)
                automask_hints: (Tensor) (b, 1, h, w)Boolean mask indicating pixels NOT removed by invalid depths & automasking.

                (If using `smoothness` regularization)
                disp_grad: (Tensor) (b, 1, h, w) Disparity spatial gradients.
                image_grad: (Tensor) (b, 1, h, w) Image spatial gradients.
            }
        )
        """
        if 'idx_stereo' in x: y['imgs_stereo'] = y['supp_imgs'][x['idx_stereo']]
        loss, loss_dict = 0., {}

        for k, crit in self.losses.items():
            with self.timer(f'Loss-{k}'):
                l2, ld2 = None, None  # Stereo loss & dict

                # IMAGED-BASED RECONSTRUCTION LOSS
                if k == 'img_recon':
                    l, ld = h.image_recon(
                        crit, self.synth, depths=fwd['depth_up'], masks=fwd.get('mask_up'),
                        imgs=y['imgs'], supp_imgs=y['supp_imgs'], Ts=fwd['Ts'], Ks=fwd.get('K', y['K']),
                    )

                    if 'disp_stereo' in fwd:  # VIRTUAL STEREO
                        l2, ld2 = h.image_recon(
                            crit, self.synth, depths=fwd['depth_stereo_up'], masks=fwd.get('mask_stereo_up'),
                            imgs=y['imgs_stereo'], supp_imgs=y['imgs'][None],
                            Ts=y['T_stereo'].inverse()[None], Ks=fwd.get('K', y['K']),
                        )

                # FEATURE-BASED RECONSTRUCTION LOSS
                elif k == 'feat_recon':
                    feat, supp_feat = self.extract_features(fwd, x, y)
                    l, ld = h.feat_recon(
                        crit, self.synth, depths=fwd['depth_up'], masks=fwd.get('mask_up'),
                        feats=feat, supp_feats=supp_feat, Ts=fwd['Ts'], Ks=fwd.get('K', y['K'])
                    )

                # AUTOENCODER IMAGE RECONSTRUCTION
                elif k == 'autoenc_recon':
                    l, ld = h.autoenc_recon(
                        crit, preds=fwd['autoenc_imgs_up'], targets=y['imgs'],
                        supp_preds=fwd['supp_autoenc_imgs_up'], supp_targets=y['supp_imgs'],
                    )

                # VIRTUAL STEREO CONSISTENCY
                elif k == 'stereo_const':
                    assert 'disp_stereo' in fwd, 'Missing virtual stereo prediction "disp_stereo".'
                    assert 'T_stereo' in y, 'Missing stereo pair "T_stereo".'
                    l, ld = h.stereo_const(
                        crit, self.synth, disps=fwd['disp_up'], depths=fwd['depth_up'],
                        disps_stereo=fwd['disp_stereo_up'], depths_stereo=fwd['depth_stereo_up'],
                        T_stereo=y['T_stereo'], K=fwd.get('K', y['K']),
                    )

                # PROXY DEPTH REGRESSION
                elif k == 'depth_regr':
                    assert 'depth_hints' in y, 'Missing proxy depth prediction "depth_hints".'
                    l, ld = h.depth_regr(
                        crit, self.synth, photo=self.losses['img_recon'].compute_photo,
                        depths=fwd['depth_up'], targets=y['depth_hints'], imgs=y['imgs'], supp_imgs=y['supp_imgs'],
                        Ts=fwd['Ts'], Ks=fwd.get('K', y['K']),
                    )

                # DISPARITY SMOOTHNESS REGULARIZATION
                elif k == 'disp_smooth':
                    l, ld = h.disp_smooth(crit, fwd['disp'], y['imgs'])
                    if 'disp_stereo' in fwd:
                        l2, ld2 = h.disp_smooth(crit, fwd['disp_stereo'], y['imgs_stereo'])  # VIRTUAL STEREO

                # FEATURE FIRST-ORDER NON-SMOOTHNESS
                elif k == 'feat_peaky':
                    l, ld = h.feat_smooth(crit, fwd['autoenc_feats'], y['imgs'], fwd['supp_autoenc_feats'], y['supp_imgs'])

                # FEATURE SECOND-ORDER NON-SMOOTHNESS
                elif k == 'feat_smooth':
                    l, ld = h.feat_smooth(crit, fwd['autoenc_feats'], y['imgs'], fwd['supp_autoenc_feats'], y['supp_imgs'])

                # OCCLUSION REGULARIZATION
                elif k == 'disp_occ':
                    l, ld = h.disp_occ(crit, fwd['disp'])
                    if 'disp_stereo' in fwd: l += h.disp_occ(crit, fwd['disp_stereo'])[0]  # VIRTUAL STEREO

                # PREDICTIVE MASK REGULARIZATION
                elif k == 'disp_mask':
                    assert 'mask' in fwd, 'Missing masks in predictions.'
                    l, ld = h.disp_mask(crit, fwd['mask'])
                    if 'mask_stereo' in fwd: l += h.disp_mask(crit, fwd['mask_stereo'])[0]  # VIRTUAL STEREO

                else: raise ValueError(f'Missing loss key: "{k}"')

            loss += self.weights[k] * l
            loss_dict[f'loss_{k}'] = l
            loss_dict.update(ld)

            if l2 is not None:
                assert ld2 is not None
                loss += self.weights[k] * l2
                loss_dict[f'loss_stereo_{k}'] = l2
                loss_dict.update({f'stereo_{k}': v for k, v in ld2.items()})

        return loss, loss_dict

    @torch.no_grad()
    def extract_features(self, fwd: ty.TensorDict, x: ty.TensorDict, y: ty.TensorDict) -> tuple[ty.T, ty.T]:
        if 'autoencoder' in self.nets:
            feat = fwd['autoenc_feats']
            supp_feat = fwd['supp_autoenc_feats']
        else:
            feat = fwd['depth_feats']
            supp_feat = self.nets['depth'].encoder(x['supp_imgs'].flatten(0, 1))
            supp_feat = ops.op(supp_feat, fn='unflatten', dim=0, sizes=(len(x['supp_idxs']), -1))

        return feat, supp_feat

    def summarize_depth(self, fwd):
        """Compute average & std depth/disparity for logging."""
        d = {}
        d.update({f'disp_mean_{k}': v.mean().item() for k, v in fwd['disp_up'].items()})
        d.update({f'disp_std_{k}': v.std().item() for k, v in fwd['disp_up'].items()})

        d.update({f'depth_mean_{k}': v.mean().item() for k, v in fwd['depth_up'].items()})
        d.update({f'depth_std_{k}': v.std().item() for k, v in fwd['depth_up'].items()})

        if 'stereo_disp' in fwd:
            d.update({f'disp_stereo_mean_{k}': v.mean().item() for k, v in fwd['disp_stereo_up'].items()})
            d.update({f'disp_stereo_std_{k}': v.std().item() for k, v in fwd['disp_stereo_up'].items()})

            d.update({f'depth_stereo_mean_{k}': v.mean().item() for k, v in fwd['depth_stereo_up'].items()})
            d.update({f'depth_stereo_std_{k}': v.std().item() for k, v in fwd['depth_stereo_up'].items()})

        return d

    def summarize_pose(self, fwd):
        """Compute average & std translation/rotation for logging."""
        d = {}
        for k, v in fwd.items():
            if not k.startswith('T_'): continue

            ts = v[..., :3, 3]
            ts = ts.pow(2).sum(-1).sqrt()
            d[f'{k}_t_mean'] = ts.mean().item()
            d[f'{k}_t_std'] = ts.std().item()

            Tr = v[..., :3, :3].diagonal(dim1=-2, dim2=-1).mean(dim=-1)
            d[f'{k}_R_mean'] = Tr.mean().item()
            d[f'{k}_R_std'] = Tr.std().item()

        return d

    def summarize_K(self, fwd):
        """Compute average focal lengths and principal points for logging."""
        if 'K' not in fwd: return {}
        return {
            'fx': fwd['fs'][..., 0].mean(),
            'fy': fwd['fs'][..., 1].mean(),
            'cx': fwd['cs'][..., 0].mean(),
            'cy': fwd['cs'][..., 1].mean(),
        }

    @torch.no_grad()
    def compute_metrics(self, pred: ty.T, target: ty.T) -> ty.TensorDict:
        """Compute depth metrics for a dataset batch.

        :param pred: (Tensor) (b, 1, h, w) Scaled network depth predictions.
        :param target: (Tensor) (b, 1, h, w) Ground-truth LiDAR depth.
        :return: metrics: (TensorDict) Average metrics across batch.
        """
        min, max = self.min_depth or 0.1, self.max_depth or 100
        pred = ops.interpolate_like(pred, target, mode='bilinear', align_corners=False).clamp(min, max)

        mask = (target > min) & (target < max)
        target = target.where(mask, target.new_tensor(torch.nan))
        pred = pred.where(mask, pred.new_tensor(torch.nan))

        pred, target = pred.flatten(1), target.flatten(1)
        r = target.nanmedian(dim=1, keepdim=True).values / pred.nanmedian(dim=1, keepdim=True).values
        pred *= r

        pred.clamp_(min, max), target.clamp_(min, max)
        metrics = {k: metric(pred, target) for k, metric in self.metrics.items()}
        return metrics
