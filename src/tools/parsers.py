"""Tools to instantiate training components from config dicts."""
from collections import OrderedDict

import torch
from timm.optim.optim_factory import create_optimizer_v2
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import src.registry as reg
import src.typing as ty
from src import LOGGER
from src.utils import ConcatDataLoader, metrics

__all__ = ['get_net', 'get_loss', 'get_ds', 'get_dl', 'get_opt', 'get_sched', 'get_metrics']


T = ty.TypeVar('T')


def get_cls(cls_dict: dict[str, ty.Type[T]], /, *args, type: str, **kwargs) -> T:
    """Instantiate an arbitrary class from a collection.

    Including `type` makes it a keyword-only argument. This has the double benefit of forcing the user to pass it as a
    keyword argument, as well as popping it from the cfg kwargs.

    :param cls_dict: (dict[str, cls]) Dict containing mappings to the classes to choose from.
    :param args: (tuple) Args to forward to target class.
    :param type: (str) Key of the target class. Keyword-only argument.
    :param kwargs: (dict) Kwargs to forward to target class.
    :return: Target class instance.
    """
    try: return cls_dict[type](*args, **kwargs)
    except Exception as e: raise ValueError(f'Error using "{type}" in {list(cls_dict)}') from e


def get_net(cfg: dict) -> nn.ModuleDict:
    """Instantiate the target networks from a cfg dict.

    Depth estimation typically consists of multiple networks, commonly `depth` and `pose`.
    We assume that, within a given category, we can use different classes interchangeably.
    For instance, all `depth` networks take a single image as input and produce a multi-scale output, while all
    `pose` networks take multiple images and produce relative poses for each pair.

    Networks can be omitted by setting their cfg to `None`. Useful when overriding the default cfg.
    See `cfg/defaults.yaml` for a full example.

    Example:
    ```
    cfg = {
        'depth': {
            'enc_name': 'convnext_base',
            'pretrained': True,
            'dec_name': 'monodepth',
            'out_scales': [0, 1, 2, 3],
        },
        'pose': {
            'enc_name': 'resnet18',
            'pretrained': True,
        },
    }
    ```

    :param cfg: (NetCfg) Dict of dicts, containing the network `type` and kwargs for each network.
    :return: (nn.ModuleDict) Dict of instantiated networks.
    """
    reg.trigger_nets()
    reg.trigger_decoders()
    nets = {k: get_cls(reg.NET_REG, type=k, **kw) for k, kw in cfg.items() if kw is not None}
    return nn.ModuleDict(OrderedDict(nets))


def get_loss(cfg: dict) -> tuple[nn.ModuleDict, nn.ParameterDict]:
    """Instantiate the target losses from a cfg dict.

    In addition to the kwargs required to instantiate the loss, we also expect a `weight` kwarg, used to
    balance the various losses when computing the final loss. (Default: 1)

    Losses can be omitted by setting their cfg to `None`. Useful when overriding the default cfg.
    See `cfg/defaults.yaml` for a full example.

    Example:
    ```
    cfg = {
        'img_recon': {
            'weight': 1,
            'loss_name': 'ssim',
            'use_min': True,
        }

        'disp_smooth': {
            'weight': 0.001,
            'use_edges': True,
        }
    ```

    :param cfg: (LossDict) Dict of dicts, containing the loss `type`, `weight` and kwargs for each loss.
    :return: (nn.ModuleDict) Dict of instantiated losses.
    """
    reg.trigger_losses()
    losses, weights = nn.ModuleDict(), nn.ParameterDict()
    for k, kw in cfg.items():
        if kw is None: continue
        weights[k] = nn.Parameter(torch.as_tensor(kw.pop('weight', 1)), requires_grad=False)
        losses[k] = reg.LOSS_REG[k](**kw)

    return losses, weights


def get_ds(cfg: dict, mode: ty.N[str] = None) -> dict[str, Dataset]:
    """Instantiate the target datasets from a cfg dict.

    Datasets consist of a default cfg for each class, which can be overriden based on a `mode` sub-dict.

    Datasets can be omitted by setting their cfg to `None`. Useful when overriding the default cfg.
    See `cfg/defaults.yaml` for a full example.

    Example:
    ```
    cfg = {
        'kitti_lmdb': {
            'split': 'eigen_zhou',
            'shape': (192, 640),
            'supp_idxs': [-1, 1, 0],

            'train': {'mode': 'train', 'use_aug': True},
            'val': {'mode': 'val', 'use_aug': False},
        }

        'slow_tv_lmdb': {
            'split': 'all',
            'shape': (384, 640),
            'supp_idxs': [-1, 1],

            'train': {'mode': 'train', 'use_aug': True},
            'val': {'mode': 'val', 'use_aug': False},
        }
    ```

    :param cfg: (DataCfg) Dict of dicts, containing the dataset `type` and kwargs for each dataset.
    :param mode: (str) Mode to use for the dataset. If `None`, use the default cfg.
    :return: (dict[str, Dataset]) Dict of instantiated datasets.
    """
    reg.trigger_datas()
    ds = {}
    for t, kw in cfg.items():
        if kw is None: continue
        assert isinstance(kw, dict), f"Expected dict of dicts. Got '{kw}'."
        c = {k: v for k, v in kw.items() if k not in {'train', 'val', 'test'}}
        if mode: c.update(kw.get(mode, {}))
        ds[t] = get_cls(reg.DATA_REG, type=t, **c)
    return ds


def get_dl(mode: str, cfg_ds: dict, cfg_dl: dict) -> DataLoader:
    """Instantiate the target dataloader from a cfg dict.

    Dataloaders consist of a default cfg, which can be overriden based on a `mode` sub-dict.
    The datasets are expected to be a subclass of `BaseDataset`, which provides a `collate_fn` method.
    By default, we use `pin_memory=True`.

    If training with multiple datasets, we use the custom `ConcatDataset` class, which concatenates all datasets
    such that each batch contains samples from only one dataset. This is due to each dataset potentially having
    different images shapes.

    See `cfg/defaults.yaml` for a full example.

    Example:
    ```
    cfg = {
        'batch_size': 4,
        'num_workers': 4,
        'drop_last': True,

        'train': { 'shuffle': True },
        'val': { 'shuffle': False },
    }
    ```

    :param mode: (str) Mode to use for the dataloader. If `None`, use the default cfg.
    :param cfg_ds: (DataCfg) Dict of dicts, containing the dataset `type` and kwargs for each dataset.
    :param cfg_dl: (LoaderCfg) Dict of dicts, containing the dataloader kwargs.
    :return: (DataLoader) Instantiated dataloader.
    """
    ds = get_ds(cfg_ds, mode)
    ds = list(ds.values())

    cfg = {k: v for k, v in cfg_dl.items() if k not in {'train', 'val', 'test'}} | cfg_dl.get(mode, {})
    cfg['pin_memory'] = cfg.get('pin_memory', True)
    cfg['collate_fn'] = ds[0].collate_fn

    use_ddp = cfg.pop('use_ddp', False)
    seed = cfg.pop('seed', 42)

    if use_ddp:
        shuffle, drop_last = cfg.pop('shuffle', False), cfg.pop('drop_last', False)
        seeds = [seed*10**i for i, _ in enumerate(ds)]
        samplers = [DistributedSampler(d, shuffle=shuffle, drop_last=drop_last, seed=s) for d, s in zip(ds, seeds)]
    else:
        samplers = [None for _ in ds]

    dl = [DataLoader(d, sampler=s, **cfg) for d, s in zip(ds, samplers)]
    return dl[0] if len(dl) == 1 else ConcatDataLoader(dl)


def get_opt(parameters: ty.U[ty.Iterable, nn.Module], cfg: dict) -> optim.Optimizer:
    """Instantiate the target optimizer from a cfg dict. Wrapper for `timm` `create_optimizer_v2`.

    Example:
    ```
    cfg = {
        'type': 'adamw',
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'frozen_bn': True,
    }
    ```

    :param parameters: (Iterable|nn.Module) Parameters to forward to the optimizer (in any `torch` format).
    :param cfg: (OptCfg) Target optimizer `type` and kwargs to forward to it.
    :return: (Optimizer) Instantiated optimizer.
    """
    if 'type' in cfg: cfg['opt'] = cfg.pop('type')
    elif 'opt' not in cfg: raise KeyError("Must provide a cfg key `type` or `opt` when instantiating an optimizer.")

    if cfg.pop('frozen_bn', False):
        if not isinstance(parameters, nn.Module):
            raise ValueError("Cannot freeze batch norm parameters unless given nn.Module")

        for m in parameters.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(False)

    if blr := cfg.pop('backbone_lr', False):
        if not isinstance(parameters, nn.Module): raise ValueError("Cannot set backbone LR unless given nn.Module")
        if blr == cfg['lr']: raise ValueError("Backbone LR must be different from the main LR")

        LOGGER.info(f"Setting backbone LR to {blr} with base LR {cfg['lr']}...")
        parameters = [
            {'params': (p for n, p in parameters.named_parameters() if 'encoder' not in n)},
            {'params': (p for n, p in parameters.named_parameters() if 'encoder' in n), 'lr': blr},
        ]

    return create_optimizer_v2(parameters, **cfg)


def get_sched(opt: optim.Optimizer, cfg: dict[str, dict]) -> dict[str, ty._LRScheduler]:
    """Instantiate the target schedulers from a cfg dict. Wrapper for `timm` `create_scheduler_v2`.

    Example:
    ```
    cfg = {
        'steplr': {
            'step_size': 10,
            'gamma': 0.1,
        },

        'linear': {
        'start_factor: 0.1,
        'total_iters': 4,
        },
    }
    ```

    :param opt: (Optimizer) Optimizer schedule for.
    :param cfg: (SchedCfg) Dict of dicts, containing the scheduler `type` and kwargs for each scheduler.
    :return: (dict[str, _LRScheduler]) Dict of instantiated schedulers.
    """
    sch = {k: get_cls(reg.SCHED_REG, opt, type=k, **kw) for k, kw in cfg.items() if kw is not None}
    return sch


def get_metrics() -> nn.ModuleDict:
    """Instantiate the collection of depth metrics to monitor."""
    return nn.ModuleDict({
        'MAE': metrics.MAE(),
        'RMSE': metrics.RMSE(),
        'LogSI': metrics.ScaleInvariant(mode='log'),
        'AbsRel': metrics.AbsRel(),
        'Acc': metrics.DeltaAcc(delta=1.25),
    })
