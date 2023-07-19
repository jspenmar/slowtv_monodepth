"""Adapted from MiDaS (https://github.com/isl-org/MiDaS/tree/master/midas/backbones)"""
import torch
import torch.nn as nn

__all__ = [
    'activations', 'get_activation', 'get_readout_oper',
    'Slice', 'AddReadout', 'ProjectReadout', 'Transpose',
    'make_backbone_default', 'forward_default', 'forward_adapted_unflatten',
]


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == 'ignore': return [Slice(start_index)] * len(features)
    elif use_readout == 'add': return [AddReadout(start_index)] * len(features)
    elif use_readout == 'project': return [ProjectReadout(vit_features, start_index) for _ in features]
    else: raise ValueError(f"Invalid readout operation. ({use_readout} vs. {('ignore', 'add', 'project')})")


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super().__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super().__init__()
        self.start_index = start_index

    def forward(self, x):
        readout = ((x[:, 0]+x[:, 1])/2) if self.start_index == 2 else x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super().__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2*in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)
        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


def forward_default(net, x, function_name='forward_features'):
    getattr(net.model, function_name)(x)

    layer_1 = net.activations['1']
    layer_2 = net.activations['2']
    layer_3 = net.activations['3']
    layer_4 = net.activations['4']

    if hasattr(net, 'act_postprocess1'): layer_1 = net.act_postprocess1(layer_1)
    if hasattr(net, 'act_postprocess2'): layer_2 = net.act_postprocess2(layer_2)
    if hasattr(net, 'act_postprocess3'): layer_3 = net.act_postprocess3(layer_3)
    if hasattr(net, 'act_postprocess4'): layer_4 = net.act_postprocess4(layer_4)

    return layer_1, layer_2, layer_3, layer_4


def forward_adapted_unflatten(net, x, function_name='forward_features'):
    b, c, h, w = x.shape
    getattr(net.model, function_name)(x)

    layer_1 = net.activations['1']
    layer_2 = net.activations['2']
    layer_3 = net.activations['3']
    layer_4 = net.activations['4']

    layer_1 = net.act_postprocess1[:2](layer_1)
    layer_2 = net.act_postprocess2[:2](layer_2)
    layer_3 = net.act_postprocess3[:2](layer_3)
    layer_4 = net.act_postprocess4[:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(2, torch.Size([h//net.model.patch_size[1], w//net.model.patch_size[0]]))
    )

    if layer_1.ndim == 3: layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3: layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3: layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3: layer_4 = unflatten(layer_4)

    layer_1 = net.act_postprocess1[3:](layer_1)
    layer_2 = net.act_postprocess2[3:](layer_2)
    layer_3 = net.act_postprocess3[3:](layer_3)
    layer_4 = net.act_postprocess4[3:](layer_4)
    return layer_1, layer_2, layer_3, layer_4


def make_backbone_default(
        model,
        features=(96, 192, 384, 768),
        size=(384, 384),
        hooks=(2, 5, 8, 11),
        vit_features=768,
        use_readout='ignore',
        start_index=1,
        start_index_readout=1,
):
    net = nn.Module()
    net.n_ch = features

    net.model = model
    net.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    net.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    net.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    net.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))
    net.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index_readout)

    # 32, 48, 136, 384
    net.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0]//16, size[1]//16])),
        nn.Conv2d(
            in_channels=vit_features, out_channels=features[0],
            kernel_size=1, stride=1, padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0], out_channels=features[0],
            kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1,
        ),
    )

    net.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0]//16, size[1]//16])),
        nn.Conv2d(
            in_channels=vit_features, out_channels=features[1],
            kernel_size=1, stride=1, padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1], out_channels=features[1],
            kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1,
        ),
    )

    net.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0]//16, size[1]//16])),
        nn.Conv2d(
            in_channels=vit_features, out_channels=features[2],
            kernel_size=1, stride=1, padding=0,
        ),
    )

    net.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0]//16, size[1]//16])),
        nn.Conv2d(
            in_channels=vit_features, out_channels=features[3],
            kernel_size=1, stride=1, padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3], out_channels=features[3],
            kernel_size=3, stride=2, padding=1,
        ),
    )

    net.model.start_index = start_index
    net.model.patch_size = [16, 16]
    return net
