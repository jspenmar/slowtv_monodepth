from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from networks import DepthNet


def get_device(name: Optional[str] = None) -> torch.device:
    """Get torch device to run predictions on."""
    if name: return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(ckpt_file: Path) -> nn.Module:
    """Load pretrained model from checkpoint file."""
    print(f'-> Loading model from "{ckpt_file}"...')

    ckpt = torch.load(ckpt_file, map_location='cpu')
    cfg = ckpt['hyper_parameters']['cfg']['net']['depth']
    state_dict = {k.replace('nets.depth.', ''): v for k, v in ckpt['state_dict'].items() if 'nets.depth' in k}

    net = DepthNet(**cfg).eval()
    net.load_state_dict(state_dict)
    for p in net.parameters(): p.requires_grad = False

    return net


def get_files(path: Path, ext: str) -> list[Path]:
    """Get all files in a directory with a given extension."""
    files = sorted(path.glob(f'*{ext}'))
    if not files: raise FileNotFoundError(f'No files found in "{path}" with extension "{ext}".')

    print(f'-> Found {len(files)} files to predict...')
    return files


def load_img(img_file: Path, width: int, height: int) -> tuple[Tensor, tuple[int, int]]:
    """Load and preprocess the input to the network."""
    img = Image.open(img_file).convert('RGB')
    img = np.array(img, dtype=np.float32) / 255.
    img = torch.as_tensor(img).permute(2, 0, 1)[None]

    ref_shape = img.shape[-2:]
    shape = get_img_shape(img, width, height)
    img = F.interpolate(img, size=shape, mode='bilinear', align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = (img - mean) / std
    return img, ref_shape


def get_img_shape(img: Tensor, w: int, h: int) -> tuple[int, int]:
    """Return the target image shape for resizing.
    If image is landscape, resize to fixed width and scale height to fit aspect ratio.
    If image is portrait, resize to fixed height and scale width to fit aspect ratio.
    The final width and height are rounded to the closest multiple of 32.
    """
    img_h, img_w = img.shape[2:]
    new_h, new_w = (int(w*img_h/img_w), w) if img_w >= img_h else (h, int(h*img_w/img_h))
    new_h, new_w = round(new_h/32)*32, round(new_w/32)*32
    return new_h, new_w


def save_disp(disp: Tensor, img_file: Path, out_dir: Path, out_ext: list[str]) -> None:
    """Save disparity predictions as either an RGB visualization or a numpy array."""
    name = img_file.stem
    disp = disp.cpu().numpy().squeeze()

    for ext in out_ext:
        if ext == '.png':
            disp = (colorize_disp(disp) * 255).astype(np.uint8)
            Image.fromarray(disp).save(out_dir/f'{name}{ext}')
        elif ext == '.npy':
            np.save(out_dir/f'{name}{ext}', disp)

        else:
            raise ValueError(f'Invalid extension "{ext}".')


def colorize_disp(disp: np.ndarray) -> np.ndarray:
    """Convert disparity predictions to RGB visualization."""
    vmin, vmax = 0, np.percentile(disp, 95)
    disp = (disp.clip(vmin, vmax) - vmin) / (vmax - vmin + 1e-5)  # Normalize [0, 1]
    disp = plt.get_cmap('turbo')(disp)[..., :-1]  # Remove alpha
    return disp


def main(args):
    device = get_device(args.device)
    net = load_model(args.ckpt_file).to(device)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f'-> Saving predictions to "{args.out_dir}"...')

    img_files = get_files(args.img_dir, args.img_ext)
    for file in tqdm(img_files):
        img, ref_shape = load_img(file, args.width, args.height)
        disp = net(img.to(device))['disp'][0]
        disp = F.interpolate(disp, size=ref_shape, mode='bilinear', align_corners=False)
        if args.out_dir: save_disp(disp, file, args.out_dir, args.out_ext)


if __name__ == '__main__':
    parser = ArgumentParser('Sample script to run KBR predictions on a directory of images.')
    parser.add_argument('--ckpt-file', type=Path, required=True, help='Pretrained checkpoint to load.')
    parser.add_argument('--img-dir', type=Path, required=False, help='Path to directory containing images.')
    parser.add_argument('--img-ext', default='.png', help='Image extension to search for.')
    parser.add_argument('--out-dir', type=Path, default=None, help='Path to directory to save predictions.')
    parser.add_argument('--out-ext', nargs='+', default=['.png', '.npy'], help='Extensions to save predictions as (png for viz, npy for disp map).')
    parser.add_argument('--width', default=640, help='Width required when resizing landscape images.')
    parser.add_argument('--height', default=384, help='Height required when resizing portrait images.')
    parser.add_argument('--device', default=None, help='Device to run predictions on.')

    main(parser.parse_args())
