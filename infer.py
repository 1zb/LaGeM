from tqdm import tqdm
from pathlib import Path
import util.misc as misc
from util.shapenet import ShapeNet, category_ids
from util.objaverse import Objaverse
import models_ae
import mcubes
import trimesh
from scipy.spatial import cKDTree as KDTree
import numpy as np
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
import yaml
import math

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='AutoEncoder', type=str,
                    metavar='MODEL', help='Name of model to train')
parser.add_argument(
    '--pth', default='output/ae/checkpoint-350.pth', type=str)
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--pc_path', type=str, required=True)
args = parser.parse_args()


# import utils


def main():
    print(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = models_ae.__dict__[args.model]()
    device = torch.device(args.device)

    model.eval()
    model.load_state_dict(torch.load(args.pth, map_location='cpu')[
                          'model'], strict=True)
    model.to(device)
    # print(model)

    density = 256
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(
        np.float32)).view(3, -1).transpose(0, 1)[None].cuda()

    surface = trimesh.load(args.data_path).vertices
    assert surface.shape[0] == 8192

    ## normalize
    shifts = (surface.max(axis=0) + surface.min(axis=0)) / 2
    surface = surface - shifts
    distances = np.linalg.norm(surface, axis=1)
    scale = 1 / np.max(distances)
    surface *= scale
    
    surface = torch.from_numpy(surface.astype(np.float32)).to(device)
    
    with torch.no_grad():
        outputs = model(surface[None], grid)['logits'][0]

        volume = outputs.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy() * (-1)

        verts, faces = mcubes.marching_cubes(volume, 0)
        verts *= gap
        verts -= 1.
        m = trimesh.Trimesh(verts, faces)

        m.export('output.ply')
