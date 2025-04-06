
import os
import glob
import random

import yaml 

import torch
from torch.utils import data

import numpy as np

from PIL import Image

import h5py

import csv

class Objaverse(data.Dataset):
    def __init__(
        self, 
        split, 
        transform=None, 
        sampling=True, 
        num_samples=4096, 
        return_surface=True, 
        surface_sampling=True, 
        pc_size=2048, 
        ):
        
        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.npz_folder = '/ibex/project/c2281/objaverse'

        with open('util/{}.csv'.format(split), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            model_filenames = [(os.path.join(self.npz_folder, row[0], row[1]+'.npz'), row[2]) for row in reader]

        self.models = model_filenames#[:512]

    def __getitem__(self, idx):
        # idx = idx % len(self.models)

        # category = self.models[idx]['category']
        # model = self.models[idx]['model']
        
        npz_path = self.models[idx][0]
        try:
            with np.load(npz_path) as data:
                vol_points = data['vol_points']
                vol_sdf = data['vol_sdf']
                near_points = data['near_points']
                near_sdf = data['near_sdf']
                surface = data['surface_points']
        except Exception as e:
            idx = np.random.randint(self.__len__())
            return self.__getitem__(idx)



        ##
        if self.split == 'train':
            scale = torch.rand(1, 3) * 0.5 + 0.75
            scale = scale.numpy()
            vol_points = vol_points * scale
            near_points = near_points * scale
            surface = surface * scale

            distances = np.linalg.norm(surface, axis=1)
            scale = 1 / np.max(distances) * 0.999
            # scale = (1 / np.abs(vw).max()) * 0.99
            surface *= scale
            near_points *= scale
            vol_points *= scale
        ###

        if self.return_surface:
            if self.surface_sampling:
                ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
            surface = torch.from_numpy(surface)


        if self.sampling:
            ###
            pos_vol_id = vol_sdf<0

            if pos_vol_id.sum() > 512:
                ind = np.random.default_rng().choice(pos_vol_id.sum(), self.num_samples//2, replace=False)
                pos_vol_points = vol_points[pos_vol_id][ind]
                pos_vol_sdf = vol_sdf[pos_vol_id][ind]
            else:
                pos_vol_id = near_sdf<0
                if pos_vol_id.sum() > 512:
                    ind = np.random.default_rng().choice(pos_vol_id.sum(), self.num_samples//2, replace=False)
                    pos_vol_points = near_points[pos_vol_id][ind]
                    pos_vol_sdf = near_sdf[pos_vol_id][ind]
                else:
                    ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples//2, replace=False)
                pos_vol_points = vol_points[ind]
                pos_vol_sdf = vol_sdf[ind]

            neg_vol_id = vol_sdf>=0

            ind = np.random.default_rng().choice(neg_vol_id.sum(), self.num_samples//2, replace=False)
            neg_vol_points = vol_points[neg_vol_id][ind]
            neg_vol_sdf = vol_sdf[neg_vol_id][ind]

            vol_points = np.concatenate([pos_vol_points, neg_vol_points], axis=0)
            vol_sdf = np.concatenate([pos_vol_sdf, neg_vol_sdf], axis=0)
            ###

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_sdf = near_sdf[ind]

        
        vol_points = torch.from_numpy(vol_points)
        vol_sdf = torch.from_numpy(vol_sdf).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_sdf = torch.from_numpy(near_sdf).float()

            points = torch.cat([vol_points, near_points], dim=0)
            sdf = torch.cat([vol_sdf, near_sdf], dim=0)
        else:

            near_points = torch.from_numpy(near_points)
            near_sdf = torch.from_numpy(near_sdf).float()

            points = vol_points
            sdf = vol_sdf

        if self.transform:
            surface, points = self.transform(surface, points)

        if self.split == 'train':
            perm = torch.randperm(3)
            points = points[:, perm]
            surface = surface[:, perm]

            negative = torch.randint(2, size=(3,)) * 2 - 1
            points *= negative[None]
            surface *= negative[None]

            roll = torch.randn(1)
            yaw = torch.randn(1)
            pitch = torch.randn(1)

            tensor_0 = torch.zeros(1)
            tensor_1 = torch.ones(1)

            RX = torch.stack([
                            torch.stack([tensor_1, tensor_0, tensor_0]),
                            torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                            torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

            RY = torch.stack([
                            torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                            torch.stack([tensor_0, tensor_1, tensor_0]),
                            torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)

            RZ = torch.stack([
                            torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                            torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                            torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

            R = torch.mm(RZ, RY)
            R = torch.mm(R, RX)

            points = torch.mm(points, R).detach()
            surface = torch.mm(surface, R).detach()




        sdf = (sdf<0).float()

        if self.return_surface:
            return points, sdf, surface, self.models[idx][1], npz_path
        else:
            return points, sdf

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models)

# m = Objaverse('train', sampling=False)
# print(len(m))
# from tqdm import tqdm
# np.random.seed(seed=0)
# import trimesh
# for i in tqdm(range(len(m)), total=len(m)):
#     p, l, s, t, n = m[i]
#     print(i, n)
#     # print(p[l<0].max(dim=0)[0], s.max(dim=0)[0])
#     print(p.shape, l.shape, s.shape, t)
#     # trimesh.PointCloud(s).export('test.ply')
#     print(s.pow(2).sum(dim=1).max())
#     break