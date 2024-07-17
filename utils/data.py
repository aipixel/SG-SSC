from fnmatch import fnmatch
import os
import torch
from torch.utils.data import TensorDataset
from skimage import io
import numpy as np
import random


def get_file_prefixes_from_path(data_path, criteria="*.bin"):
    prefixes = []

    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if fnmatch(name, criteria):
                prefixes.append(os.path.join(path, name)[:-(len(criteria)-1)])

    prefixes.sort()

    return prefixes


class MultimodalDataset(TensorDataset):
    """Face Landmarks dataset."""

    def __init__(self, file_prefixes, transf=None, read_rgb=True, read_normals=False, read_xyz=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transf   (callable, optional): Optional transform to be applied
                on a sample.
        """

        """
        scenes_file = os.path.join(root_dir, file_name)
        with open(scenes_file,"r") as f:
            self.scenes = f.readlines()
        self.root_dir = root_dir
        """
        self.scenes = file_prefixes
        self.transf = transf
        self.rgb = read_rgb
        self.normals = read_normals
        self.xyz = read_xyz

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'labels': self.read_labels(self.scenes[idx])}
        if self.rgb:
            sample.update({'rgb': self.read_rgb(self.scenes[idx])})
        if self.normals:
            sample.update({'normals': self.read_normals(self.scenes[idx])})
        if self.xyz:
            sample.update({'xyz': self.read_xyz(self.scenes[idx])})

        if self.transf:
            sample = self.transf(sample)

        return sample

    def read_rgb(self, prefix):
        #  return io.imread('{}/data/images/img_{}.png'.format(self.root_dir, prefix[:4]))
        return io.imread('{}_color.jpg'.format(prefix))

    def read_labels(self, prefix):
        import numpy as np
        #  im = scipy.io.loadmat('{}/segmentation/img_{}.mat'.format(self.root_dir, prefix[:4]))['segmentation']
        im = io.imread('{}_labels.png'.format(prefix))#, as_gray=True)
        return im

    #  def read_hha(self, prefix):
    #    im = io.imread('{}/data/hha/img_{}.png'.format(self.root_dir, prefix[:4]))
    #    return im

    def read_normals(self, prefix):
        im = io.imread('{}_normals.png'.format(prefix))
        return im

    def read_xyz(self, prefix):
        im = io.imread('{}_xyz.png'.format(prefix))
        return im

class DL2Dev:
    def __init__(self, dl, dev):
        self.dl = dl
        self.dev = dev
        self.dataset = dl.dataset

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            y = None
            x = []
            for key in b.keys():
                data = b[key].to(self.dev)
                if key == 'labels':
                    y = data
                else:
                    x.append(data)

            yield x, y


class SSCMultimodalDataset(TensorDataset):
    """Face Landmarks dataset."""

    def __init__(self, file_prefixes, data_augmentation=False):
        """
        Args:
            file_prefixes: list of preprocessed files prefixes to read.
        """
        self.scenes = file_prefixes
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        loaded = np.load(self.scenes[idx]+'.npz')
        vox_tsdf = loaded['vox_tsdf']
        vox_prior = loaded['vox_prior']
        gt = loaded['gt']
        vox_weights = loaded['vox_weights']
        cam_pose = loaded['cam_pose']
        vox_origin = loaded['vox_origin']

        if self.data_augmentation:
            if random.random() >= .5:
                vox_tsdf = np.swapaxes(vox_tsdf, axis1=0, axis2=2).copy()
                vox_prior = np.swapaxes(vox_prior, axis1=0, axis2=2).copy()
                gt = np.swapaxes(gt, axis1=0, axis2=2).copy()
                vox_weights = np.swapaxes(vox_weights, axis1=0, axis2=2).copy()

            if random.random() >= .5:
                vox_tsdf = np.flip(vox_tsdf, axis=0).copy()
                vox_prior = np.flip(vox_prior, axis=0).copy()
                gt = np.flip(gt, axis=0).copy()
                vox_weights = np.flip(vox_weights, axis=0).copy()

            if random.random() >= .5:
                vox_tsdf = np.flip(vox_tsdf, axis=2).copy()
                vox_prior = np.flip(vox_prior, axis=2).copy()
                gt = np.flip(gt, axis=2).copy()
                vox_weights = np.flip(vox_weights, axis=2).copy()

        sample = {
            'vox_tsdf': torch.from_numpy(vox_tsdf).reshape(1, 240, 144, 240),
            'vox_prior': torch.from_numpy(np.moveaxis(vox_prior, -1, 0)),
            'gt': torch.from_numpy(gt),
            'vox_weights':  torch.from_numpy(vox_weights),
            'cam_pose':  torch.from_numpy(cam_pose),
            'vox_origin':  torch.from_numpy(vox_origin)
        }
        return sample

def img_pred(img_prior):
    out_h = 480
    out_w = 640
    in_h = 468
    in_w = 625
    
    top, left = (out_h - in_h) // 2, (out_w - in_w) // 2
    
    prior = np.zeros((480, 640, 12), np.float32)
    prior[:,:,0] = 1.0
    
    a = np.zeros((480, 640, 1), np.float32)
    img_prior = np.concatenate((a, img_prior), axis=-1)
    
    prior[top:top + in_h, left:left + in_w:] = img_prior[top:top + in_h, left:left + in_w:]
    return prior

def sample2dev(in_samp,dev):
    out_samp = {}
    for key in in_samp.keys():
        out_samp.update({key:in_samp[key].to(dev)})
    return out_samp