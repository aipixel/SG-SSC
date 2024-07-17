import torch
import numpy as np
from torchvision.transforms.functional import normalize, resize, adjust_brightness, adjust_contrast,  adjust_gamma, InterpolationMode
from torchvision.transforms.functional import adjust_hue, adjust_saturation

import random
from PIL import Image

from abc import ABC, abstractmethod


class Pad(object):
    """Pad image and mask to the desired size.
    Args:
      size (int) : minimum length/width.
      img_val (tuple) : image padding value.
      msk_val (int) : mask padding value.
    """

    def __init__(self, size, img_val, msk_val):
        self.size = size
        self.img_val = img_val
        self.msk_val = msk_val

    def __call__(self, sample):
        labels = sample['labels']
        h, w = labels.shape[:2]
        h_pad = int(np.clip(((self.size[0] - h) + 1) // 2, 0, 1e6))
        w_pad = int(np.clip(((self.size[1]  - w) + 1) // 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))

        out_sample = {}

        for key in sample.keys():
            data = sample[key]
            if key != 'labels':
                out_data = np.stack(
                                [
                                    np.pad(
                                        data[:, :, c],
                                        pad,
                                        mode="constant",
                                        constant_values=self.img_val[c],
                                    )
                                    for c in range(3)
                                ],
                                axis=2,
                            )
            else:
                out_data = np.pad(
                        data, pad, mode="constant", constant_values=self.msk_val
                    )

            out_sample.update({key: out_data})

        return out_sample


class BaseResize(ABC):
    """Base resize class.

    Args:
    """

    @abstractmethod
    def get_factor(self):
        pass

    def __call__(self, sample):

        labels = sample['labels']

        r_factor = self.get_factor()

        h, w = int(labels.shape[0] * r_factor), int(labels.shape[1] * r_factor)

        out_sample = {}

        for key in sample.keys():
            data = sample[key]
            if key != 'labels':
                out_data = np.array(resize(Image.fromarray(data), [h, w]))
            else:
                #out_data = np.array(resize(Image.fromarray(data), [h, w], Image.NEAREST))
                out_data = np.array(resize(Image.fromarray(data), [h, w], interpolation=InterpolationMode.NEAREST))
            out_sample.update({key: out_data})

        return out_sample


class Resize(BaseResize):
    """Crop randomly the image in a sample.

    Args:
        factor (float): Desired resize factor.
    """

    def __init__(self, factor=0.5):
        assert isinstance(factor, float)
        self.factor = factor

    def get_factor(self):
        return self.factor


class RandomResize(BaseResize):
    """Crop randomly the image in a sample.

    Args:
        factor (tuple): min and mx factor.
    """

    def __init__(self, factor):
        assert isinstance(factor, tuple)
        assert len(factor) == 2
        self.factor = factor

    def get_factor(self):
        return random.uniform(self.factor[0], self.factor[1])


class BaseCrop(ABC):
    """Crop randomly the image in a sample.

    Args:
    """

    @abstractmethod
    def get_top_left(self, h, w):
        pass

    def __call__(self, sample):
        labels = sample['labels']

        h, w = labels.shape[:2]

        new_h, new_w, top, left = self.get_top_left(h, w)

        out_sample = {}

        for key in sample.keys():
            data = sample[key]
            out_data = data[top: top + new_h, left: left + new_w]

            out_sample.update({key: out_data})

        return out_sample


class RandomCrop(BaseCrop):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def get_top_left(self, h, w):
        new_h, new_w = self.output_size
        top = 0 if h <= new_h else np.random.randint(0, h - new_h)
        left = 0 if w <= new_w else np.random.randint(0, w - new_w)
        return new_h, new_w, top, left


class CenterCrop(BaseCrop):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def get_top_left(self, h, w):
        new_h, new_w = self.output_size
        top = (h - new_h)//2
        left = (w - new_w)//2
        return new_h, new_w, top, left


class HorizontalFlip(object):
    """Crop randomly the image in a sample.

    Args:
        prob (float): Flip probability.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):

        rand = np.random.uniform(low=0.0, high=1.0, size=None)

        if rand < self.prob:
            out_sample = {}

            for key in sample.keys():
                data = sample[key]
                out_data = np.flip(data, axis=1).copy()

                out_sample.update({key: out_data})

            return out_sample
        else:
            return sample


class ToTensor(object):
    """Convert arrays in sample to Tensors."""

    def __init__(self, num_classes=11):
        self.num_classes = num_classes

    def __call__(self, sample):
        out_sample = {}

        for key in sample.keys():
            data = sample[key]
            if key != 'labels':
                out_data = torch.from_numpy(data.astype(np.float32).transpose((2, 0, 1)) / 255)
            else:
                out_data = data.astype(np.uint8) - 1  # empty class 0 -> 255
                out_data[out_data == 255] = self.num_classes
                out_data = torch.from_numpy(out_data.astype(np.int))

            out_sample.update({key: out_data})

        return out_sample


class Normalize(object):
    """Crop randomly the image in a sample.

    Args:
        mean (list): Normalize mean.
        std (list): Normalize std.
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        out_sample = {}

        for key in sample.keys():
            data = sample[key]
            if key == 'rgb':
                out_data = normalize(data, mean=self.mean, std=self.std)
            else:
                out_data = data

            out_sample.update({key: out_data})

        return out_sample


class __RandomColorJitter(object):
    """Crop randomly the image in a sample.

    Args:
        brightness (tuple): brightness range.
        contrast (tuple): contrast range.
        gamma (tuple): gamma range.
        hue (tuple): hue range.
        saturation (tuple): saturation range.
    """

    def __init__(self, brightness=(0.7, 1.3), contrast=(0.7, 1.3), gamma=(0.7, 1.3), hue=(-0.05, 0.05),
                 saturation=(0.7, 1.3)):

        assert isinstance(brightness, tuple)
        assert len(brightness) == 2
        self.brightness = brightness

        assert isinstance(contrast, tuple)
        assert len(contrast) == 2
        self.contrast = contrast

        assert isinstance(gamma, tuple)
        assert len(gamma) == 2
        self.gamma = gamma

        assert isinstance(hue, tuple)
        assert len(hue) == 2
        self.hue = hue

        assert isinstance(saturation, tuple)
        assert len(saturation) == 2
        self.saturation = saturation

    def __call__(self, sample):
        rgb, labels = sample['rgb'], sample['labels']

        functions = [
            adjust_brightness, adjust_contrast, adjust_gamma,
            adjust_hue, adjust_saturation
        ]

        ranges = [
            self.brightness, self.contrast, self.gamma,
            self.hue, self.saturation
        ]

        im = Image.fromarray(rgb)

        for function, rg in zip(functions, ranges):
            factor = random.uniform(rg[0], rg[1])
            im = function(im, factor)

        rgb = np.array(im)

        return {'rgb': rgb, 'labels': labels}
