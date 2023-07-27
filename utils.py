import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class AUGLoss(nn.Module):
    """
    Custom loss module to calculate the Average Uniqueness Group Loss (AUG Loss).
    The AUG Loss measures the distance between two feature vectors (x1, x2).
    """

    def __init__(self):
        super(AUGLoss, self).__init__()

    def forward(self, x1, x2):
        """
        Calculates the AUG Loss between two input feature vectors.

        Args:
            x1 (torch.Tensor): Input feature vector 1.
            x2 (torch.Tensor): Input feature vector 2.

        Returns:
            torch.Tensor: The AUG Loss value.
        """
        b = (x1 - x2)
        b = b * b
        b = b.sum(1)
        b = torch.sqrt(b)
        return b.sum()

class RandomRotate(object):
    """
    Data augmentation class to randomly rotate images.
    """

    def __call__(self, images):
        """
        Applies random rotation to a batch of input images.

        Args:
            images (numpy.ndarray): Batch of input images.

        Returns:
            numpy.ndarray: Batch of randomly rotated images.
        """
        rotated = np.stack([self.random_rotate(x) for x in images])
        return rotated
    
    def random_rotate(self, image):
        """
        Randomly rotates a single input image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Rotated image.
        """
        rand_num = np.random.randint(0, 4)
        if rand_num == 0:
            return np.rot90(image, k=1, axes=(0, 1))
        elif rand_num == 1:
            return np.rot90(image, k=2, axes=(0, 1))
        elif rand_num == 2:
            return np.rot90(image, k=3, axes=(0, 1))
        else:
            return image

class Normalize(object):
    """
    Data augmentation class to normalize images.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, images):
        """
        Applies normalization to a batch of input images.

        Args:
            images (numpy.ndarray): Batch of input images.

        Returns:
            numpy.ndarray: Batch of normalized images.
        """
        normalized = np.stack([F.normalize(x, self.mean, self.std, self.inplace) for x in images])
        return normalized

class Grayscale(object):
    """
    Data augmentation class to convert images to grayscale with a given probability.
    """

    def __init__(self, prob=1):
        self.prob = prob

    def __call__(self, images):
        """
        Converts a batch of input images to grayscale with a certain probability.

        Args:
            images (numpy.ndarray): Batch of input images.

        Returns:
            numpy.ndarray: Batch of images, possibly converted to grayscale.
        """
        random_num = np.random.randint(100, size=1)[0]
        if random_num <= self.prob * 100:
            gray_images = (images[:, 0, :, :] + images[:, 1, :, :] + images[:, 2, :, :]) / 3
            gray_scaled = gray_images.unsqueeze(1).repeat(1, 3, 1, 1)
            return gray_scaled
        else:
            return images

class ToTensor(object):
    """
    Data transformation class to convert numpy arrays to PyTorch tensors.
    """

    def __call__(self, images):
        """
        Converts a batch of numpy arrays to PyTorch tensors.

        Args:
            images (numpy.ndarray): Batch of input images.

        Returns:
            torch.Tensor: Batch of PyTorch tensors.
        """
        images = images.transpose((0, 3, 1, 2))
        return torch.from_numpy(images).float()

class AverageMeter(object):
    """
    Helper class to keep track of and calculate averages for certain values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the values of the average meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the average meter with a new value.

        Args:
            val (float): New value to update the average meter with.
            n (int): Number of samples corresponding to the new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def merge_on_lat_lon(df1, df2, keys=['cluster_lat', 'cluster_lon'], how='inner'):
    """
    Allows two dataframes to be merged on lat/lon
    Necessary because pandas has trouble merging on floats (understandably so)
    """
    df1 = df1.copy()
    df2 = df2.copy()
    
    # must use ints for merging, as floats induce errors
    df1['merge_lat'] = (10000 * df1[keys[0]]).astype(int)
    df1['merge_lon'] = (10000 * df1[keys[1]]).astype(int)
    
    df2['merge_lat'] = (10000 * df2[keys[0]]).astype(int)
    df2['merge_lon'] = (10000 * df2[keys[1]]).astype(int)
    
    df2.drop(keys, axis=1, inplace=True)
    merged = pd.merge(df1, df2, on=['merge_lat', 'merge_lon'], how=how)
    merged.drop(['merge_lat', 'merge_lon'], axis=1, inplace=True)
    return merged

def create_space(lat, lon, s=10):
    """Creates a s km x s km square centered on (lat, lon)"""
    v = (180/math.pi)*(500/6378137)*s # roughly 0.045 for s=10
    return lat - v, lon - v, lat + v, lon + v

def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

def deg2num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return (xtile, ytile)