"""Utility functions for PyTorch algorithms. (from garage)

A collection of common functions that are used by Pytorch algos.

This collection of functions can be used to manage the following:
    - Pytorch GPU usage
        - setting the default Pytorch GPU
        - converting Tensors to GPU Tensors
        - Converting Tensors into `numpy.ndarray` format and vice versa
    - Updating model parameters
"""
import copy

import gym
import torch
from torch import nn
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu

def compute_advantages(discount, gae_lambda, max_episode_length, baselines,
                       rewards):
    """Calculate advantages.

    Advantages are a discounted cumulative sum.

    Calculate advantages using a baseline according to Generalized Advantage
    Estimation (GAE)

    The discounted cumulative sum can be computed using conv2d with filter.
    filter:
        [1, (discount * gae_lambda), (discount * gae_lambda) ^ 2, ...]
        where the length is same with max_episode_length.

    baselines and rewards are also has same shape.
        baselines:
        [ [b_11, b_12, b_13, ... b_1n],
          [b_21, b_22, b_23, ... b_2n],
          ...
          [b_m1, b_m2, b_m3, ... b_mn] ]
        rewards:
        [ [r_11, r_12, r_13, ... r_1n],
          [r_21, r_22, r_23, ... r_2n],
          ...
          [r_m1, r_m2, r_m3, ... r_mn] ]

    Args:
        discount (float): RL discount factor (i.e. gamma).
        gae_lambda (float): Lambda, as used for Generalized Advantage
            Estimation (GAE).
        max_episode_length (int): Maximum length of a single rollout.
        baselines (torch.Tensor): A 2D vector of value function estimates with
            shape (N, T), where N is the batch dimension (number of episodes)
            and T is the maximum path length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining
            elements in that episode should be set to 0.
        rewards (torch.Tensor): A 2D vector of per-step rewards with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum path length experienced by the agent. If an episode
            terminates in fewer than T time steps, the remaining elements in
            that episode should be set to 0.

    Returns:
        torch.Tensor: A 2D vector of calculated advantage values with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum path length experienced by the agent. If an episode
            terminates in fewer than T time steps, the remaining values in that
            episode should be set to 0.

    """
    adv_filter = torch.full((1, 1, 1, max_episode_length - 1),
                            discount * gae_lambda).to(ptu.device)
    adv_filter = torch.cumprod(F.pad(adv_filter, (1, 0), value=1), dim=-1)

    deltas = (rewards + discount * F.pad(baselines, (0, 1))[:, 1:] - baselines)
    deltas = F.pad(deltas,
                   (0, max_episode_length - 1)).unsqueeze(0).unsqueeze(0)

    advantages = F.conv2d(deltas, adv_filter, stride=1).reshape(rewards.shape)
    return advantages


def pad_to_last(nums, total_length, axis=-1, val=0):
    """Pad val to last in nums in given axis.

    length of the result in given axis should be total_length.

    Raises:
      IndexError: If the input axis value is out of range of the nums array

    Args:
        nums (numpy.ndarray): The array to pad.
        total_length (int): The final width of the Array.
        axis (int): Axis along which a sum is performed.
        val (int): The value to set the padded value.

    Returns:
        torch.Tensor: Padded array

    """
    tensor = torch.Tensor(nums)
    axis = (axis + len(tensor.shape)) if axis < 0 else axis

    if len(tensor.shape) <= axis:
        raise IndexError('axis {} is out of range {}'.format(
            axis, tensor.shape))

    padding_config = [0, 0] * len(tensor.shape)
    padding_idx = abs(axis - len(tensor.shape)) * 2 - 1
    padding_config[padding_idx] = max(total_length - tensor.shape[axis], val)
    return F.pad(tensor, padding_config)


def filter_valids(tensor, valids):
    """Filter out tensor using valids (last index of valid tensors).

    valids contains last indices of each rows.

    Args:
        tensor (torch.Tensor): The tensor to filter
        valids (list[int]): Array of length of the valid values

    Returns:
        torch.Tensor: Filtered Tensor

    """
    return [tensor[i][:valid] for i, valid in enumerate(valids)]


def flatten_batch(tensor):
    """Flatten a batch of observations.

    Reshape a tensor of size (X, Y, Z) into (X*Y, Z)

    Args:
        tensor (torch.Tensor): Tensor to flatten.

    Returns:
        torch.Tensor: Flattened tensor.

    """
    return tensor.reshape((-1, ) + tensor.shape[2:])


def flatten_to_single_vector(tensor):
    """Collapse the C x H x W values per representation into a single long vector.

    Reshape a tensor of size (N, C, H, W) into (N, C * H * W).

    Args:
        tensor (torch.tensor): batch of data.

    Returns:
        torch.Tensor: Reshaped view of that data (analogous to numpy.reshape)

    """
    N = tensor.shape[0]  # read in N, C, H, W
    return tensor.view(N, -1)

def product_of_gaussians(mus, sigmas_squared):
    """Compute mu, sigma of product of gaussians.

    Args:
        mus (torch.Tensor): Means, with shape :math:`(N, M)`. M is the number
            of mean values.
        sigmas_squared (torch.Tensor): Variances, with shape :math:`(N, V)`. V
            is the number of variance values.

    Returns:
        torch.Tensor: Mu of product of gaussians, with shape :math:`(N, 1)`.
        torch.Tensor: Sigma of product of gaussians, with shape :math:`(N, 1)`.

    """
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared