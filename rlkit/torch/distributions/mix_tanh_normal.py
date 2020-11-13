import torch
from torch.distributions import Distribution, Normal, Categorical, Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
import rlkit.torch.pytorch_util as ptu


class MixTanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, weight, normal_mean, normal_std, epsilon=1e-6):
        self.weight = weight
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.mix = Categorical(logits=self.weight)
        self.comp = Independent(Normal(normal_mean, normal_std),1)
        self.gmm = MixtureSameFamily(self.mix, self.comp)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.gmm.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.gmm.log_prob(pre_tanh_value).unsqueeze(-1) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.gmm.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        ind = torch.nn.functional.gumbel_softmax(self.weight,tau=1, hard=True)
        # onehot batch x num_d
        normal = Normal(self.normal_mean[ind.bool()],self.normal_std[ind.bool()])
        z = normal.rsample()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def entropy(self):
        """Returns entropy of the underlying normal distribution.

        Returns:
            torch.Tensor: entropy of the underlying normal distribution.

        """
        raise NotImplementedError
