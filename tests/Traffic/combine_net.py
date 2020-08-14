import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch.networks import Mlp

class CombineNet(torch.nn.Module):
    def __init__(self, 
                encoders,
                decoder,
                no_gradient=False,
                ):
        super(CombineNet, self).__init__()

        self.encoders = nn.ModuleList(encoders)
        self.decoder = decoder
        self.no_gradient = no_gradient

    def forward(self, obs, **kwargs):
        if hasattr(self,'no_gradient') and self.no_gradient:
            latent = []
            latent.append(self.encoders[0](obs))
            with torch.no_grad():
                for encoder in self.encoders[1:]:
                    latent.append(encoder(obs))
            latent = torch.cat(latent, dim=-1)
        else:
            latent = torch.cat([
                        encoder(obs) for encoder in self.encoders
                        ], dim=-1)
        output = self.decoder(latent, **kwargs)
        return output


