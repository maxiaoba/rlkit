import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlkit.torch.networks import Mlp

class MLPAutoEncoder(torch.nn.Module):
    def __init__(self, 
                input_dim,
                output_dim,
                latent_dims,
                encode_mlp_kwargs,
                decode_mlp_kwargs,
                no_gradient,
                ):
        super(MLPAutoEncoder, self).__init__()

        self.no_gradient = no_gradient

        self.encode_mlps = nn.ModuleList()
        for latent_dim in latent_dims:
            mlp = Mlp(
                    input_size=input_dim,
                    output_size=latent_dim,
                    **encode_mlp_kwargs,
                    )
            self.encode_mlps.append(mlp)

        self.decode_mlp = Mlp(
                            input_size=np.sum(latent_dims),
                            output_size=output_dim,
                            **decode_mlp_kwargs,
                            )

    def forward(self, obs, **kwargs):
        if hasattr(self,'no_gradient') and self.no_gradient:
            latent = []
            latent.append(self.encode_mlps[0](obs))
            with torch.no_grad():
                for encode_mlp in self.encode_mlps[1:]:
                    latent.append(encode_mlp(obs))
            latent = torch.cat(latent, dim=-1)
        else:
            latent = torch.cat([
                        encode_mlp(obs) for encode_mlp in self.encode_mlps
                        ], dim=-1)
        output = self.decode_mlp(latent, **kwargs)
        return output


