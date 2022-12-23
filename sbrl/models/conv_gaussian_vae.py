"""
This is where all the neural network magic happens!

Whether this model is a Q-function, policy, or dynamics model, it's up to you.
"""
from typing import Callable

import torch
import torch.distributions as D
import torch.nn as nn

from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.utils.python_utils import AttrDict


class ConvGaussianVAE(Model):

    def __init__(self, params, env_spec):
        super().__init__(params, env_spec)

        self.zd = params.z_dim;
        self.tzd = params.z_dim ** 2  # (z_dim * (z_dim+1)) // 2
        self.beta = float(params.beta)

        self.pz_mean = nn.Parameter(torch.zeros(self.zd).unsqueeze(0), requires_grad=False)
        self.pz_diag = nn.Parameter(torch.eye(self.zd).unsqueeze(0), requires_grad=False)

        self.enc = params.enc_block.to_module_list(as_sequential=True)
        self.dec = params.dec_block.to_module_list(as_sequential=True)

        self.preproc = Preprocess()

        if hasattr(params, 'loss_fn') and isinstance(params.loss_fn, Callable):
            logger.debug("Using params defined loss fn")
            self._loss_fn = params.loss_fn

        self.to(self.device)

    def raw_to_dist(self, raw, s, ts):
        mean, logdiag = torch.split(raw, [s, s], dim=-1)
        # cov_m = cov.view(-1,s,s)
        diag = torch.exp(logdiag)  # makes strictly positive
        b = torch.eye(diag.size(-1), device=diag.device).float()
        c = diag.unsqueeze(-1).expand(*diag.size(), diag.size(-1))
        cov_m = c * b

        return D.MultivariateNormal(loc=mean, scale_tril=cov_m)

    def sample_encoder(self, rawout):
        zdist = self.raw_to_dist(rawout, self.zd, self.tzd)
        z = zdist.mean + torch.bmm(zdist.covariance_matrix,
                                   torch.randn((rawout.shape[0], self.zd, 1), device=rawout.device)).squeeze(-1)
        return z, zdist

    def sample_decoder(self, xmean):
        xmean = xmean.to(self.device)
        # xmean is image (N, 3, 32, 32) mean
        x = xmean + torch.randn(xmean.shape, device=self.device)
        return torch.clamp(x, -1, 1)

    def decode_forward(self, z, postproc=False):
        xmean = self.dec(z)  # -1 -> 1

        if postproc:
            xmean = self.preproc.inverse(xmean)

        return xmean

    def forward(self, x, preproc=False, postproc=False, encode_only=False, **kwargs):  # (N, 3, 32, 32):
        x = x.float().to(self.device)
        if preproc:
            x = self.preproc.forward(x)  # -1 -> 1

        rawout = self.enc(x)
        z, zdist = self.sample_encoder(rawout)

        if encode_only:
            return z

        xmean = self.dec(z)  # -1 -> 1

        if postproc:
            xmean = self.preproc.inverse(xmean)

        return AttrDict(
            xmean=xmean,
            zdist=zdist,
            z=z
        )

    def sample(self, N, decoder_noise=False):
        zs = torch.randn((N, self.zd)).cuda()
        xmean = self.dec(zs)  # -1 -> 1
        if decoder_noise:
            out = self.sample_decoder(xmean)
        else:
            out = xmean
        return self.preproc.inverse(out)

    # returns elbo, kl, reconstruction
    def loss(self, x):
        if hasattr(self, '_loss_fn'):
            return self._loss_fn(AttrDict(
                x=x,
                model=self,
            ))
        else:
            out = self.forward(x, preproc=True, postproc=True)  # -1 -> 1
            xmean = out.xmean
            z = out.z
            zdist = out.zdist
            recon = - (((x - xmean.double()) ** 2).sum([-1, -2, -3]))  # + np.log(2*np.pi)*3*32*32)

            v1 = torch.diagonal(zdist.covariance_matrix, dim1=-2, dim2=-1)  # (N,zd)
            dkl = 0.5 * (-torch.log(v1).sum(-1) + v1.sum(-1) + zdist.mean.norm(dim=-1) ** 2 - self.zd)
            elbo = recon - self.beta * dkl.double()
            loss = -elbo.mean()

            return AttrDict(loss=loss, recon=-recon.mean(), dkl=dkl.mean())


class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.DTOL = 1e-20

    def forward(self, x):
        x = x.float().permute(0, 3, 1, 2).contiguous()  # x comes in (N,H,W,C)
        # dequantization
        # if dequantize:
        #   x += torch.rand(x.shape, device=x.device)
        # else:
        #   x += 0.5   # middle of bins
        # x /= 256.0
        # return torch.log(x.div(1-x + self.DTOL))
        # x = (x / 128.0) - 1.0  # (-1, 1)
        return x

    def inverse(self, out):  # out came from network
        x = out
        # eo = torch.exp(out)
        # x = (1+self.DTOL)*eo / (1+eo)
        # quantization
        # x = (out + 1) * 128.0  # (0 to 256)
        # x = x.floor().long()
        return x.permute(0, 2, 3, 1).contiguous()  # x comes in (N,H,W,C)
