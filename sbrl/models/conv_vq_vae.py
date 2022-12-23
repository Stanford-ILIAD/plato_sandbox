"""
This is where all the neural network magic happens!

Whether this model is a Q-function, policy, or dynamics model, it's up to you.
"""
from typing import Callable

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from sbrl.experiments import logger
from sbrl.models.model import Model
from sbrl.utils.python_utils import AttrDict
from sbrl.utils.torch_utils import LinConv, MaskConv2d, ResMaskBlock, LayerNorm


class ConvVQVAE(Model):

    def __init__(self, params, env_spec):
        super().__init__(params, env_spec)

        self.K = params.K
        self.D = params.D
        self.H = params.z_H
        self.W = params.z_W
        self.beta = params.beta
        # self.enc, self.dec = self._make_enc_dec(params)
        self.enc = params.enc_block.to_module_list(as_sequential=True)
        self.dec = params.dec_block.to_module_list(as_sequential=True)

        self.preproc = Preprocess()
        self.vq = ConvVQ(self.K, self.D, self.H, self.W)
        self.prior = PixelCNN(self.D, self.H, self.W, self.K, self.vq)

        if hasattr(params, 'loss_fn') and isinstance(params.loss_fn, Callable):
            logger.debug("Using params defined loss fn")
            self._loss_fn = params.loss_fn

        self.to(self.device)

    def sample_decoder(self, xmean):
        xmean = xmean.to(self.device)
        # xmean is image (N, 3, 32, 32) mean
        x = xmean + torch.randn(xmean.shape, device=self.device)
        return torch.clamp(x, -1, 1)

    def encode_forward(self, x, preproc=False, quantize=False):
        if preproc:
            x = self.preproc.forward(x)  # -1 -> 1

        z_e = self.enc(x)  # unquantized
        if quantize:
            z, z_q = self.vq(z_e)  # quantized

            return z, z_e, z_q

        return z_e

    def decode_forward(self, z, postproc=False, quantize=True):
        if quantize:
            z, z_q = self.vq(z)  # quantized

        xmean = self.dec(z)  # -1 -> 1

        if postproc:
            xmean = self.preproc.inverse(xmean)

        return xmean

    def forward(self, x, preproc=False, postproc=False, **kwargs):  # (N, 3, 32, 32):
        z, z_e, z_q = self.encode_forward(x, preproc=preproc, quantize=True)
        xmean = self.decode_forward(z, postproc=postproc, quantize=False)

        return AttrDict(xmean=xmean, z=z, z_e=z_e, z_q=z_q)

    def sample(self, N, device=torch.cuda.current_device(), decoder_noise=False):
        self.eval()

        zs = self.vq.sample(N, self.prior, device=device)
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
            x = self.preproc.forward(x)  # -1 -> 1
            out = self.forward(x)  # -1 -> 1
            xmean, z, z_e, z_q = out.xmean, out.z, out.z_e, out.z_q

            recon = - ((x - xmean) ** 2).mean([-3, -2, -1])  # + np.log(2*np.pi)*3*32*32)
            vq_obj = - ((z_e.detach() - z_q) ** 2).mean([-3, -2, -1])
            commit = - self.beta * ((z_e - z_q.detach()) ** 2).mean([-3, -2, -1])
            loss = -(recon + vq_obj + commit).mean()

            return AttrDict(loss=loss, recon=-recon.mean(), vq_obj=-vq_obj.mean())

###############################################################################


class PixelCNN(nn.Module):
    def __init__(self, in_channels, H, W, out_channels, vq):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = H * W
        self.H = H
        self.W = W
        self.vq = vq

        # 3-layer embedding followed by pixcnn
        net = [
            LinConv(True, False, self.in_channels, 64), nn.ReLU(),
            LinConv(False, False, 64, 64), nn.ReLU(),
            LinConv(False, True, 64, 64), nn.ReLU(),
            MaskConv2d("A", 64, 64, 7, 1, 3), LayerNorm(64), nn.ReLU(),
            ResMaskBlock(64),
            ResMaskBlock(64),
            ResMaskBlock(64),
            ResMaskBlock(64),
            ResMaskBlock(64),
            ResMaskBlock(64),
            ResMaskBlock(64),
            ResMaskBlock(64),
            ResMaskBlock(64),
            ResMaskBlock(64),
            nn.ReLU(),
            MaskConv2d("B", 64, 512, 1), LayerNorm(512), nn.ReLU(),
            MaskConv2d("B", 512, out_channels, 1),
        ]
        self.net = nn.Sequential(*net)

    def forward(self, input):  # (N, 256, 8, 8)
        return self.net.forward(input)

    def loss(self, input):
        out = self.forward(input)  # (N, 256, 8, 8) -> (N, 128, 8, 8)
        true_idx = self.vq.get_idx(input)  # (N,D,H,W) -> (N,H,W)
        # onehot = torch.eye(self.out_channels, device=input.device)[true_idx].permute(0, 3, 1,
        #                                                                              2).contiguous()  # (N,K,H,W)

        return F.cross_entropy(out, true_idx)


################################################################################

# vector quantization
class ConvVQ(nn.Module):
    def __init__(self, K, D, H, W):
        super().__init__()
        self.K = K
        self.D = D
        self.H = H
        self.W = W
        emb = torch.empty((K, D)).uniform_(-1 / K, 1 / K)
        self.emb = nn.Parameter(emb, requires_grad=True)

    def get_idx(self, vec):  # (N,D,H,W)
        distances = (self.emb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) - vec.unsqueeze(1)).norm(
            dim=2)  # (1, K, D, 1, 1) - (N, 1, D, H, W) -> (N, K, H, W)
        min_idx = torch.argmin(distances, dim=1)  # (N, H, W), min across K vectors
        return min_idx

    def forward(self, z_e):
        # nearest neighbor
        min_idx = self.get_idx(z_e)  # (N,H,W)
        z_q = self.emb[min_idx].permute(0, 3, 1, 2).contiguous()  # (K, D) [ (N, H, W) ] -> (N, H, W, D) -> (N, D, H, W)
        return (z_q - z_e).detach() + z_e, z_q

    def sample(self, num_samples, cnn_prior, device=torch.cuda.current_device()):
        self.eval()
        cnn_prior.eval()

        with torch.no_grad():
            in_data = torch.zeros((num_samples, self.D, self.H, self.W), device=device).float()

            for h in range(self.H):
                for w in range(self.W):  # going in order (width first)
                    # ipdb.set_trace()
                    k_dist = cnn_prior.forward(in_data, )[:, :, h, w]  # (N,K,8,8) -> (N,K)

                    cat = D.Categorical(logits=k_dist)

                    idxs = cat.sample()  # (N,) indices in [0, K)
                    chosen = self.emb.index_select(dim=0, index=idxs)  # (N, D)

                    in_data.data[:, :, h, w] = chosen.detach()

        return in_data


###############################################################################


class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.float().permute(0, 3, 1, 2).contiguous()  # x comes in (N,H,W,C)
        x = (x / 0.5) - 1.0  # (-1, 1)
        return x

    def inverse(self, out):  # out came from network
        x = (out + 1) * 0.5  # (0 to 1)
        return x.permute(0, 2, 3, 1).contiguous()  # x comes in (N,H,W,C)
