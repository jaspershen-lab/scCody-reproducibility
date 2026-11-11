from __future__ import annotations
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from sccody.gmvae import sample_gaussian

class scNODE(nn.Module):
    """VAE plus ODE latent dynamics used by the active scNODE baseline."""

    def __init__(self, input_dim, latent_dim, output_dim, vae, diffeq_decoder):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.vae = vae
        self.diffeq_decoder = diffeq_decoder

    @property
    def latent_encoder(self):
        return self.vae.encoder

    @property
    def obs_decoder(self):
        return self.vae.decoder

    def forward(self, data, tps, batch_size=None):
        first_tp_data = data[0]
        if batch_size is not None:
            cell_idx = np.random.choice(np.arange(first_tp_data.shape[0]), size=batch_size, replace=first_tp_data.shape[0] < batch_size)
            first_tp_data = first_tp_data[cell_idx, :]
        mu, logvar = self.vae.encode(first_tp_data)
        std = torch.exp(0.5 * logvar) + 1e-08
        first_latent_dist = dist.Normal(mu, std)
        z0 = sample_gaussian(mu, std)
        latent_seq, g_seq = self._latent_dynamics(z0, tps)
        recon_obs = self.vae.decode(latent_seq)
        return (recon_obs, first_latent_dist, first_tp_data, latent_seq, g_seq)

    def predict(self, first_tp_data, tps_to_predict, n_cells):
        mu, logvar = self.vae.encode(first_tp_data)
        std = torch.exp(0.5 * logvar) + 1e-08
        if mu.shape[0] == 0:
            raise ValueError('scNODE prediction requires at least one source cell')
        repeat_times = (int(n_cells) + mu.shape[0] - 1) // mu.shape[0]
        mu = mu.repeat((repeat_times, 1))[:n_cells, :]
        std = std.repeat((repeat_times, 1))[:n_cells, :]
        z0 = sample_gaussian(mu, std)
        latent_seq, g_seq = self._latent_dynamics(z0, tps_to_predict)
        recon_obs = self.vae.decode(latent_seq)
        return (z0, latent_seq, recon_obs, g_seq)

    def _latent_dynamics(self, z0, tps):
        decoded = self.diffeq_decoder(z0, tps)
        g_seq = torch.zeros(decoded.shape[0], decoded.shape[1], 1, device=decoded.device, dtype=decoded.dtype)
        return (decoded, g_seq)

    def computeDrift(self, latent_var, t=None):
        return self.diffeq_decoder.net.forwardWithTime(t, latent_var)

    def vaeReconstruct(self, data):
        latent_list, recons_list = ([], [])
        dev, dt = self._param_device_dtype()
        for t_data in data:
            x = t_data.to(device=dev, dtype=dt) if torch.is_tensor(t_data) else torch.as_tensor(t_data, device=dev, dtype=dt)
            mu, logvar = self.vae.encode(x)
            std = torch.exp(0.5 * logvar) + 1e-08
            z = sample_gaussian(mu, std)
            latent_list.append(z)
            recons_list.append(self.vae.decode(z))
        return (latent_list, recons_list)

    def _param_device_dtype(self):
        p = next(self.parameters(), None)
        if p is not None:
            return (p.device, p.dtype)
        return (torch.device('cpu'), torch.float32)
