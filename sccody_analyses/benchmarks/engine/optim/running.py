from __future__ import annotations
import torch
import torch.nn as nn
from sccody_analyses.benchmarks.engine.scnode_model.diff_solver import ODE
from sccody_analyses.benchmarks.engine.scnode_model.dynamic_model import scNODE
from sccody_analyses.benchmarks.engine.scnode_model.layer import LinearNet, LinearVAENet
from sccody_analyses.checkpoint_selection import METRICS, CheckpointSelector

def add_checkpoint_selection_args(parser, *, default_every: int) -> None:
    group = parser.add_argument_group('checkpoint selection')
    group.add_argument('--epoch-selection-metric', choices=list(METRICS), default='holdout_L_gene')
    group.add_argument('--normal-holdout-selection-every-epochs', type=int, default=default_every)
    group.add_argument('--normal-holdout-selection-min-cells', type=int, default=100)
    group.add_argument('--normal-holdout-selection-max-groups', type=int, default=80)
    group.add_argument('--normal-holdout-selection-max-cells', type=int, default=256)

class ScNODEVAE(nn.Module):
    """Adapter for the upstream scNODE VAE modules.

    The scNODE wrapper here expects encode() to return (mu, logvar), while
    the upstream encoder returns (mu, std).  Convert at the boundary so the rest
    of the training code can keep the current interface without changing the
    underlying scNODE architecture.
    """

    def __init__(self, input_dim: int, latent_dim: int, enc_latent_list, dec_latent_list, latent_enc_act: str, latent_dec_act: str):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.uses_library_size = False
        self.encoder = LinearVAENet(input_dim=self.input_dim, latent_size_list=list(enc_latent_list) if enc_latent_list else None, output_dim=self.latent_dim, act_name=latent_enc_act)
        self.decoder = LinearNet(input_dim=self.latent_dim, latent_size_list=list(dec_latent_list) if dec_latent_list else None, output_dim=self.input_dim, act_name=latent_dec_act)

    def encode(self, x):
        mu, std = self.encoder(x)
        logvar = 2.0 * torch.log(std.clamp_min(1e-08))
        return (mu, logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) + 1e-08
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return (self.decode(z), mu, logvar)

def constructscNODEModel(n_genes, latent_dim, enc_latent_list=None, dec_latent_list=None, drift_latent_size=None, latent_enc_act: str='none', latent_dec_act: str='relu', drift_act: str='relu', ode_method: str='euler', ode_step_size: float | None=None):
    """Construct the scNODE comparator from the upstream scNODE modules."""
    if enc_latent_list is None:
        enc_latent_list = [256]
    if dec_latent_list is None:
        dec_latent_list = [256]
    if drift_latent_size is None:
        drift_latent_size = [512]
    vae = ScNODEVAE(input_dim=n_genes, latent_dim=latent_dim, enc_latent_list=enc_latent_list, dec_latent_list=dec_latent_list, latent_enc_act=latent_enc_act, latent_dec_act=latent_dec_act)
    drift = LinearNet(input_dim=latent_dim, latent_size_list=list(drift_latent_size) if drift_latent_size else None, output_dim=latent_dim, act_name=drift_act)
    decoder = ODE(input_dim=latent_dim, drift_net=drift, ode_method=ode_method, step_size=ode_step_size)
    return scNODE(input_dim=n_genes, latent_dim=latent_dim, output_dim=n_genes, vae=vae, diffeq_decoder=decoder)

def scNODEPredict(latent_ode_model, first_tp_data, tps, n_cells):
    """Predict expression trajectories from a trained scNODE model."""
    latent_ode_model.eval()
    with torch.no_grad():
        _, _, all_pred_data, g_seq = latent_ode_model.predict(first_tp_data, tps, n_cells=n_cells)
    return (all_pred_data.cpu().detach().numpy(), g_seq.cpu().detach().numpy())
