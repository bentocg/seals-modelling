import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import seaborn as sns
import torch
from pyro.nn import PyroModule
from torch import nn
from pyro.distributions import Poisson, Binomial, Beta, Gamma
import math
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import logging

# for CI testing
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.1.0')
pyro.set_rng_seed(1)

# Set matplotlib settings
plt.style.use('default')

patch_data = gpd.read_file('/home/bento/seals-modelling/covariates/covariates_patch.dbf')
subp_data = gpd.read_file('/home/bento/seals-modelling/covariates/covariates_subp.dbf')
logging.basicConfig(format='%(message)s', level=logging.INFO)
print(subp_data.columns)

# convert data to tensors
n_ice = torch.Tensor(subp_data.n_ice)
n_obs = torch.Tensor(subp_data.n_obs)
floe_size = torch.Tensor(subp_data.floe_size)
cover_subp = torch.Tensor(subp_data.sea_ice_co)
sigmoid = nn.Sigmoid()


def model(n_ice, n_obs, floe_size, cover_subp):
    a_floe = pyro.sample("a_floe", dist.Normal(1., 1.))
    b_floe = pyro.sample("b_floe", dist.Normal(0., 1.))
    a_cover = pyro.sample("a_cover", dist.Normal(1., 1.))
    b_cover = pyro.sample("b_cover", dist.Normal(0., 1.))
    a_cover_b = pyro.sample("a_cover_b", dist.Normal(1., 1.))
    b_cover_b = pyro.sample("b_cover_b", dist.Normal(0., 1.))
    lambda_ice = sigmoid((a_floe * floe_size + b_floe))
    alpha_det = sigmoid((a_cover * cover_subp + b_cover))
    beta_det = sigmoid((a_cover_b * cover_subp + b_cover_b))
    with pyro.plate('subp', size=len(floe_size)):
        N_ice = pyro.sample('N_ice', Poisson(lambda_ice), obs=n_ice)
        phi_det = pyro.sample('phi_det', Beta(alpha_det, beta_det)) * (N_ice > 0).float()
        N_obs = pyro.sample('N_obs', Binomial(N_ice, phi_det), obs=n_obs)


def guide(n_ice, n_obs, floe_size, cover_subp):
    a_floe_loc = pyro.param('a_floe_loc', torch.tensor(0.))
    b_floe_loc = pyro.param('b_floe_loc', torch.tensor(0.))
    a_floe_scale = pyro.param('a_floe_scale', torch.tensor(1.), constraint=constraints.greater_than(0.01))
    b_floe_scale = pyro.param('b_floe_scale', torch.tensor(1.), constraint=constraints.greater_than(0.01))
    a_cover_loc = pyro.param('a_cover_loc', torch.tensor(1.))
    b_cover_loc = pyro.param('b_cover_loc', torch.tensor(1.))
    a_cover_b_loc = pyro.param('a_cover_b_loc', torch.tensor(1.))
    b_cover_b_loc = pyro.param('b_cover_b_loc', torch.tensor(1.))
    a_cover_scale = pyro.param('a_cover_scale', torch.tensor(1.), constraint=constraints.greater_than(0.01))
    b_cover_scale = pyro.param('b_cover_scale', torch.tensor(1.), constraint=constraints.greater_than(0.01))
    a_cover_b_scale = pyro.param('a_cover_b_scale', torch.tensor(1.), constraint=constraints.greater_than(0.01))
    b_cover_b_scale = pyro.param('b_cover_b_scale', torch.tensor(1.), constraint=constraints.greater_than(0.01))
    a_floe = pyro.sample("a_floe", dist.LogNormal(a_floe_loc, a_floe_scale))
    b_floe = pyro.sample("b_floe", dist.LogNormal(b_floe_loc, b_floe_scale))
    a_cover = pyro.sample("a_cover", dist.LogNormal(a_cover_loc, a_cover_scale))
    b_cover = pyro.sample("b_cover", dist.LogNormal(b_cover_loc, b_cover_scale))
    a_cover_b = pyro.sample("a_cover_b", dist.LogNormal(a_cover_b_loc, a_cover_b_scale))
    b_cover_b = pyro.sample("b_cover_b", dist.LogNormal(b_cover_b_loc, b_cover_b_scale))
    lambda_ice = sigmoid((a_floe * floe_size + b_floe))
    alpha_det = sigmoid((a_cover * cover_subp + b_cover))
    beta_det = sigmoid((a_cover_b * cover_subp + b_cover_b))
    with pyro.plate('subp', size=len(floe_size), subsample_size=10):
        phi_det = pyro.sample('phi_det', Beta(alpha_det, beta_det))


# set up the optimizer
adam_params = {"lr": 1, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

#from pyro.infer import MCMC, NUTS
#nuts_kernel = NUTS(model)
#mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=0)
#mcmc.run(n_ice, n_obs, floe_size, cover_subp)
#hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())


n_steps = 30000
# do gradient steps
for step in range(n_steps):
    elbo = svi.step(n_ice, n_obs, floe_size, cover_subp)
    if step % 200 == 0:
        print('.', end='')
        logging.info("Elbo loss: {}".format(elbo))

# grab the learned variational parameters
alpha_q = pyro.param("a_loc").item()
beta_q = pyro.param("b_loc").item()

# here we use some facts about the beta distribution
# compute the inferred mean of the coin's fairness
print(alpha_q)
print(beta_q)
