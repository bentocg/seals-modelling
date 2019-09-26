from __future__ import absolute_import, division, print_function

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO
from pyro.contrib.autoguide import AutoMultivariateNormal
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
import pyro.optim as optim

pyro.set_rng_seed(1)


logging.basicConfig(format='%(message)s', level=logging.INFO)
# Enable validation checks
pyro.enable_validation(True)
smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(1)
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
rugged_data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")


# Draw fake parameters -- set number of patches and subpatches
SUBPATCHES_IN_PATCH = 16
N_PATCHES = 100
patch_data = pd.DataFrame()
subpatch_data = pd.DataFrame()

# helper function to normalize
def normalize(par_list):
    par_list = par_list.float()
    return (par_list - par_list.mean()) / (par_list.std() + 1E-8)


# Draw fake parameters (at this point, completely uncorrelated)
# -- PATCH SCALE = sea ice cover (patch), chlorophyl, distance to shelf break, julian day
# create samplers
chl_sampler = torch.distributions.log_normal.LogNormal(1, 1)
dis_sampler = torch.distributions.log_normal.LogNormal(2, 3)
day_sampler = torch.distributions.Categorical(torch.Tensor([float(ele) for ele in range(1, 366)]))

# append patch data
patch_data.append({'chl': normalize(chl_sampler.sample([N_PATCHES])),
                   'dist': normalize(dis_sampler.sample([N_PATCHES])),
                   'day': normalize(day_sampler.sample([N_PATCHES]))}, ignore_index=True)


# -- SUBPATCH_SCALE = observed seals, sea ice cover (subpatch), floe size median, time of the day
# create samplers
flo_size_sampler = torch.distributions.log_normal.LogNormal(1, 1)
time_sampler = torch.distributions.uniform.Uniform(0, 24)
sea_ice_cover_sub_sampler = torch.distributions.uniform.Uniform(0, 100)

# normalize parameter values
sea_ice_cover_patch = []
for i in range(N_PATCHES):
    sea_ice_subp = sea_ice_cover_sub_sampler.sample(SUBPATCHES_IN_PATCH)
    sea_ice_cover_patch.append(sea_ice_subp.sum())
    subpatch_data.append({'sea_ice_cover_subp': })

patch_data.assign(sea_ice_patch=normalize(sea_ice_cover_patch))

# define model
def model(sea_ice, observed_seals, julian_day, time, :
    a = pyro.sample("a", dist.Normal(8., 1000.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("bAR", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
    with pyro.plate("data", len(ruggedness)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)