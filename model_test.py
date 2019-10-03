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
from pyro.util import optional
from pyro import poutine

pyro.set_rng_seed(1)

logging.basicConfig(format='%(message)s', level=logging.INFO)
# Enable validation checks
pyro.enable_validation(True)
smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(1)

# Draw fake parameters -- set number of patches and subpatches
SUBPATCHES_IN_PATCH = 16
N_PATCHES = 100
N_CHAINS = 3
training_data = pd.DataFrame()


# helper function to normalize
def normalize(par_list):
    par_list = par_list.float()
    return (par_list - par_list.mean()) / (par_list.std() + 1E-8)


# Draw fake parameters (at this point, completely uncorrelated)
# -- PATCH SCALE = sea ice cover (patch), chlorophyl, distance to shelf break, julian day
patch_level = ['chl', 'dist', 'jul_day', 'time', 'sea_ice_patch']

# create samplers
chl_sampler = dist.LogNormal(1, 1)
dis_sampler = dist.LogNormal(2, 3)
day_sampler = dist.Categorical(torch.Tensor([float(ele) for ele in range(1, 366)]))
time_sampler = dist.Uniform(0, 24)

# append patch data

training_data = training_data.assign(chl=normalize(chl_sampler.sample([N_PATCHES])),
                                     dist=normalize(dis_sampler.sample([N_PATCHES])),
                                     jul_day=normalize(day_sampler.sample([N_PATCHES])),
                                     time=normalize(time_sampler.sample([N_PATCHES])))

# -- SUBPATCH_SCALE = observed seals, sea ice cover (subpatch), floe size median, time of the day
subpatch_level = ['flo_size', 'N_ice', 'N_obs', 'sea_ice_subp']

# create samplers
flo_size_sampler = dist.LogNormal(1, 1)
sea_ice_cover_sub_sampler = dist.Uniform(0, 100)
seals_sampler = dist.Poisson(5)

# append subpatch data
# sample % sea ice cover at the subpatch level
sea_ice_subp = sea_ice_cover_sub_sampler.sample([N_PATCHES, SUBPATCHES_IN_PATCH])
# append % sea ice cover at the patch level by taking the average (ensures consistency accross scales)
training_data = training_data.assign(
    sea_ice_patch=normalize(torch.Tensor([ele.mean() for ele in sea_ice_subp])),
    sea_ice_subp=normalize(sea_ice_subp),
    flo_size=normalize(flo_size_sampler.sample([N_PATCHES, SUBPATCHES_IN_PATCH])))

# sample seals on ice (use a Bernoulli to simulate unnocupied patches)
bernoulli_haul = dist.Bernoulli(0.3)
N_ice = seals_sampler.sample([N_PATCHES, SUBPATCHES_IN_PATCH]) * bernoulli_haul.sample([N_PATCHES, SUBPATCHES_IN_PATCH])

# get observed seals using a Binomial with true seals assuming 0.4 detection probability + false positives
obs_sampler = dist.Binomial(N_ice, 0.4)
fp_sampler = dist.Poisson(3)

# add seal counts to training data
training_data = training_data.assign(N_obs=
                                     obs_sampler.sample() +
                                     fp_sampler.sample([N_PATCHES, SUBPATCHES_IN_PATCH]),
                                     N_ice=N_ice)


# define model


def model(data):
    # model parameters
    lambda_total = 0.
    haul_prob_patch = 0.
    haul_prob_subp = [0. for i in range(SUBPATCHES_IN_PATCH)]
    det_probs = [0. for i in range(SUBPATCHES_IN_PATCH)]

    # softplus transform link function
    softplus = torch.nn.Softplus()

    # hyperparameter starting values
    a_tot = pyro.sample('a_tot', dist.Normal(8., 1000.))
    b_tot = pyro.sample('b_tot', dist.Normal(0., 1.))
    a_con = pyro.sample('a_con', dist.Normal(8., 1000.))
    b_con = pyro.sample('b_con', dist.Normal(0., 1.))
    a_fp = pyro.sample('a_fp', dist.Normal(8., 1000.))
    b_fp = pyro.sample('b_fp', dist.Normal(0., 1.))
    a1_det0 = pyro.sample('a1_det0', dist.Beta(1, 1))
    b_det0 = pyro.sample('b_det0', dist.Beta(1, 1))
    a1_det1 = pyro.sample('a1_det1', dist.Beta(1, 1))
    b_det1 = pyro.sample('b_det1', dist.Beta(1, 1))

    # model hierarchy
    with pyro.plate('patches', N_PATCHES):
        # linear regression for paramters
        lambda_total = softplus(a_tot * torch.Tensor(data.chl.values) + b_tot)

        # draws total population size
        with poutine.block():
            N_total = pyro.sample("N_total", dist.Poisson(lambda_total))

        # draw haulout probability at the patch level
        haul_prob_patch = pyro.sample('haul_prob_patch', dist.Beta(torch.Tensor([1]), torch.Tensor([1])))

        with pyro.plate('subpatches', SUBPATCHES_IN_PATCH):
            concentration_subp = softplus(a_con * torch.Tensor(data.flo_size) + b_con)
            false_pos = softplus(a_fp * torch.Tensor(data.sea_ice_subp) + b_fp)

            # draw total number of seals on ice
            N_ice = pyro.sample("N_ice", dist.DirichletMultinomial(total_count=(N_total * haul_prob_patch).int(),
                                                                   concentration=concentration_subp),
                                obs=torch.Tensor(data.N_ice))

            # get observed seals (includes false-positives)
            #print(N_ice.shape)
            #print(dist.Poisson(false_pos).sample().shape)
            #False_pos = pyro.sample("False_pos", dist.Poisson(false_pos))
            #det_probs = pyro.sample('det_probs', dist.Beta(softplus(a1_det0 * torch.Tensor(data.sea_ice_subp) + b_det0),
            #                                               a1_det1 * torch.Tensor(data.sea_ice_subp) + b_det1))
            #pyro.sample("N_obs", dist.Binomial(N_ice + False_pos / det_probs, det_probs), obs=torch.Tensor(data.N_obs))


nuts_kernel = NUTS(model)
intial_params = {'a_tot': torch.zeros([N_CHAINS]),
                 'b_tot': torch.zeros([N_CHAINS]),
                 'a_con': torch.zeros([N_CHAINS]),
                 'b_con': torch.zeros([N_CHAINS]),
                 'a_fp': torch.zeros([N_CHAINS]),
                 'b_fp': torch.zeros([N_CHAINS])}


mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200, num_chains=N_CHAINS)
mcmc.run(training_data)


#hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
