import os
import torch
import pyro
import pandas as pd
from torch.distributions import constraints
from pyro import distributions as dist
from pyro.distributions.util import broadcast_shape
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.poutine as poutine
from pyro.optim import Adam
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from functools import partial

smoke_test = ('CI' in os.environ)
pyro.enable_validation(True) 
torch.set_default_tensor_type(torch.cuda.FloatTensor)

 # <---- This is always a good idea!

pyro.clear_param_store()
N_rows = 100
N_cols = 3
N_covs_patch = 5
N_covs_subp = 5

# observed and true
data = pd.DataFrame({
    'obs': [dist.Poisson(5).sample([N_cols]) for _ in range(N_rows)],
    'true': [dist.Poisson(5).sample([N_cols]) for _ in range(N_rows)]
})

for i in range(N_covs_patch):
    data[f"cov_patch_{i}"] = [dist.Gamma(1, 1).sample([1]) for _ in range(N_rows)]
for j in range(N_covs_subp):
    data[f"cov_subp_{j}"] = [dist.Gamma(1, 1).sample([N_cols]) for _ in range(N_rows)]

@config_enumerate(default='parallel')
@poutine.broadcast
def model_fun1(data, observe):
    # softplus transform as a link function
    softplus = torch.nn.Softplus()

    # extract patch and subpatch data
    patch_cols = [idx for idx, ele in enumerate(data.columns) if 'patch' in ele]
    subp_cols = [idx for idx, ele in enumerate(data.columns) if 'subp' in ele]
    cov_patch = data.iloc[:, patch_cols]
    cov_subp = data.iloc[:, subp_cols]

    # unroll covariate data to Tensors
    gt_N_ice = torch.Tensor([ele.cpu().numpy() for ele in data.true])
    gt_N_obs = torch.Tensor([ele.cpu().numpy() for ele in data.obs])
    cov_p = torch.randn([N_rows, 1, N_covs_patch])
    cov_s = torch.randn([N_rows, N_cols, N_covs_subp])
    for i in range(N_covs_patch):
        cov_p[:, :, i] = torch.Tensor([ele.cpu().numpy() for ele in cov_patch.iloc[:, i]])
    for i in range(N_covs_subp):
        cov_s[:, :, i] = torch.Tensor([ele.cpu().numpy() for ele in cov_subp.iloc[:, i]])

    
    # parameter names -- patch
    patch_par_names = []
    for par in ['a_', 'b_']:
        patch_par_names.extend([
            par + ele for ele in ['lambda_total', 'alpha_haul_prob_patch', 'beta_haul_prob_patch']
        ])

    # parameter names -- subpatch
    subp_par_names = []
    for par in ['a_', 'b_']:
        subp_par_names.extend([
            par + ele for ele in [
                'lambda_false_pos', 'alpha_haul_prob_subp', 'beta_haul_prob_subp', 'alpha_det',
                'beta_det'
            ]
        ])
    
    # parameter starting values
    alphas_patch = pyro.param('alphas_patch', torch.Tensor([1] * len(patch_par_names)), constraint=constraints.positive)
    betas_patch = pyro.param('betas_patch', torch.Tensor([1] * len(patch_par_names)), constraint=constraints.positive)
    alphas_subp = pyro.param('alphas_subp', torch.Tensor([1] * len(subp_par_names)), constraint=constraints.positive)
    betas_subp = pyro.param('betas_subp', torch.Tensor([1] * len(subp_par_names)), constraint=constraints.positive)

    patch_params = {
        ele: pyro.sample(
            ele,
            dist.Gamma(alphas_patch[idx], betas_patch[idx]).expand([N_covs_patch]).independent(1))
        for idx, ele in enumerate(patch_par_names)
    }

    
    subp_params = {
        ele: pyro.sample(
            ele,
            dist.Gamma(alphas_subp[idx], betas_subp[idx]).expand([N_covs_subp]).independent(1))
        for idx, ele in enumerate(subp_par_names)
    }

    # create plates for parallelizing
    x = pyro.plate('x', size=N_rows, dim=-2)
    y = pyro.plate('y', size=N_cols, dim=-1)

    # patch loop
    with x:
        # deterministic linear functions
        lambda_total = softplus(
            torch.sum(patch_params['a_lambda_total'] * cov_p + patch_params['b_lambda_total']))
        alpha_haul_prob_patch = softplus(
            torch.sum(patch_params['a_alpha_haul_prob_patch'] * cov_p +
                      patch_params['b_alpha_haul_prob_patch']))
        beta_haul_prob_patch = softplus(
            torch.sum(patch_params['a_beta_haul_prob_patch'] * cov_p +
                      patch_params['b_beta_haul_prob_patch']))

        # draw haul out probability for patches
        haul_prob_patch = pyro.sample('haul_prob_patch',
                                      dist.Beta(alpha_haul_prob_patch, beta_haul_prob_patch))

        # get total number of seals for patches
        N_total = pyro.sample('N_total', dist.Poisson(lambda_total * haul_prob_patch))

    # subpatch loop
    with x, y:
        # deterministic linear functions
        lambda_false_pos = softplus(
            torch.sum(subp_params['a_lambda_false_pos'] * cov_s +
                      subp_params['b_lambda_false_pos']))
        alpha_haul_prob_subp = softplus(
            torch.sum(subp_params['a_alpha_haul_prob_subp'] * cov_s +
                      subp_params['b_alpha_haul_prob_subp']))
        beta_haul_prob_subp = softplus(
            torch.sum(subp_params['a_beta_haul_prob_subp'] * cov_s +
                      subp_params['b_beta_haul_prob_subp']))
        alpha_det_subp = softplus(
            torch.sum(subp_params['a_alpha_det'] * cov_s + subp_params['b_alpha_det']))
        beta_det_subp = softplus(
            torch.sum(subp_params['a_beta_det'] * cov_s + subp_params['b_beta_det']))

        # draw haul out probability for subpatches (subpatch specific)
        haul_prob_subp = pyro.sample('haul_prob_subp',
                                     dist.Gamma(alpha_haul_prob_subp, beta_haul_prob_subp))
        det_prob_subp = pyro.sample('det_subp', dist.Beta(alpha_det_subp, beta_det_subp))
        false_pos = pyro.sample('false_pos', dist.Poisson(lambda_false_pos))

    if observe:
        for i in pyro.irange('rows', N_rows):
            N_ice = pyro.sample(f'N_ice_{i}',
                                dist.DirichletMultinomial(concentration=haul_prob_subp[i, :],
                                                          total_count=N_total[i], 
                                                          validate_args=False),
                                obs=gt_N_ice[i, :])
            for j in pyro.irange(f'cols_{i}', N_cols):
                pyro.sample(f'N_det_{i}_{j}',
                            dist.Binomial(total_count=(N_ice[j] +
                                                       false_pos[i, j] / det_prob_subp[i, j]).int(),
                                          probs=det_prob_subp[i, j]),
                            obs=gt_N_obs[i, j])


# new formulation with batch size
adam_params = {"lr": 0.0005}
optimizer = Adam(adam_params)
model = partial(model_fun1, observe=True)
guide = partial(model_fun1, observe=False)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 2501
for step in range(n_steps):
    svi.step(data)
# HMC will not work with latent discrete variables
