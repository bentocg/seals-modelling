
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

smoke_test = ('CI' in os.environ)
pyro.enable_validation(True)    # <---- This is always a good idea!

# We'll ue this helper to check our models are correct.
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)


N_rows = 10
N_cols = 5

ba = pd.DataFrame({'a': [ele for ele in range(N_rows)],
                   'b': [[ele for ele in range(N_cols)] for ele in range(N_rows)]})


def model_fun1(data, observe=True):
    alpha = torch.tensor(6.0)
    beta = torch.tensor(10.0)
    mus = pyro.sample('mus', dist.Gamma(alpha, beta).expand([2]).independent(1))
    x = pyro.plate('x', size=N_rows, dim=-2)
    y = pyro.plate('y', size=N_cols, dim=-1)

    if observe:
        with x:
            coco1 = pyro.sample('coco1', dist.Normal(mus[0], 10), obs=torch.Tensor(data.a))
        with x, y:
            coco2 = pyro.sample('coco2', dist.Normal(mus[1], 10), obs=torch.Tensor(data.b))



nuts_kernel = NUTS(model_fun1)

mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
#mcmc.run(ba)



# more complex situation
N_rows = 100
N_cols = 16
N_covs_patch = 5
N_covs_subp = 5

# observed and true
data = pd.DataFrame({'obs': [dist.Poisson(5).sample([N_rows]) for _ in range(N_cols)],
                     'true': [dist.Poisson(5).sample([N_rows]) for _ in range(N_cols)]})

for i in range(N_covs_patch):
    data[f"cov_patch_{i}"] = [dist.Gamma(1, 1).sample([1]) for _ in range(N_cols)]
for j in range(N_covs_subp):
    data[f"cov_subp_{j}"] = [dist.Gamma(1, 1).sample([N_rows]) for _ in range(N_cols)]

print(data)


def model_fun2(data, observe=True):
    # softplus transform as a link function
    softplus = torch.nn.Softplus()

    # get columns with patch and subpatch covariates
    patch_cols = [idx for idx, ele in data.columns if 'patch' in ele]
    subp_cols = [idx for idx, ele in data.columns if 'subp' in ele]
    cov_patch = data.iloc[:, patch_cols]
    cov_subp = data.iloc[:, subp_cols]

    # convert covariate data to Tensors
    cov_p = torch.random.randn([])


    # parameter starting values
    alphas = torch.tensor(6.0)
    betas = torch.tensor(10.0)

    # draw hyperparameters for linear functions
    a_patch = pyro.sample('a_patch', dist.Gamma(alphas, betas).expand([N_covs_patch]).independent(1))
    b_patch = pyro.sample('b_patch', dist.Gamma(alphas, betas).expand([N_covs_patch]).independent(1))
    a_subp = pyro.sample('a_patch', dist.Gamma(alphas, betas).expand([N_covs_subp]).independent(1))
    b_subp = pyro.sample('b_patch', dist.Gamma(alphas, betas).expand([N_covs_subp]).independent(1))
    x = pyro.plate('x', size=N_rows, dim=-2)
    y = pyro.plate('y', size=N_cols, dim=-1)

    # patch loop
    with x:
        lambda_total = softplus([a_patch * cov_patch + b_patch])
        N_total = pyro.sample('N_total', dist.Poisson)

    # subpatch loop
    with x, y:
        coco2 = pyro.sample('coco2', dist.Normal(mus[1], 10), obs=torch.Tensor(data.b))



