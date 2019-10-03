
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


def model_fun(data, observe=True):
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


nuts_kernel = NUTS(model_fun)

mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(ba)










