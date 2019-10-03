import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.distributions import constraints
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS

# INTRO TO PYRO
# Model: a function that can generate data -- these data can be compared to observed data
# Guide: a function that provides proposal distributions for each parameter present in Model

pyro.enable_validation(True)
pyro.clear_param_store()

data = torch.cat((torch.zeros(9), torch.ones(7), torch.empty(4).fill_(2.)))


def model(data):
    alpha = torch.tensor(6.0)
    beta = torch.tensor(10.0)
    pay_probs = pyro.sample('pay_probs', dist.Beta(alpha, beta).expand([3]).independent(1))
    normalized_pay_probs = pay_probs / torch.sum(pay_probs)

    with pyro.plate('data_loop', len(data)):
        pyro.sample('obs', dist.Categorical(probs=normalized_pay_probs), obs=data)


def guide(data):
    alphas = pyro.param('alphas', torch.tensor(6.).expand([3]), constraint=constraints.positive)
    betas = pyro.param('betas', torch.tensor(10.).expand([3]), constraint=constraints.positive)

    pyro.sample('pay_probs', dist.Beta(alphas, betas).independent(1))


def print_progress():
    alphas = pyro.param("alphas")
    betas = pyro.param("betas")

    if torch.cuda.is_available():
        alphas.cuda()
        betas.cuda()

    means = alphas / (alphas + betas)
    normalized_means = means / torch.sum(means)
    factors = betas / (alphas * (1.0 + alphas + betas))
    stdevs = normalized_means * torch.sqrt(factors)

    tiger_pays_string = "probability Tiger pays: {0:.3f} +/- {1:.2f}".format(normalized_means[0], stdevs[0])
    jason_pays_string = "probability Jason pays: {0:.3f} +/- {1:.2f}".format(normalized_means[1], stdevs[1])
    james_pays_string = "probability James pays: {0:.3f} +/- {1:.2f}".format(normalized_means[2], stdevs[2])
    print("[", step, "|", tiger_pays_string, "|", jason_pays_string, "|", james_pays_string, "]")


adam_params = {"lr": 0.0005}
optimizer = Adam(adam_params)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 2501
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print_progress()

nuts_kernel = NUTS(model)

mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(data)

hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
