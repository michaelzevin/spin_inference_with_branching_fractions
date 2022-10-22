import numpy as np
import arviz as az
import numpyro
from numpyro.infer import NUTS,MCMC
from jax import random
import getData
from models import likelihoods
import pdb

np.random.seed(190412)


# Run over several chains to check convergence
nChains = 3
numpyro.set_host_device_count(nChains)

# Get dictionaries holding injections and posterior samples
injectionDict = getData.getInjections()
sampleDict = getData.getSamples()

# Set up NUTS sampler over our likelihood
kernel = NUTS(likelihoods.betaPlusIsotropic)
mcmc = MCMC(kernel,num_warmup=300,num_samples=3000,num_chains=nChains)

# Choose a random key and run over our model
rng_key = random.PRNGKey(11)
rng_key,rng_key_ = random.split(rng_key)
mcmc.run(rng_key_,sampleDict,injectionDict)
mcmc.print_summary()

# Save data
data = az.from_numpyro(mcmc)
az.to_netcdf(data,"./output/betaPlusIsotropic_spintilts.cdf")
