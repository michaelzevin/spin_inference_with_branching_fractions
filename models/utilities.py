import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erf
from jax.scipy.stats import beta

def truncatedNormal(samples,mu,sigma,lowCutoff,highCutoff):

    """
    Jax-enabled truncated normal distribution

    Parameters:
    -----------
    samples : jax.numpy.array or float
        Locations at which to evaluate probability density
    mu : float
        Mean of truncated normal
    sigma : float
        Standard deviation of truncated normal
    lowCutoff : float
        Lower truncation bound
    highCutoff : float
        Upper truncation bound

    Returns:
    --------
    ps : jax.numpy.array or float
        Probability density at the locations of `samples`
    """

    a = (lowCutoff-mu)/jnp.sqrt(2*sigma**2)
    b = (highCutoff-mu)/jnp.sqrt(2*sigma**2)
    norm = jnp.sqrt(sigma**2*np.pi/2)*(-erf(a) + erf(b))
    prob = jnp.exp(-(samples-mu)**2/(2.*sigma**2))/norm
    return prob


def betaDistribution(samples,a,b,betaMin=0,betaMax=1):
    """
    Jax-enabled beta distribution, scaled to the range [betaMin,betaMax]

    Parameters:
    -----------
    samples : jax.numpy.array or float
        Locations at which to evaluate probability density
    a : float
        Beta distribution shape parameter `a`
    b : float
        Beta distribution shape parameter `b`
    betaMin : float
        Lower end of range
    betaMax : float
        Upper end of range

    Returns:
    --------
    ps : jax.numpy.array or float
        Probability density at the locations of `samples`
    """
    prob = beta.pdf(samples, a, b, loc=betaMin, scale=betaMax-betaMin)
    return prob
