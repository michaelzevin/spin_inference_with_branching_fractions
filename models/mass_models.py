import numpy as np
import jax.numpy as jnp

###################
### MASS MODELS ###
###################

def PowerlawPlusPeakPrimary(m1, alpha=-3.51, mMin=5.00, mMax=88.21, lambdaPeak=0.033, meanPeak=33.61, sigmaPeak=4.72, dmMinTaper=4.88, dmMaxTaper=5.0):
    """
    Powerlaw+Peak mass model for calculating p(m1)

    Default parameters correspond to the median values from GWTC-3 populations analysis
        (https://arxiv.org/pdf/2111.03634.pdf)


    Parameters:
    -----------
    m1 : float or array
        primary mass sample(s)
    alpha : float
        Spectral index for the power-law of the primary mass distribution
    mMin : float
        Minimum mass of the power-law component of the primary mass distribution
    mMax : float
        Maximum mass of the power-law component of the primary mass distribution
    lambdaPeak : float
        Fraction of BBH systems in the Gaussian component
    meanPeak : float
        Mean of the Gaussian component in the primary mass distribution
    sigmaPeak : float
        Width of the Gaussian component in the primary mass distribution
    dmMinTaper : float
        Range of mass tapering at the lower end of the mass distribution
    dmMaxTaper : float
        Range of mass tapering at the upper end of the mass distribution (FIXME)

    Returns:
    --------
    p_m1 : jax.numpy.array
        Unnormalized array of probability densities
    """

    # power law component for m1
    p_m1_pl = (1.+alpha)*m1**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
    # gaussian peak
    p_m1_peak = jnp.exp(-0.5*(m1-meanPeak)**2./sigmaPeak**2)/jnp.sqrt(2.*np.pi*sigmaPeak**2.)
    # combined
    p_m1 = lambdaPeak*p_m1_peak + (1.-lambdaPeak)*p_m1_pl
    # low- and high-mass smoothing
    low_filter = jnp.exp(-(m1-mMin)**2/(2.*dmMinTaper**2))
    low_filter = jnp.where(m1<mMin,low_filter,1.)
    high_filter = jnp.exp(-(m1-mMax)**2/(2.*dmMaxTaper**2))
    high_filter = jnp.where(m1>mMax,high_filter,1.)
    p_m1 *= low_filter*high_filter

    return p_m1


def PowerlawSecondary(m1, m2, betaq=0.96, mMin=5.00, dmMinTaper=4.88):
    """
    Powerlaw model for secondary mass p(m2|m1)

    Default parameters correspond to the median values from GWTC-3 populations analysis
        (https://arxiv.org/pdf/2111.03634.pdf)


    Parameters:
    -----------
    m1 : float or array
        primary mass sample(s)
    m2 : float or array
        secondary mass sample(s)
    betaq : float
        Spectral index for the power-law of the mass ratio distribution
    mMin : float
        Minimum mass of the power-law component of the primary mass distribution
    dmMinTaper : float
        Range of mass tapering at the lower end of the mass distribution

    Returns:
    --------
    p_m2 : jax.numpy.array
        the mass model evaluated at the input samples m2
    """

    # powerlaw for m2 dependent on m1
    p_m2 = (1.+betaq)*jnp.power(m2,betaq)/(jnp.power(m1,1.+betaq)-jnp.power(mMin,1.+betaq))

    # low-mass smoothing
    low_filter = jnp.exp(-(m2-mMin)**2/(2.*dmMinTaper**2))
    low_filter = jnp.where(m2<mMin,low_filter,1.)
    p_m2 *= low_filter

    return p_m2
