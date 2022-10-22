import numpy as np
import jax as jnp
from models import utilities

###################
### SPIN MODELS ###
###################

def IsotropicPlusBetaSpinTilts(cost1,cost2,BF,tiltBetaA,tiltBetaB):
    """
    Spin tilt model that considers two components:
    - an isotropic component with branching fraction (1-BF)
    - an 'aligned' component defined as a beta distribution with shape parameters
        a >= 1 and 1 <= b <= a (set in the priors) such that the isotropic
        component prefers aligned spins, with branching fraction BF

    Parameters:
    -----------
    cost1 : float or array
        primary spin cos tilt sample(s)
    cost2 : float or array
        secondary spin cos tilt sample(s)
    BF : float
        branching fraction for isotropic component (BF) and aligned component (1-BF)
    tiltBetaA : float
        Beta distribution shape parameter `a` for aligned component
    tiltBetaB : float
        Beta distribution shape parameter `b` for aligned component

    Returns:
    --------
    p_cost1_cost2 : jax.numpy.array
        Unnormalized array of probability densities
    """

    # isotropic component
    p_iso = BF / 4.0   # divided by 4 since this is 0.5 for each component

    # aligned component
    p_aligned1 = utilities.betaDistribution(cost1, tiltBetaA, tiltBetaB, betaMin=-1, betaMax=1)
    p_aligned2 = utilities.betaDistribution(cost2, tiltBetaA, tiltBetaB, betaMin=-1, betaMax=1)
    p_aligned = (1-BF) * p_aligned1 * p_aligned2

    p_cost1_cost2 = p_iso + p_aligned
    return p_cost1_cost2


def BetaSpinMags(chi1,chi2,chiBetaA,chiBetaB):
    """
    Spin magnitudes are drawn from a Beta distribution and are iid

    Parameters:
    -----------
    chi1 : float or array
        primary spin magnitude sample(s)
    chi2 : float or array
        secondary spin magnitude sample(s)
    chiBetaA : float
        Beta distribution shape parameter `a`
    chiBetaB : float
        Beta distribution shape parameter `b`

    Returns:
    --------
    p_chi1_chi2 : jax.numpy.array
        Unnormalized array of probability densities
    """

    # iid beta distribution for spins
    p_chi1 = utilities.betaDistribution(chi1, chiBetaA, chiBetaB, betaMin=0, betaMax=1)
    p_chi2 = utilities.betaDistribution(chi2, chiBetaA, chiBetaB, betaMin=0, betaMax=1)

    p_chi1_chi2 = p_chi1 * p_chi2
    return p_chi1_chi2



