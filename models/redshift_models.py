import numpy as np
import jax.numpy as jnp

#######################
### REDSHIFT MODELS ###
#######################

def PowerlawRedshift(z, dVdz, kappa=2.7):

    """
    Redshift model that follows a powerlaw in (1+z)

    Parameters:
    -----------
    z : float or array
        redshift sample(s)
    dVdz : float or array
        d(comoving volume)/dz sample(s)

    Returns:
    --------
    p_redshifts : jax.numpy.array
        evaluations at the input samples
    """

    p_redshifts = dVdz*jnp.power(1.+z,kappa-1.)

    return p_redshifts
