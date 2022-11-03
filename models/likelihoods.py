import numpy as np
import jax.numpy as jnp
from jax import vmap
import numpyro
import numpyro.distributions as dist
import pdb

from models import mass_models
from models import spin_models
from models import redshift_models

def betaPlusIsotropic(sampleDict,injectionDict):
    """
    Implementation of mixture spin tilt model that consists of:
        - a tilt model that combines an isotropic and ~aligned component,
            where the aligned component is a Beta distribution with a>=1 and 1<=b<=a
        - an iid spin magnitude distribution that is modeled as a Beta distribution
        - a PL+Peak mass model with parameters fixed to the GWTC-3 medians
        - a powerlaw mass ratio component parameterized by Beta_q
        - a powerlaw redshift model with parameters fixed to the GWTC-3 medians

    Parameters:
    -----------
    sampleDict : dict
        Precomputed dictionary with posterior sampels for each event in the catalog
    injectionDict : dict
        Precomputed dictionary with successfully recovered injections
    mMin : float
        Minimum black hole mass
    """

    ### Sample our hyperparameters
    ### NOTE: primary mass and redshift hyperparameters are fixed to GWTC-3 median values
    # beta_q: powerlaw index on the conditional secondary mass distribution p(m2|m1)
    # BF: branching fraction between isotropic and aligned components of spin tilt distribution
    # tiltBetaA: `a` shape parameter of Beta distribution for costilt aligned component
    # tiltBetaB: `b` shape parameter of Beta distribution for costilt aligned component
    # chiBetaA: `a` shape parameter of Beta distribution for iid spin magnitude distribution
    # chiBetaB: `b` shape parameter of Beta distribution for iid spin magnitude distribution
    log_R20 = numpyro.sample("log_R20", dist.Uniform(-2,1))
    R20 = numpyro.deterministic("R20", 10.**log_R20)
    beta_q = numpyro.sample("beta_q", dist.Normal(0,3))
    BF = numpyro.sample("BF", dist.Uniform(0,1))   # NOTE: this is what we will constrain
    log_tiltBetaA = numpyro.sample("log_tiltBetaA", dist.Normal(0,1))
    log_tiltBetaB = numpyro.sample("log_tiltBetaB", dist.Normal(0,1))
    log_chiBetaA = numpyro.sample("log_chiBetaA", dist.Normal(0,1))
    log_chiBetaB = numpyro.sample("log_chiBetaB", dist.Normal(0,1))

    """alpha = numpyro.sample("alpha",dist.Normal(-2,3))
    mMin = numpyro.sample("mMin",dist.Uniform(3,10))
    mMax = numpyro.sample("mMax",dist.Uniform(50,100))
    lambdaPeak = numpyro.sample("lambdaPeak",dist.Uniform(0,1))
    meanPeak = numpyro.sample("meanPeak",dist.Uniform(20,50))
    sigmaPeak = numpyro.sample("sigmaPeak",dist.Uniform(2,10))
    dmMinTaper = numpyro.sample("dmMinTaper",dist.Uniform(1,10))
    dmMaxTaper = numpyro.sample("dmMaxTaper",dist.Uniform(1,10))
    kappa = numpyro.sample("kappa",dist.Normal(3.,2))"""

    # Fixed parameters
    alpha=-3.51
    mMin=5.0
    mMax=88.21
    lambdaPeak=0.033
    meanPeak=33.61
    sigmaPeak=4.72
    dmMinTaper=4.88
    dmMaxTaper=5.0   # FIXME: don't know if this is actually in GWTC-3?
    kappa=2.7

    # Normalization
    p_m1_norm = mass_models.PowerlawPlusPeakPrimary(20.,alpha,mMin,mMax,lambdaPeak,meanPeak,sigmaPeak,dmMinTaper,dmMaxTaper)
    p_z_norm = (1.+0.2)**kappa

    # Read out found injections for detection efficiency factor in hyperposterior
    # NOTE: `pop_reweight` is the inverse of the draw weights for each event
    m1_det = injectionDict['m1']
    m2_det = injectionDict['m2']
    chi1_det = injectionDict['a1']
    chi2_det = injectionDict['a2']
    cost1_det = injectionDict['cost1']
    cost2_det = injectionDict['cost2']
    z_det = injectionDict['z']
    dVdz_det = injectionDict['dVdz']
    p_draw = injectionDict['p_draw_m1m2z'] * injectionDict['p_draw_a1a2cost1cost2']
    #pop_reweight = injectionDict['pop_reweight']   # FIXME: where is this used?  # defined in getData, used when we fix mass parameters.

    # Compute proposed population weights from the injections
    p_m1_det = mass_models.PowerlawPlusPeakPrimary(m1_det,alpha,mMin,mMax,lambdaPeak,meanPeak,sigmaPeak,dmMinTaper,dmMaxTaper)
    p_m2_det = mass_models.PowerlawSecondary(m1_det,m2_det,mMin,dmMinTaper)
    p_chi_det = spin_models.BetaSpinMags(chi1_det,chi2_det,10**log_chiBetaA,10**log_chiBetaB)
    p_tilt_det = spin_models.IsotropicPlusBetaSpinTilts(cost1_det,cost2_det,BF,10**log_tiltBetaA,10**log_tiltBetaB)
    p_z_det = redshift_models.PowerlawRedshift(z_det,dVdz_det,kappa)
    R_pop_det = R20*p_m1_det*p_m2_det*p_z_det*p_chi_det*p_tilt_det

    # Form ratio of injection weights over draw weights
    Tobs = 2.   # years, timespan that injections were put into data
    inj_weights = R_pop_det*Tobs / p_draw

    # As a fit diagnostic, compute effective number of injections
    nEff_inj = jnp.sum(inj_weights)**2 / jnp.sum(inj_weights**2)
    nObs = 1.0*len(sampleDict)
    numpyro.deterministic("nEff_inj_per_event",nEff_inj/nObs)

    # Compute net detection efficiency and add to log-likelihood
    Nexp = jnp.sum(inj_weights)/injectionDict['nTrials']
    numpyro.factor("rate",-Nexp)


    ### The following function defined the per-event log-likelihood
    # m1_sample: primary mass posterior sample
    # m2_sample: secondary mass posterior sample
    # z_sample: redshift posterior sample
    # dVdz_sample: differential comoving volume at each redshift sample location
    # chi1_sample: primary spin magnitude sample
    # chi2_sample: secondary spin magnitude sample
    # cost1_sample: cosine of primary tilt angle sample
    # cost2_sample: cosine of secondary tilt angle sample
    # priors: PE priors on each sample to be divided out (only needed for redshift prior currently)
    def logp(m1_sample,m2_sample,z_sample,dVdz_sample,chi1_sample,chi2_sample,cost1_sample,cost2_sample,priors):
        # Compute proposed population weights
        p_m1 = mass_models.PowerlawPlusPeakPrimary(m1_sample,alpha,mMin,mMax,lambdaPeak,meanPeak,sigmaPeak,dmMinTaper,dmMaxTaper)
        p_m2 = mass_models.PowerlawSecondary(m1_sample,m2_sample,mMin,dmMinTaper)
        p_z = redshift_models.PowerlawRedshift(z_sample,dVdz_sample,kappa)
        p_chi = spin_models.BetaSpinMags(chi1_sample,chi2_sample,10**log_chiBetaA,10**log_chiBetaB)
        p_tilt = spin_models.IsotropicPlusBetaSpinTilts(cost1_sample,cost2_sample,BF,10**log_tiltBetaA,10**log_tiltBetaB)
        R_pop = R20*p_m1*p_m2*p_z*p_chi*p_tilt

        mc_weights = R_pop / priors

        # Compute effective number of samples and return log-likelihood
        n_eff = jnp.sum(mc_weights)**2 / jnp.sum(mc_weights**2)
        return jnp.log(jnp.mean(mc_weights)),n_eff


    # Map log-likelihood over each event in our catalog using vmap
    log_ps,n_effs = vmap(logp)(
                        jnp.array([sampleDict[k]['m1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['m2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['dVc_dz'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['a2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost1'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['cost2'] for k in sampleDict]),
                        jnp.array([sampleDict[k]['z_prior'] for k in sampleDict]))

    # As a diagnostic, save minimum number of effective samples across all events
    numpyro.deterministic('min_log_neff',jnp.min(jnp.log10(n_effs)))

    # Tally log-likelihoods across our catalog
    numpyro.factor("logp",jnp.sum(log_ps))
