import numpy as np
import os
dirname = os.path.dirname(__file__)

def reweighting_function_archi(m1,m2,a1,a2,cost1,cost2,z,dVdz):

    kappa = 1.
    alpha = -3.7
    mu_m1 = 35.
    sig_m1 = 3.
    f_peak = 0.005
    mMax = 90.
    mMin = 9.
    dmMax = 10.
    dmMin = 5.
    bq = 0.

    p_m1 = massModel(m1,alpha,mu_m1,sig_m1,f_peak,mMax,mMin,dmMax,dmMin)
    p_m2 = m2**bq/(m1**(1.+bq) - 2.**(1.+bq))

    p_z = dVdz*(1.+z)**(kappa-1.)
    z_grid = np.linspace(0,2.3,1000)
    dVdz_grid = 4.*np.pi*Planck15.differential_comoving_volume(z_grid).to(u.Gpc**3*u.sr**(-1)).value
    norm = np.trapz(dVdz_grid*(1.+z_grid)**(kappa-1.))
    p_z /= norm

    return p_m1*p_m2*p_z

def reweighting_function_arlnm1(m1,m2,a1,a2,cost1,cost2,z,dVdz):

    """
    Function that computes weights under a fixed population similar to that preferred by
    the ln(m1) AR1 analysis.
    """

    kappa = 3.
    bq = 1.5
    alpha = -2.5
    m0 = 10.
    f_pl = 0.3
    f_peak_in_peak1 = 0.9

    p_m1 = np.zeros_like(m1)
    p_m1 += (1.-f_pl)*f_peak_in_peak1*np.exp(-(m1-10.)**2/(2.*0.8**2))/np.sqrt(2.*np.pi*0.8**2)
    p_m1 += (1.-f_pl)*(1.-f_peak_in_peak1)*np.exp(-(m1-35.)**2/(2.*2.**2.))/np.sqrt(2.*np.pi*2.**2)
    p_m1_pl = np.ones(m1.size)
    p_m1_pl[m1<m0] = 1.
    p_m1_pl[m1>=m0] = (m1[m1>=m0]/m0)**alpha
    p_m1_pl /= (alpha*m0 - 2. - alpha*2.)/(1.+alpha)
    p_m1 += f_pl*p_m1_pl
    p_m1[m1<2.] = 0.
    p_m1[m1>100.] = 0.

    p_m2 = m2**bq/(m1**(1.+bq)-2.**(1.+bq))
    p_m2[m2<2.] = 0.
    p_m2[m2>100.] = 0.

    p_z = dVdz*(1.+z)**(kappa-1.)
    z_grid = np.linspace(0,2.3,1000)
    dVdz_grid = 4.*np.pi*Planck15.differential_comoving_volume(z_grid).to(u.Gpc**3*u.sr**(-1)).value
    norm = np.trapz(dVdz_grid*(1.+z_grid)**(kappa-1.))
    p_z /= norm

    p_a1 = 1.
    p_a2 = 1.
    p_cost1 = 1./2.
    p_cost2 = 1./2.

    return (p_m1*p_m2*p_a1*p_a2*p_cost1*p_cost2*p_z)

def reweighting_function_arlnm1_q(m1,m2,a1,a2,cost1,cost2,z,dVdz):

    """
    Function that computes weights under a fixed population similar to that preferred by
    the ln(m1) AR1 analysis.
    """

    kappa = 3.
    q_std = 0.5
    alpha = -2.5
    m0 = 10.
    f_pl = 0.3
    f_peak_in_peak1 = 0.9

    p_m1 = np.zeros_like(m1)
    p_m1 += (1.-f_pl)*f_peak_in_peak1*np.exp(-(m1-10.)**2/(2.*0.8**2))/np.sqrt(2.*np.pi*0.8**2)
    p_m1 += (1.-f_pl)*(1.-f_peak_in_peak1)*np.exp(-(m1-35.)**2/(2.*2.**2.))/np.sqrt(2.*np.pi*2.**2)
    p_m1_pl = np.ones(m1.size)
    p_m1_pl[m1<m0] = 1.
    p_m1_pl[m1>=m0] = (m1[m1>=m0]/m0)**alpha
    p_m1_pl /= (alpha*m0 - 2. - alpha*2.)/(1.+alpha)
    p_m1 += f_pl*p_m1_pl
    p_m1[m1<2.] = 0.
    p_m1[m1>100.] = 0.

    q = m2/m1
    p_q = np.exp(-(q-1)**2/(2.*q_std**2))*2./np.sqrt(2.*np.pi*q_std**2)
    p_m2 = p_q/m1

    p_z = dVdz*(1.+z)**(kappa-1.)
    z_grid = np.linspace(0,2.3,1000)
    dVdz_grid = 4.*np.pi*Planck15.differential_comoving_volume(z_grid).to(u.Gpc**3*u.sr**(-1)).value
    norm = np.trapz(dVdz_grid*(1.+z_grid)**(kappa-1.))
    p_z /= norm

    p_a1 = 1.
    p_a2 = 1.
    p_cost1 = 1./2.
    p_cost2 = 1./2.

    return (p_m1*p_m2*p_a1*p_a2*p_cost1*p_cost2*p_z)

def getInjections(sample_limit=10000,spin='component',reweight=False,weighting_function=reweighting_function_arlnm1):

    """
    Function to load and preprocess found injections for use in numpyro likelihood functions.
    Returns
    -------
    injectionDict : dict
        Dictionary containing found injections
    """

    injectionFile = "input/injectionDict_FAR_1_in_1.pickle"
    injectionDict = np.load(injectionFile,allow_pickle=True)

    for key in injectionDict:
        if key!='nTrials':
            injectionDict[key] = np.array(injectionDict[key])

    if reweight:

        print("REWEIGHTING")

        m1 = injectionDict['m1']
        m2 = injectionDict['m2']
        z = injectionDict['z']
        dVdz = injectionDict['dVdz']

        if spin=='component':
            a1 = injectionDict['a1']
            a2 = injectionDict['a2']
            cost1 = injectionDict['cost1']
            cost2 = injectionDict['cost2']
            pDraw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_a1a2cost1cost2']
            p_new = weighting_function(m1,m2,a1,a2,cost1,cost2,z,dVdz)

        elif spin=='effective':
            Xeff = injectionDict['Xeff']
            Xp = injectionDict['Xp']
            pDraw = injectionDict['p_draw_m1m2z']*injectionDict['p_draw_chiEff_chiP']
            p_new = weighting_function(m1,m2,Xeff,Xp,z,dVdz)

        downselection_draw_weights = (p_new/pDraw)/injectionDict['nTrials']
        nEff_downselection = np.sum(downselection_draw_weights)**2/np.sum(downselection_draw_weights**2)

        new_pDraw = (p_new)/np.sum(downselection_draw_weights)
        injectionDict['downselection_Neff'] = nEff_downselection
        injectionDict['nTrials'] = sample_limit
        injectionDict['p_draw_m1m2z'] = new_pDraw

        if spin=='component':
            injectionDict['p_draw_a1a2cost1cost2'] = np.ones(new_pDraw.size)
        elif spin=='effective':
            injectionDict['p_draw_chiEff_chiP'] = np.ones(new_pDraw.size)

        print(nEff_downselection)
        inds_to_keep = np.random.choice(np.arange(new_pDraw.size),size=sample_limit,replace=True,p=downselection_draw_weights/np.sum(downselection_draw_weights))
        for key in injectionDict:
            if key!='nTrials' and key!='downselection_Neff':
                injectionDict[key] = injectionDict[key][inds_to_keep]

    return injectionDict

def getSamples(sample_limit=1000,bbh_only=True,spin='component',reweight=False,weighting_function=reweighting_function_arlnm1):

    """
    Function to load and preprocess BBH posterior samples for use in numpyro likelihood functions.

    Parameters
    ----------
    sample_limit : int or None
        If specified, will randomly downselect posterior samples, returning N=sample_limit samples per event (default None)
    bbh_only : bool
        If true, will exclude samples for BNS, NSBH, and mass-gap events (default True)
    Returns
    -------
    sampleDict : dict
        Dictionary containing posterior samples
    """

    # Dicts with samples:
    sampleDict = np.load("input/sampleDict_FAR_1_in_1_yr.pickle",allow_pickle=True)

    non_bbh = ['GW170817','S190425z','S190426c','S190814bv','S190917u','S200105ae','S200115j']
    if bbh_only:
        for event in non_bbh:
            print("Removing ",event)
            sampleDict.pop(event)

    mMin = 0.
    for event in sampleDict:

        if reweight:

            m1 = np.array(sampleDict[event]['m1'])
            m2 = np.array(sampleDict[event]['m2'])
            z = np.array(sampleDict[event]['z'])
            dVdz = np.array(sampleDict[event]['dVc_dz'])

            if spin=='component':

                a1 = np.array(sampleDict[event]['a1'])
                a2 = np.array(sampleDict[event]['a2'])
                cost1 = np.array(sampleDict[event]['cost1'])
                cost2 = np.array(sampleDict[event]['cost2'])
                prior = np.array(sampleDict[event]['z_prior'])
                p_new = weighting_function(m1,m2,a1,a2,cost1,cost2,z,dVdz)
                sampleDict[event]['z_prior'] = p_new

            elif spin=='effective':

                Xeff = np.array(sampleDict[event]['Xeff'])
                Xp = np.array(sampleDict[event]['Xp'])
                prior = np.array(sampleDict[event]['z_prior'])*np.array(sampleDict[event]['joint_priors'])
                p_new = weighting_function(m1,m2,a1,a2,cost1,cost2,z,dVdz)
                sampleDict[event]['z_prior'] = p_new
                sampleDict[event]['joint_priors'] = np.ones(Xeff.size)

            draw_weights = p_new/prior
            draw_weights /= np.sum(draw_weights)
            neff = np.sum(draw_weights)**2/np.sum(draw_weights**2)

            if neff<2.*sample_limit:
                print(event,neff)

        else:

            draw_weights = np.ones(sampleDict[event]['m1'].size)/sampleDict[event]['m1'].size

        sampleDict[event]['downselection_Neff'] = np.sum(draw_weights)**2/np.sum(draw_weights**2)

        inds_to_keep = np.random.choice(np.arange(sampleDict[event]['m1'].size),size=sample_limit,replace=True,p=draw_weights)
        for key in sampleDict[event].keys():
            if key!='downselection_Neff':
                sampleDict[event][key] = sampleDict[event][key][inds_to_keep]

        new_mMin = np.min(sampleDict[event]['m1'])
        if new_mMin>mMin:
            mMin = new_mMin

    #print("!!!!!!",mMin)

    return sampleDict




def getInjections_fixedm1z(mMin=5.0,
    mMax=88.2,
    delta_m=4.9,
    kappa=2.7,
    alpha=-3.5,
    mu_m=33.6,
    sig_m=4.7,
    f_peak = 0.03):

    """
    Function to load and preprocess found injections for use in numpyro likelihood functions.
    Arguments specify that primary mass and redshift distribution to which injections will be reweighted.
    Default arguments correspond to the 1D median results from the LVK O3b population analysis.

    Parameters
    ----------
    mMin : float
        Minimum black hole mass (Optional; default 5.0)
    mMax : float
        Maximum black hole mass (Optional; default 88.2)
    delta_m : float
        Smoothing length over which mass spectrum turns on at low masses (Optional; default 4.9)
    kappa : float
        Power-law index governing evolution of the BBH merger rate with `1+z` (Optional; default 2.7)
    alpha : float
        Power-law index governing primary mass distribution (Optional; default -3.5)
    mu_m : float
        Location of Gaussian "peak" in the primary mass distribution (Optional; default 33.6)
    sig_m : float
        Width of Gaussian "peak" in the primary mass distribution (Optional; 4.7)
    f_peak : float
        Fraction of events comprising the Gaussian "peak" (Optional; default 0.03)

    Returns
    -------
    injectionDict : dict
        Dictionary containing found injections, with factors needed to reweight to desired mass/redshift distribution.
    """

    injectionFile = os.path.join(dirname,"./input/injectionDict_FAR_1_in_1.pickle")
    injectionDict = np.load(injectionFile,allow_pickle=True)

    m1_det = np.array(injectionDict['m1'])
    m2_det = np.array(injectionDict['m2'])
    chi1_det = injectionDict['a1']
    chi2_det = injectionDict['a2']
    cost1_det = injectionDict['cost1']
    cost2_det = injectionDict['cost2']
    z_det = np.array(injectionDict['z'])
    dVdz_det = np.array(injectionDict['dVdz'])
    inj_weights = 1./(np.array(injectionDict['p_draw_a1a2cost1cost2'])*np.array(injectionDict['p_draw_m1m2z']))

    # fix mass model and get reweighting parameters
    p_m1_det_pl = (1.+alpha)*m1_det**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
    p_m1_det_peak = np.exp(-(m1_det-mu_m)**2./(2.*np.pi*sig_m**2))/np.sqrt(2.*np.pi*sig_m**2.)
    p_m1_det = f_peak*p_m1_det_peak + (1.-f_peak)*p_m1_det_pl
    p_m1_det[m1_det>mMax] = 0.
    p_m1_det[m1_det<mMin] = 0.

    smoothing = np.ones_like(m1_det)
    to_smooth = (m1_det>mMin)*(m1_det<delta_m)
    smoothing[to_smooth] = (np.exp(delta_m/m1_det[to_smooth] + delta_m/(m1_det[to_smooth]-delta_m)) + 1)**(-1.)
    p_m1_det *= smoothing

    p_z_det = dVdz_det*(1.+z_det)**(kappa-1)

    pop_reweight = p_m1_det*p_z_det*inj_weights
    injectionDict['pop_reweight'] = pop_reweight

    return injectionDict

def getSamples_fixedm1z(mMin=5.,
    mMax=88.2,
    delta_m=4.9,
    kappa=2.7,
    alpha=-3.5,
    mu_m=33.6,
    sig_m=4.7,
    f_peak = 0.03):

    """
    Function to load and preprocess BBH posterior samples for use in numpyro likelihood functions.
    Arguments specify that primary mass and redshift distribution to which injections will be reweighted.
    Default arguments correspond to the 1D median results from the LVK O3b population analysis.

    Parameters
    ----------
    mMin : float
        Minimum black hole mass (Optional; default 5.0)
    mMax : float
        Maximum black hole mass (Optional; default 88.2)
    delta_m : float
        Smoothing length over which mass spectrum turns on at low masses (Optional; default 4.9)
    kappa : float
        Power-law index governing evolution of the BBH merger rate with `1+z` (Optional; default 2.7)
    alpha : float
        Power-law index governing primary mass distribution (Optional; default -3.5)
    mu_m : float
        Location of Gaussian "peak" in the primary mass distribution (Optional; default 33.6)
    sig_m : float
        Width of Gaussian "peak" in the primary mass distribution (Optional; 4.7)
    f_peak : float
        Fraction of events comprising the Gaussian "peak" (Optional; default 0.03)

    Returns
    -------
    sampleDict : dict
        Dictionary containing posterior samples, with factors needed to reweight to desired mass/redshift distribution.
    """

    # Dicts with samples:
    sampleDictPath = os.path.join(dirname,"./input/sampleDict_FAR_1_in_1_yr.pickle")
    sampleDict = np.load(sampleDictPath,allow_pickle=True)

    # Non-BBHs
    non_BBHs = ['GW170817','S190425z','S190426c','S190814bv','S190917u','S200105ae','S200115j']
    for event in non_BBHs:
        sampleDict.pop(event)

    for event in sampleDict:
        print(event,len(sampleDict[event]['m1']))

    nEvents = len(sampleDict)
    for event in sampleDict:

        m1 = np.array(sampleDict[event]['m1'])
        m2 = np.array(sampleDict[event]['m2'])
        Xeff = np.array(sampleDict[event]['Xeff'])
        z = np.array(sampleDict[event]['z'])
        dVdz = np.array(sampleDict[event]['dVc_dz'])

        p_m1_pl = (1.+alpha)*m1**alpha/(mMax**(1.+alpha) - mMin**(1.+alpha))
        p_m1_peak = np.exp(-(m1-mu_m)**2./(2.*np.pi*sig_m**2))/np.sqrt(2.*np.pi*sig_m**2.)
        p_m1 = f_peak*p_m1_peak + (1.-f_peak)*p_m1_pl
        p_m1[m1>mMax] = 0.
        p_m1[m1<mMin] = 0.

        smoothing = np.ones_like(m1)
        to_smooth = (m1>mMin)*(m1<delta_m)
        smoothing[to_smooth] = (np.exp(delta_m/m1[to_smooth] + delta_m/(m1[to_smooth]-delta_m)) + 1)**(-1.)
        p_m1 *= smoothing

        p_z = dVdz*(1.+z)**(kappa-1)

        sampleDict[event]['weights_over_priors'] = p_m1*p_z/(sampleDict[event]['Xeff_priors']*sampleDict[event]['z_prior'])
        inds_to_keep = np.random.choice(np.arange(m1.size),size=4000,replace=True)
        for key in sampleDict[event].keys():
            sampleDict[event][key] = sampleDict[event][key][inds_to_keep]

    print("Number of events: {:d}".format(len(sampleDict)))
    return sampleDict

if __name__=="__main__":
    getSamples()
    getInjections()
