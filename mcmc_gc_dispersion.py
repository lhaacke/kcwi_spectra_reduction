import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import emcee
import corner

from scipy.optimize import minimize

################# don't use, very outdated ########################
def log_likelihood(theta, v_obs, v_err, fix_v=False):
    '''
    theta: tuple of free parameters in the mcmc
    v_obs: array of observed recessional velocities of GCs
    v_err: corresponding error of v_obs
    '''
    if fix_v:
        v_sys, sig = theta
        pre = np.log(1/(np.sqrt(2*sc.pi*(sig**2 + v_err**2))))
        chisq = 0.5*(v_obs - 2280)**2/(sig**2 + v_err**2)
    else:
        v_sys, sig = theta
        # pre = np.log(1/(np.sqrt(2*sc.pi * (sig**2 + v_err**2)) * np.sqrt(sig**2 + v_err**2))) # currently running with Jeffrey's prior
        pre = np.log(1/(np.sqrt(2*sc.pi * (sig**2 + v_err**2))))
        chisq = 0.5*(v_obs - v_sys)**2/(sig**2 + v_err**2)

    return np.sum(pre - chisq)


def log_likelihood_1d(theta, v_obs, v_err, v_gal=1405):
    '''
    theta: tuple of free parameters in the mcmc
    v_obs: array of observed recessional velocities of GCs
    v_err: corresponding error of v_obs
    '''
    sig = theta
    pre = np.log(1/(np.sqrt(2*sc.pi*(sig**2 + v_err**2))))
    chisq = 0.5*(v_obs - v_gal)**2/(sig**2 + v_err**2)
    return np.sum(pre - chisq)


def log_prior(theta, prior_range_v, prior_range_sig):
    '''
    theta: tuple of free parameters in the mcmc
    '''
    v, dsig = theta
    if (prior_range_v[0] < v < prior_range_v[1]) and (prior_range_sig[0] < dsig < prior_range_sig[1]):
        return 0.
    return -np.inf


def log_prior_1d(theta, prior_range_sig):
    '''
    theta: tuple of free parameters in the mcmc
    '''
    dsig = theta
    if (prior_range_sig[0] < dsig < prior_range_sig[1]):
        return 0.
    return -np.inf


def log_post(theta, v_obs, v_err, prior_range_v, prior_range_sig):
    '''
    theta: tuple of free parameters in the mcmc
    prior ranges: tuples of (min, max)
    '''
    lp = log_prior(theta, prior_range_v, prior_range_sig)
    if not np.isfinite(lp):
        return -np.inf
    return  lp + log_likelihood(theta, v_obs, v_err)


def log_post_1d(theta, v_obs, v_err, prior_range_sig):
    '''
    theta: tuple of free parameters in the mcmc
    prior ranges: tuples of (min, max)
    '''
    lp = log_prior_1d(theta, prior_range_sig)
    if not np.isfinite(lp):
        return -np.inf
    return  lp + log_likelihood_1d(theta, v_obs, v_err)


def sigma_rmse(v, v_gal):
    N = len(v)
    v_minus_v_gal = v - v_gal
    v_minus_v_gal_sq = np.full(len(v), np.nan)
    for el in enumerate(v_minus_v_gal):
        v_minus_v_gal_sq[el[0]] = el[1]**2
    return np.sqrt(np.nansum(v_minus_v_gal_sq)/N)


def run_mcmc(v_obs, v_err, prior_range_v, prior_range_sig, sig_exp, nwalkers=100, ndim=2):
    '''
    run the mcmc
    '''
    # INITIAL VALUES
    v_exp = np.mean(v_obs)
    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([v_exp, sig_exp]) + 10 * np.random.randn(nwalkers, ndim)

    # RUN
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, args=[v_obs, v_err, prior_range_v, prior_range_sig])
    pos, prob, state = sampler.run_mcmc(initial, 1000, progress=True)
    sampler.reset() # reset sampler before running
    sampler.run_mcmc(pos, nsteps=20000, progress=True) # run actual mcmc

    # SIGMA
    flat_samples = sampler.get_chain(discard=0, thin=15, flat=True)
    vals_16, vals_50, vals_84 = np.nanpercentile(flat_samples, 16, axis=0), np.nanpercentile(flat_samples, 50, axis=0), np.nanpercentile(flat_samples, 16, axis=0)
    v_16, sig_16 = vals_16[0], vals_16[1]
    v_50, sig_50 = vals_50[0], vals_50[1]
    v_84, sig_84 = vals_84[0], vals_84[1]

    return ((v_16, v_50, v_84), (sig_16, sig_50, sig_84))
