import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.optimize import minimize
from astropy.table import Table
from astropy.io import fits

############################# don't use very outdated ##############################
def log_likelihood(theta, v_obs, v_err):
    '''
    theta: tuple of free parameters in the mcmc
    v_obs: array of observed recessional velocities of GCs
    v_err: corresponding error of v_obs
    '''
    v_sys, sig = theta
    pre = np.log(1/(np.sqrt(2*sc.pi*(sig**2 + v_err**2))))
    chisq = 0.5*(v_obs - v_sys)**2/(sig**2 + v_err**2)

    return np.sum(pre - chisq)


def log_prior(theta, prior_range_v, prior_range_sig):
    '''
    theta: tuple of free parameters in the mcmc
    '''
    v, dsig = theta
    if (prior_range_v[0] < v < prior_range_v[1]) and (prior_range_sig[0] < dsig < prior_range_sig[1]):
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


def jackknife_sensitivity_test_from_table(GCs_finalised):
    '''
    run test on sensitivity of mcmc to outliers
    '''
    ##### run mcmc in magnitude bins, jackknife for whole sample, KCWI + MUSE
    v_by_mag_jack_all = []
    v_min_by_mag_jack_all = []
    v_max_by_mag_jack_all = []
    sig_by_mag_jack_all = []
    sig_min_by_mag_jack_all = []
    sig_max_by_mag_jack_all = []

    nwalkers = 100
    ndim = 2
    sig_exp = 17
    prior_range_sig = (0., 100.)

    for i in range(len(GCs_finalised) - 1):
        GCs_mag_sorted_jack = GCs_finalised.copy()
        GCs_mag_sorted_jack.remove_rows([-1]) # remove the mueller candidate
        GCs_mag_sorted_jack.sort(['HST_f606w_total_mag'])
        GCs_mag_sorted_jack.remove_rows([i])

        # make bins in which to run MCMC
        v_mag_list_jack = [GCs_mag_sorted_jack['v_combined'][:10].value,
                    GCs_mag_sorted_jack['v_combined'][:12].value,
                    GCs_mag_sorted_jack['v_combined'][:14].value,
                    GCs_mag_sorted_jack['v_combined'][:16].value,
                    GCs_mag_sorted_jack['v_combined'][:18].value,
                    GCs_mag_sorted_jack['v_combined'][:].value]
        dv_mag_list_jack = [GCs_mag_sorted_jack['dv_combined'][:10].value,
                    GCs_mag_sorted_jack['dv_combined'][:12].value,
                    GCs_mag_sorted_jack['dv_combined'][:14].value,
                    GCs_mag_sorted_jack['dv_combined'][:16].value,
                    GCs_mag_sorted_jack['dv_combined'][:18].value,
                    GCs_mag_sorted_jack['dv_combined'][:].value]

        # make lists in which to store individual results
        v_by_mag_jack_in = []
        v_min_by_mag_jack_in = []
        v_max_by_mag_jack_in = []
        sig_by_mag_jack_in = []
        sig_min_by_mag_jack_in = []
        sig_max_by_mag_jack_in = []

        for v_obs, v_err in zip(v_mag_list_jack, dv_mag_list_jack):
            v_exp = np.mean(v_obs)
            prior_range_v = (np.min(v_obs), np.max(v_obs))
            # get initial values
            np.random.seed(42)
            nll = lambda *args: -log_likelihood(*args)
            initial = np.array([v_exp, sig_exp]) + 10 * np.random.randn(nwalkers, ndim)
            # run burn in 
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, args=[v_obs, v_err, prior_range_v, prior_range_sig])
            pos, prob, state = sampler.run_mcmc(initial, 1000, progress=True)
            # run chain
            sampler.reset()
            sampler.run_mcmc(pos, nsteps=20000, progress=True)
        
            flat_samples = sampler.get_chain(discard=0, thin=15, flat=True)
            sig_min_by_mag_jack_in.append(np.percentile(flat_samples, 16, axis=0)[1])
            sig_by_mag_jack_in.append(np.percentile(flat_samples, 50, axis=0)[1])
            sig_max_by_mag_jack_in.append(np.percentile(flat_samples, 84, axis=0)[1])
            v_min_by_mag_jack_in.append(np.percentile(flat_samples, 16, axis=0)[0])
            v_by_mag_jack_in.append(np.percentile(flat_samples, 50, axis=0)[0])
            v_max_by_mag_jack_in.append(np.percentile(flat_samples, 84, axis=0)[0])

        print(sig_min_by_mag_jack_in)
        
        sig_min_by_mag_jack_all.append(sig_min_by_mag_jack_in)
        sig_by_mag_jack_all.append(sig_by_mag_jack_in)
        sig_max_by_mag_jack_all.append(sig_max_by_mag_jack_in)
        v_min_by_mag_jack_all.append(v_min_by_mag_jack_in)
        v_by_mag_jack_all.append(v_by_mag_jack_in)
        v_max_by_mag_jack_all.append(v_max_by_mag_jack_in)

        print(sig_min_by_mag_jack_all)

        # sig_min_by_mag_jack_in.clear()
        # sig_by_mag_jack_in.clear()
        # sig_max_by_mag_jack_in.clear()
        # v_min_by_mag_jack_in.clear()
        # v_by_mag_jack_in.clear()
        # v_max_by_mag_jack_in.clear()

        # print(sig_min_by_mag_jack_all)

    return sig_min_by_mag_jack_all, sig_by_mag_jack_all, sig_max_by_mag_jack_all, v_min_by_mag_jack_all, v_by_mag_jack_all, v_max_by_mag_jack_all

def jackknife_sensitivity(v_obs, v_err):
    '''
    run test on sensitivity of mcmc to outliers
    '''
    ##### run mcmc in magnitude bins, jackknife for whole sample, KCWI + MUSE
    v_jack_all = []
    v_min_jack_all = []
    v_max_jack_all = []
    sig_jack_all = []
    sig_min_jack_all = []
    sig_max_jack_all = []

    nwalkers = 100
    ndim = 2
    sig_exp = np.std(v_obs)
    prior_range_sig = (0., 100.)

    # loop through v_obs, for each run without ith element
    for i in range(len(v_obs)):
        print(i)
        # create sample of observed vs and errors without i object
        # select all elements of v_obs that are not in position with index i
        print(v_obs_arr)
        v_obs = np.delete(v_obs_arr, i)
        v_err = np.delete(v_err_arr, i)
        print(v_obs_arr)

        v_exp = np.mean(v_obs)
        prior_range_v = (np.min(v_obs), np.max(v_obs))
        # get initial values
        np.random.seed(42)
        nll = lambda *args: -log_likelihood(*args)
        initial = np.array([v_exp, sig_exp]) + 10 * np.random.randn(nwalkers, ndim)
        # run burn in 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, args=[v_obs, v_err, prior_range_v, prior_range_sig])
        pos, prob, state = sampler.run_mcmc(initial, 1000, progress=True)
        # run chain
        sampler.reset()
        sampler.run_mcmc(pos, nsteps=20000, progress=True)
    
        flat_samples = sampler.get_chain(discard=0, thin=15, flat=True)

        # append results to v and sigma lists
        sig_min_jack_all.append(np.percentile(flat_samples, 16, axis=0)[1])
        sig_jack_all.append(np.percentile(flat_samples, 50, axis=0)[1])
        sig_max_jack_all.append(np.percentile(flat_samples, 84, axis=0)[1])
        v_min_jack_all.append(np.percentile(flat_samples, 16, axis=0)[0])
        v_jack_all.append(np.percentile(flat_samples, 50, axis=0)[0])
        v_max_jack_all.append(np.percentile(flat_samples, 84, axis=0)[0])

    return sig_min_jack_all, sig_jack_all, sig_max_jack_all, v_min_jack_all, v_jack_all, v_max_jack_all
