import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import emcee
import corner

from scipy.optimize import minimize

################# define all functions ########################
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
        pre = np.log(1/(np.sqrt(2*sc.pi*(sig**2 + v_err**2))))
        chisq = 0.5*(v_obs - v_sys)**2/(sig**2 + v_err**2)

    return np.sum(pre - chisq)


def log_likelihood_1d(theta, v_obs, v_err):
    '''
    theta: tuple of free parameters in the mcmc
    v_obs: array of observed recessional velocities of GCs
    v_err: corresponding error of v_obs
    '''
    sig = theta
    pre = np.log(1/(np.sqrt(2*sc.pi*(sig**2 + v_err**2))))
    chisq = 0.5*(v_obs - 2280)**2/(sig**2 + v_err**2)
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


def sort_velocities(v_obs, v_err, crit):
    '''
    returns velocities sorted by criteria array
    v_obs: observed velocities
    v_err: errors associated with v_obs
    crit: criteria by which to sort v_obs and v_err (e.g. radius, magnitude,...)
    '''

    return v_obs_sorted, v_err_sorted


##################### run mcmc chain ##################################
# jonah vcc1448
# v_obs = np.array([2335, 2274, 2288, 2283, 2342, 2317, 2287, 2325, 2320, 2292])
# v_err = np.array([2,3,6,5,3,2,1,8,7,4])
# prior_range_v = (2274., 2335.)
# prior_range_sig = (0., 100.)


# Jonah GC recessional velocities:
# v_obs = np.array([2333.83, 2272.83, 2286.83, 2281.83,
#                 2340.83, 2315.83, 2285.83, 2323.83,
#                 2318.83, 2281.015])
# v_err =  np.array([2, 3, 6, 5, 3, 2, 1, 8, 7, 4])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_jonah_velocities.png'
# walker_title = 'walkers_jonah_velocities.png'
# hist_title = 'mcmc_sig_distr_jonah.png'

# more Jonah velocities - within halflight radius
v_obs = np.array([2272.83 , 2281.83 , 2340.83 , 2315.83 , 2285.83 , 2323.83 , 2318.83 , 2281.015, 2297.78, 2244.81])
v_err = np.array([ 3,  5, 3, 2, 1, 8, 7, 4,  8.76,  17.08,])
prior_range_sig = (0., 100.)
mode = '2D'
corner_title = 'mcmc_corner_jonah_halflight.png'
walker_title = 'walkers_jonah_halflight.png'
# hist_title = 'mcmc_sig_distr_jonah_halflight.png'

# Toloba's GC recessional velocities:
# v_obs = np.array([2270.94, 2347.72, 2380.78, 2297.78,
#                 2323.72, 2278.69, 2244.81, 2367.27, 2279.])
# v_err = np.array([ 6.18, 7.18, 11.42, 8.76, 3.06, 8.2,
#                             17.08, 8.47, 8.07])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_toloba_velocities.png'
# walker_title = 'walkers_toloba_velocities.png'
# hist_title = 'mcmc_sig_distr_toloba.png'

# Jonah velocities + Toloba's velocities:
# v_obs = np.array([2333.83, 2272.83, 2286.83, 2281.83,
#                     2340.83, 2315.83, 2285.83, 2323.83,
#                     2318.83, 2281.015, 2270.94, 2347.72,
#                     2380.78, 2297.78, 2244.81, 2367.27, 2279.])
# v_err = np.array([2, 3, 6, 5, 3, 2, 1, 8, 7, 4,
#                                 6.18, 7.18, 11.42, 8.76, 17.08,
#                                 8.47, 8.07])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_jonah_toloba_velocities_vfix.png'
# walker_title = 'walkers_jonah_toloba_velocities_vfix.png'
# hist_title = 'mcmc_sig_distr_jonah_toloba.png'


# mueller's values for Matlas2019 including candidates
# v_obs = np.array([2156.4, 2162.3, 2138.5, 2130.2, 2133.6, 2147.0, 2147.2, 2157.2, 2163.2, 2179.1, 2177.9, 2134.2, 2184.0, 2184.0])
# v_err = np.array([5.6, 23.5, 23.3, 13.3, 17.2, 7.8, 5.0, 13.8, 17.7, 13.7, 16.1, 18.9, 12.8, 12.8])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_mueller_matlas2019_incl_cands.png'
# walker_title = 'walkers_mueller_matlas2019_incl_cands.png'

# mueller's values for Matlas2019 excluding candidates
# v_obs = np.array([2156.4, 2162.3, 2138.5, 2130.2, 2133.6, 2147.0, 2147.2, 2157.2, 2163.2, 2179.1, 2177.9, 2134.2])
# v_err = np.array([5.6, 23.5, 23.3, 13.3, 17.2, 7.8, 5.0, 13.8, 17.7, 13.7, 16.1, 18.9])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_mueller_matlas2019.png'
# walker_title = 'walkers_mueller_matlas2019.png'

# all final UDG1 values
# v_obs = np.array([2145.75, 2147.39, 2173.53, 2147.5 , 2171.02, 2129.52, 2156.04,
#        2136.64, 2176.65, 2115.36, 2124.75, 2221.46, 2129.95, 2164.82,
#        2178.73, 2164.21, 2180.44, 2201.82, 2167.04])
# v_err = np.array([1.85,  2.93,  2.33,  6.99, 20.31,  8.49,  7.39,  2.27,  7.16,
#         7.72,  3.52,  7.09,  7.84,  7.37, 34.99,  3.56, 13.75,  9.52,
#         8.8])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_final.png'
# walker_title = 'walkers_final.png'

# my velocities for Mueller's GCs
# v_obs = np.array([2145.75, 2147.39, 2173.53, 2147.5 , 2171.02, 2129.52, 2156.04,
#        2136.64, 2129.95, 2164.21, 2167.04])
# v_err = np.array([1.85,  2.93,  2.33,  6.99, 20.31,  8.49,  7.39,  2.27,  7.84,
#         3.56,  8.8])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_mgcs_kcwi.png'
# walker_title = 'walkers_mgcs_kcwi.png'

# using just bh3l velocity with gc10 corrected from bh3m
# v_obs = np.array([2137.83, 2143.47, 2168.82, 2147.5 , 2171.02, 2129.52, 2156.04,
#        2131.31, 2176.65, 2107.32, 2144.69, 2221.46, 2129.95, 2164.82,
#        2178.73, 2164.82, 2180.44, 2201.82, 2167.04])
# v_err = np.array([1.64,  2.11,  2.88,  2.71, 19.26,  5.53,  3.63,  3.67,  3.12,
#         4.54,  5.18,  2.98,  4.47,  3.58, 34.39,  3.58, 12.15,  7.02,
#         6.01])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_bh3l.png'
# walker_title = 'walkers_bh3l.png'

# correct velocities including gc11
# v_obs = np.array([2137.8, 2143.5, 2169, 2147.5,
#         2171.0, 2129.5, 2156.0, 2131.3, 2176.7,
#         2104.3, 2104.5, 2221.5, 2130.,
#         2164.8, 2178.7, 2164.8, 2180.4, 2201.819768335,
#         2167.0377605])
# v_err = np.array([1.6, 2.1, 2.9, 2.7,
#         19.3, 5.5, 3.6, 3.7,
#         3.1, 4.6, 2.9, 3.0,
#         4.5, 3.6, 34.4, 3.6,
#         12.1, 7.0, 6.0])



# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner.png'
# walker_title = 'walkers.png'


# ###### run 2 D ######
if mode == '2D':
    nwalkers = 100
    ndim = 2
    v_exp = np.mean(v_obs)
    # sig_exp = sigma_rmse(v_obs, 2167) # to start with the velocity dispersion from rmse formula
    sig_exp = 26 # to start with stellar sigma
    prior_range_v = (np.min(v_obs), np.max(v_obs))

    # get initial values
    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([v_exp, sig_exp]) + 10 * np.random.randn(nwalkers, ndim)

    # run burn in 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, args=[v_obs, v_err, prior_range_v, prior_range_sig])
    pos, prob, state = sampler.run_mcmc(initial, 1000, progress=True)

    # print(pos)

    # soln = minimize(nll, initial, args=(v_obs, v_err))
    # # v_ml, sig_ml = soln.x
    # pos = soln.x + np.random.randn(nwalkers, ndim)
    # print(pos)

    sampler.reset()
    sampler.run_mcmc(pos, nsteps=20000, progress=True)

    # make corner plot
    flat_samples = sampler.get_chain(discard=0, thin=15, flat=True)
    # np.savetxt('../fred/NGC5846_UDG1/mcmc_results/mcmc_chain.txt', flat_samples)
    labels = ['v', 'sig']
    fig = corner.corner(flat_samples, quantiles=(0.16, 0.5, 0.84), use_math_text=True, labels=labels, show_titles=True)
    plt.savefig(''.join(['../fred/NGC5846_UDG1/mcmc_results/', corner_title]))
    plt.close(fig)

    # plot walkers
    fig2, axes2 = plt.subplots(ndim, figsize=(14, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes2[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes2[-1].set_xlabel('step number')
    plt.savefig(''.join(['../fred/NGC5846_UDG1/mcmc_results/', walker_title]))
    plt.close(fig2)



###### run 1 D ######
elif mode == '1D':
    nwalkers = 100
    ndim = 1
    # v_exp = np.mean(v_obs)
    # sig_exp = sigma_rmse(v_obs, 2167) # to start with the velocity dispersion from rmse formula
    sig_exp = 26 # to start with stellar sigma
    # prior_range_v = (np.min(v_obs), np.max(v_obs))

    # get initial values
    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([sig_exp]) + 10 * np.random.randn(nwalkers, ndim)

    # run burn in 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post_1d, args=[v_obs, v_err, prior_range_sig])
    pos, prob, state = sampler.run_mcmc(initial, 1000, progress=True)

    # print(pos)

    # soln = minimize(nll, initial, args=(v_obs, v_err))
    # # v_ml, sig_ml = soln.x
    # pos = soln.x + np.random.randn(nwalkers, ndim)
    # print(pos)

    sampler.reset()
    sampler.run_mcmc(pos, nsteps=20000, progress=True)

    ############################## print sigma from 1d fit #################################
    flat_samples = sampler.get_chain(discard=0, thin=15, flat=True)
    sig_min = np.nanpercentile(flat_samples, 16, axis=0)
    sig_med = np.median(flat_samples, axis=0)
    sig_max = np.nanpercentile(flat_samples, 84, axis=0)
    # print 16th, 50th, 84th percentile
    print(sig_min)
    print(sig_med)
    print(sig_max)

    # make histogram with distribution
    np.set_printoptions(precision=2)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(flat_samples, alpha=0.5)
    ax.axvline(sig_min, c='k')
    ax.axvline(sig_med, c='k')
    ax.axvline(sig_max, c='k')
    ax.set(xlabel=r'$\sigma$ (km/s)',
            title=r'$\sigma$ = {} + {} - {} (km/s)'.format(sig_med, sig_max-sig_med, sig_med-sig_min))
    plt.savefig(''.join(['../fred/NGC5846_UDG1/mcmc_results/', hist_title]))

