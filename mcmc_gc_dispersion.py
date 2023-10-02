import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import emcee
import corner

from scipy.optimize import minimize

################# define all functions ########################
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


def sigma_rmse(v, v_gal):
    N = len(v)
    v_minus_v_gal = v - v_gal
    v_minus_v_gal_sq = np.full(len(v), np.nan)
    for el in enumerate(v_minus_v_gal):
        v_minus_v_gal_sq[el[0]] = el[1]**2
    return np.sqrt(np.nansum(v_minus_v_gal_sq)/N)


##################### run mcmc chain ##################################
# jonah vcc1448
# v_obs = np.array([2335, 2274, 2288, 2283, 2342, 2317, 2287, 2325, 2320, 2292])
# v_err = np.array([2,3,6,5,3,2,1,8,7,4])
# prior_range_v = (2274., 2335.)
# prior_range_sig = (0., 100.)

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
# v_obs = np.array([2147.0, 2149.0, 2173.0, 2148.0, 2130.0, 2155.0, 2135.0, 2176.0, 2116.0, 
#                 2130.0, 2220.0, 2132.0, 2130.0, 2164.0, 2180.0, 2094.0, 2189.0, 2164.0, 2175.0,
#                 2059.0, 2202.0, 2169.0, 2130.0])
# v_err = np.array([4.0, 6.0, 5.0, 3.0, 6.0, 4.0, 5.0, 3.0, 4.0, 8.0, 3.0, 15.0,
#                     4.0, 4.0, 34.0, 7.0, 10.0, 7.0, 12.0, 6.0, 7.0, 6.0, 7.0])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_final_prel.png'
# walker_title = 'walkers_final_prel.png'

# # excluding extremes - bottom two and top two
# v_obs = np.array([2147.0, 2149.0, 2173.0, 2148.0, 2130.0, 2155.0, 2135.0, 2176.0, 
#                 2130.0, 2132.0, 2130.0, 2164.0, 2180.0, 2189.0, 2164.0, 2175.0,
#                 2059.0, 2169.0, 2130.0])
# v_err = np.array([4.0, 6.0, 5.0, 3.0, 6.0, 4.0, 5.0, 3.0, 8.0, 15.0,
#                     4.0, 4.0, 34.0, 10.0, 7.0, 12.0, 6.0, 6.0, 7.0])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_final_prel.png'
# walker_title = 'walkers_final_prel.png'

# using just bh3l velocity
# v_obs = np.array([2138.0, 2144.0, 2169.0, 2148.0, 2130.0, 2155.0, 2131.0,
#                 2176.0, 2143.0, 2220.0, 2132.0, 2130.0, 2164.0, 2180.0, 2094.0,
#                 2189.0, 2164.0, 2175.0, 2059.0, 2202.0, 2169.0, 2130.0])
# v_err = np.array([2.0, 2.0, 3.0, 3.0, 6.0, 4.0, 4.0, 3.0, 5.0, 3.0,
#                 15.0, 4.0, 4.0, 34.0, 7.0, 10.0, 4.0, 12.0, 6.0, 7.0, 6.0, 7.0])
# prior_range_sig = (0., 100.)
# corner_title = 'mcmc_corner_final_prel.png'
# walker_title = 'walkers_final_prel.png'


# my velocities for Mueller's GCs
v_obs = np.array([2148.0, 2151.0, 2172.0, 2147.0,
                2171.0, 2130.0, 2156.0, 2133.0,
                2130.0, 2164.0, 2167.0])
v_err = np.array([4.0, 7.0, 5.0, 5.0,
                22.0, 8.0, 6.0, 6.0,
                7.0, 8.0, 9.0])
prior_range_sig = (0., 100.)
corner_title = 'mcmc_corner_mueller_gcs.png'
walker_title = 'walkers_mueller_gcs.png'






###### run with likelihood only ######
nwalkers = 100
ndim = 2
v_exp = np.mean(v_obs)
# sig_exp = sigma_rmse(v_obs, 2167) # to start with the velocity dispersion from rmse formula
sig_exp = 17 # to start with stellar sigma
prior_range_v = (np.min(v_obs), np.max(v_obs))

# get initial values
np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([v_exp, sig_exp]) + 10 * np.random.randn(nwalkers, ndim)

# run burn in 
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, args=[v_obs, v_err, prior_range_v, prior_range_sig])
pos, prob, state = sampler.run_mcmc(initial, 1000, progress=True)

print(pos)

# soln = minimize(nll, initial, args=(v_obs, v_err))
# # v_ml, sig_ml = soln.x
# pos = soln.x + np.random.randn(nwalkers, ndim)
# print(pos)

sampler.reset()
sampler.run_mcmc(pos, nsteps=20000, progress=True)


############################################# making plots #############################################

# make corner plot
flat_samples = sampler.get_chain(discard=0, thin=15, flat=True)
np.savetxt('../fred/NGC5846_UDG1/mcmc_chain.txt', flat_samples)
labels = ['v', 'sig']
fig = corner.corner(flat_samples, quantiles=(0.16, 0.5, 0.84), use_math_text=True, labels=labels, show_titles=True)
# print 16th, 50th, 84th percentile
print(np.nanpercentile(flat_samples, 16, axis=0))
print(np.median(flat_samples, axis=0))
print(np.nanpercentile(flat_samples, 84, axis=0))
plt.savefig(''.join(['../fred/NGC5846_UDG1/', corner_title]))
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
plt.savefig(''.join(['../fred/NGC5846_UDG1/', walker_title]))
plt.close(fig2)
