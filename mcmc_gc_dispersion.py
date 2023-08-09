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


def sig_rms():
    '''
    compute root mean squared velocity dispersion after Doppel21
    '''
    return 0


##################### run mcmc chain ##################################
# jonah vcc1448
# v_obs = np.array([2335, 2274, 2288, 2283, 2342, 2317, 2287, 2325, 2320, 2292])
# v_err = np.array([2,3,6,5,3,2,1,8,7,4])
# prior_range_v = (2274., 2335.)
# prior_range_sig = (0., 100.)

# mueller's values for Matlas2019 including candidates
# v_obs = np.array([2156.4, 2162.3, 2138.5, 2130.2, 2133.6, 2147.0, 2147.2, 2157.2, 2163.2, 2179.1, 2177.9, 2134.2, 2184.0, 2184.0])
# v_err = np.array([5.6, 23.5, 23.3, 13.3, 17.2, 7.8, 5.0, 13.8, 17.7, 13.7, 16.1, 18.9, 12.8, 12.8])
# prior_range_v = (2130., 2179.)
# prior_range_sig = (0., 100.)

# mueller's values for Matlas2019 excluding candidates
v_obs = np.array([2156.4, 2162.3, 2138.5, 2130.2, 2133.6, 2147.0, 2147.2, 2157.2, 2163.2, 2179.1, 2177.9, 2134.2])
v_err = np.array([5.6, 23.5, 23.3, 13.3, 17.2, 7.8, 5.0, 13.8, 17.7, 13.7, 16.1, 18.9])
prior_range_v = (2130., 2179.)
prior_range_sig = (0., 100.)



###### run with likelihood only ######
nwalkers = 100
ndim = 2
v_exp = np.mean(v_obs)
sig_exp = 9.4

# get initial values
np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([v_exp, sig_exp]) + 0.5 * np.random.randn(nwalkers, ndim)

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
np.savetxt('mcmc_chain.txt', flat_samples)
labels = ['v', 'sig']
fig = corner.corner(flat_samples, quantiles=(0.16, 0.5, 0.84), use_math_text=True, labels=labels)
# print 16th, 50th, 84th percentile
print(np.nanpercentile(flat_samples, 16, axis=0))
print(np.median(flat_samples, axis=0))
print(np.nanpercentile(flat_samples, 84, axis=0))
plt.savefig('../mcmc_corner_mueller_matlas2019.png')
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
plt.savefig('../walkers_mueller_matlas2019.png')
plt.close(fig2)
