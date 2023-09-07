import glob
from os import path
from time import perf_counter as clock

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage
import numpy as np

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

def vac_to_air(lam_vac):
    # IAU standard: AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)
    lam_air = lam_vac / (1.0 + 2.735182e-4 + 131.4182 / lam_vac**2 + 2.76249e8 / lam_vac**4)
    return lam_air

def fit_vel_sigma(spectrum, save_as, z, grating, shift_spec=True, cut_spec=False, fit=False, bootstrap=False, smoothed_spec='', mask_skylines=False, plot_results=False):
    '''
    currently working for KCWI spectra (and specifically for swinburne or yale observed of NGC5846_UDG1)
    spectrum: fits file with spectrum to fit
    save_as: if fit==True - txt file to write result to
            if bootstrap==True - fits file to write result to
    z: initial redshift guess
    grating: (which KCWI grating was used, which uni observed)
    shift_spec: shift spectrum to redshift zero before fitting if True
    cut_spec: remove wavelengths outside of wavgood
    fit: fit multiple combinations of degree, mdegree
    bootstrap: fit 10 times with 10 different parts of the spectrum masked
    smoothed_spec: smoothed spectrum of same shape as spectrum for plotting
    '''

    ppxf_dir = path.dirname(path.realpath(util.__file__))

    ################################ PROCESS GALAXY SPECTRUM ##############################
    # read the spectrum and define the wavelength range to fit
    file = spectrum
    hdu = fits.open(file)
    gal_lin = hdu[0].data
    h1 = hdu[0].header

    inf_check = np.isinf(gal_lin)
    nan_check = np.isnan(gal_lin)
    gal_lin[inf_check] = np.median(gal_lin)
    gal_lin[nan_check] = np.nanmedian(gal_lin)

    lamRange1 = np.array([h1['WAVALL0'], h1['WAVALL1']])
    print(grating)

    if grating=='BH3_Medium':
        fwhm_gal = np.average(lamRange1/9000)
    elif grating=='BH3_Large':
        fwhm_gal = np.average(lamRange1/4500)
    elif grating=='BL_Large':
        fwhm_gal = np.average(lamRange1/900)
    print('{}, fwhm_gal:{}'.format(grating, fwhm_gal))

    # redshift adjustment if necessary
    # controlled by bool shift_spec, default True
    if shift_spec:                   # Use these lines if your spectrum is at high-z
        redshift_0 = z              # Initial guess of the galaxy redshift
        lamRange1 = lamRange1/(1 + redshift_0)     # Compute approximate restframe wavelength range
        fwhm_gal /= (1 + redshift_0)      # Adjust resolution in wavelength units
        redshift = 0
    else:
        redshift_0 = z
        redshift = redshift_0

    galaxy, ln_lam1, velscale = util.log_rebin(lamRange1, gal_lin)
    galaxy = galaxy/np.median(galaxy)  # Normalize spectrum to avoid numerical issues
    noise = np.full_like(galaxy, 1)           # Assume constant noise per pixel here


    ################################# GET THE TEMPLATES #####################################
    coelho = glob.glob(ppxf_dir + '/s_coelho14_highres/*.fits')
    # FWHM_tem = 2.51     # Vazdekis+16 spectra have a constant resolution FWHM of 2.51A.
    velscale_ratio = 2  # adopts 2x higher spectral sampling for templates than for galaxy

    hdu = fits.open(coelho[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lam2 = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
    fwhm_tem = np.average(lam2/20000) # R=20000 for coelho library

    good_lam = (lam2 > lamRange1[0]/1.02) & (lam2 < lamRange1[1]*1.02)
    ssp, lam2 = ssp[good_lam], lam2[good_lam]

    lamRange2 = [np.min(lam2), np.max(lam2)]
    sspNew, ln_lam2 = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)[:2]
    templates = np.empty((sspNew.size, len(coelho)))

    fwhm_dif = np.sqrt(fwhm_gal**2 - fwhm_tem**2)
    sigma = fwhm_dif/2.355/h2['CDELT1']  # Sigma difference in pixels

    for j, file in enumerate(coelho):
        hdu = fits.open(file)
        ssp = hdu[0].data
        ssp = ndimage.gaussian_filter1d(ssp[good_lam], sigma)
        sspNew = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)[0]
        templates[:, j] = sspNew/np.median(sspNew[sspNew > 0])  # Normalizes templates

    ############################### MASK PIXELS ############################################
    if cut_spec:
        print('spectrum cut')
        n_pix_left = int((h1['WAVGOOD0']-h1['WAVALL0']) / h1['CDELT1'] + 5)
        n_pix_right = int((h1['WAVALL1']-h1['WAVGOOD1']) / h1['CDELT1'] + 5)
        mask = np.full_like(galaxy, 1)
        mask[:n_pix_left] = 0
        mask[(len(galaxy)-n_pix_right):] = 0
        goodPixels = np.flatnonzero(mask)
    elif bootstrap:
        print('making masks for bootstrap error')
        bootstrap_masks = np.ones(shape=(10, len(galaxy)))
        goodPixels = []
        # masking edges for all of theme
        n_pix_left = int((h1['WAVGOOD0']-h1['WAVALL0']) / h1['CDELT1'] + 5)
        n_pix_right = int((h1['WAVALL1']-h1['WAVGOOD1']) / h1['CDELT1'] + 5)
        mask_frac = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.)]
        for mask in enumerate(bootstrap_masks):
            mask[1][:n_pix_left] = 0
            mask[1][(len(galaxy)-n_pix_right):] = 0
            mask[1][int(mask_frac[mask[0]][0]*len(galaxy)) : int(mask_frac[mask[0]][1]*len(galaxy))] = 0
            goodPixels.append(np.flatnonzero(mask[1]))
        print('bootstrap masks done')
    elif mask_skylines:
        skylines = np.array([5199, 5577, 5592])
        mask = np.full_like(galaxy, 1)
        mask[:10] = 0
        mask[-10:] = 0
        for line in skylines:
            c = ((ln_lam1 < np.log(line - 5)) | (ln_lam1 > np.log(line + 5)))
            mask_indices = np.where(c)
            mask[mask_indices] = 0
        goodPixels = np.flatnonzero(mask)
    else:
        print('spectrum not cut')
        mask = np.full_like(galaxy, 1)
        mask[:10] = 0
        mask[-10:] = 0
        goodPixels = np.flatnonzero(mask)
    print(len(goodPixels))

    ######################### FIT A GC WITH DIFFERENT DEGREES, MDEGREES ##################
    c = 299792.458
    if shift_spec:
        vel = c*np.log(1 + redshift)   # eq.(8) of Cappellari (2017, MNRAS)
    else:
        vel = c*np.log(1 + redshift_0)
    print('starting guess velocity: {}'.format(vel))
    start = [vel, 20.]  # (km/s), starting guess for [V, sigma]
    t = clock()

    if fit:
        degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        mdegree = [1, 2, 3, 4, 5, 6, 7, 8]
        res = np.recarray(shape = (len(degree)*len(mdegree)), # array to store the result for each degree combination in
                dtype = [('z_ini', float), ('deg', int), ('mdeg', int), ('v', float), ('v_err', float), ('z', float), ('z_err', float), ('sig', float),
                        ('sig_err', float), ('sn_median', float)]) # one for deg, mdeg, v, v_err, redshift, redshift_error, S/N
        i = 0
        for deg in degree: # iterate through degrees
            for mdeg in mdegree: # iterate through mdegrees
                pp = ppxf(templates, galaxy, noise, velscale, start, goodpixels=goodPixels,
                        moments=2, degree=deg, mdegree=mdeg,
                        lam=np.exp(ln_lam1), lam_temp=np.exp(ln_lam2),
                        velscale_ratio=velscale_ratio)
                vcosm = c*np.log(1 + z)  # This is the initial redshift estimate
                vpec = pp.sol[0]          # This is the fitted residual velocity
                vtot = vcosm + vpec       # I add the two velocities before computing z
                print('vcosm: {}, vpec: {}, vtot: {}'.format(vcosm, vpec, vtot))

                redshift_best = np.exp(vtot/c) - 1          # eq.(8) Cappellari (2017)
                errors = pp.error*np.sqrt(pp.chi2)          # Assume the fit is good
                redshift_err = np.exp(vtot/c)*errors[0]/c   # Error propagation
                sn_median = np.median(2*np.sqrt((pp.bestfit/(pp.galaxy - pp.bestfit))**2)) # signal to noise ratio median
                sn_average = np.average(np.sqrt((pp.bestfit/(pp.galaxy - pp.bestfit))**2)) # signal to noise ratio average
                # fill array with results
                res[i] = (z, deg, mdeg, vtot, errors[0], redshift_best, redshift_err, pp.sol[1], errors[1], sn_median)
                i += 1
                if plot_results and (((deg < 3) and (mdeg < 3)) or ((deg > 12) and (mdeg < 3))):
                    save_plot = '{}_z{}_deg{}_mdeg{}.png'.format(spectrum[:-5], z, deg, mdeg)
                    plot_result(pp, save_plot, redshift_best, lamRange1, smoothed_spec=smoothed_spec)

    elif bootstrap:
        # bootstrap_masks: array with different masks with different parts of the spectra masked
        # plan: run fit for each mask in the bootstrap mask array and save results same as with fit
        deg = 1
        mdeg = 1
        res = np.recarray(shape = (len(bootstrap_masks) + 1), # array to store the result for each degree combination in
                dtype = [('deg', int), ('mdeg', int), ('v', float), ('v_err', float), ('z', float), ('z_err', float), ('sig', float),
                        ('sig_err', float), ('sn_median', float), ('sn_average', float)]) # one for deg, mdeg, v, v_err, redshift, redshift_error, S/N
        i = 0
        for mask in goodPixels:
            pp = ppxf(templates, galaxy, noise, velscale, start, goodpixels=mask,
                    moments=2, degree=deg, mdegree=mdeg,
                    lam=np.exp(ln_lam1), lam_temp=np.exp(ln_lam2),
                    velscale_ratio=velscale_ratio)

            vcosm = c*np.log(1 + z)  # This is the initial redshift estimate
            vpec = pp.sol[0]          # This is the fitted residual velocity
            vtot = vcosm + vpec       # I add the two velocities before computing z
            print('vcosm: {}, vpec: {}, vtot: {}'.format(vcosm, vpec, vtot))

            redshift_best = np.exp(vtot/c) - 1          # eq.(8) Cappellari (2017)
            errors = pp.error*np.sqrt(pp.chi2)          # Assume the fit is good
            redshift_err = np.exp(vtot/c)*errors[0]/c   # Error propagation
            sn_median = np.median(np.sqrt((pp.bestfit/(pp.galaxy - pp.bestfit))**2)) # signal to noise ratio median
            sn_average = np.average(np.sqrt((pp.bestfit/(pp.galaxy - pp.bestfit))**2)) # signal to noise ratio average
            # fill array with results
            res[i] = (deg, mdeg, vtot, errors[0], redshift_best, redshift_err, pp.sol[1], errors[1], sn_median, sn_average)
            i += 1
        res[i] = (99, 99, np.average(res['v'][:-1]), np.average(res['v_err'][:-1]), np.average(res['z'][:-1]), np.average(res['z_err'][:-1]),
                    np.average(res['sig'][:-1]), np.average(res['sig_err'][:-1]), np.average(res['sn_median'][:-1]), np.average(res['sn_average'][:-1]))
    else:
        pp = ppxf(templates, galaxy, noise, velscale, start, goodpixels=goodPixels,
                  plot=False, moments=2, degree=10, mdegree=6,
                  lam=np.exp(ln_lam1), lam_temp=np.exp(ln_lam2),
                  velscale_ratio=velscale_ratio)

        # The updated best-fitting redshift is given by the following
        # lines (using equation 8 of Cappellari 2017, MNRAS)
        vcosm = c*np.log(1 + redshift_0)            # This is the initial redshift estimate
        print(vcosm)
        vpec = pp.sol[0]                            # This is the fitted residual velocity
        print(vpec)
        vtot = vcosm + vpec                         # I add the two velocities before computing z
        print(vtot)
        if shift_spec:
            redshift_best = np.exp(vtot/c) - 1          # eq.(8) Cappellari (2017)
            errors = pp.error*np.sqrt(pp.chi2)          # Assume the fit is good
            redshift_err = np.exp(vtot/c)*errors[0]/c   # Error propagation
        else:
            redshift_best = np.exp(vpec/c) - 1          # eq.(8) Cappellari (2017)
            errors = pp.error*np.sqrt(pp.chi2)          # Assume the fit is good
            redshift_err = np.exp(vpec/c)*errors[0]/c   # Error propagation
        sn = np.median(2*np.sqrt(((pp.bestfit-pp.apoly)/(pp.galaxy - pp.bestfit))**2))

        print("Formal errors:")
        print("     dV    dsigma   dh3      dh4")
        print("".join("%8.2g" % f for f in errors))
        print('Elapsed time in pPXF: %.2f s' % (clock() - t))
        print(f"Best-fitting redshift z = {redshift_best:#.7f} "
            f"+/- {redshift_err:#.2g}")
        print(sn)

    if fit:
        with open(save_as, 'a') as f:
            for line in res:
                f.write(str(line))
                f.write('\n')
    elif bootstrap:
        fits.writeto(save_as, res, overwrite=True)
    elif plot_results:
        plot_result(pp, save_as, redshift_best, lamRange1, smoothed_spec=smoothed_spec)
        # plt.savefig(save_as)
        plt.close()



def plot_result(pp, save_as, z_best, lamRange1, spec=True, shift_z=True, smoothed_spec=''):
    scale = 1e4  # divide by scale to convert Angstrom to micron

    if pp.lam is None:
        plt.xlabel("Pixels")
        x = np.arange(pp.galaxy.size)
    else:
        plt.xlabel(r"$\lambda_{\rm rest}\; (\mu{\rm m})$")
        if shift_z:
            x = pp.lam*(1+z_best)
        else:
            x = pp.lam

    plt.ylabel("Relative Flux ($f_\lambda$)")

    # Plot observed spectrum
    if spec:
        ll, rr = np.min(x), np.max(x)
        print(ll)
        print(rr)
        resid = pp.galaxy - pp.bestfit
        bestfit = pp.bestfit
        sig3 = np.percentile(abs(resid[pp.goodpixels]), 99.73)
        ref = np.min(bestfit) - 2*sig3
        mx = np.max(bestfit) + sig3
        resid += ref                           # Offset residuals to avoid overlap
        mn = np.min(resid[pp.goodpixels])    # Plot all fitted residuals without clipping

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18, 12))
        plt.subplots_adjust(hspace=.0)

        # plot the galaxy and goodPixels
        ax[1].plot(x, pp.galaxy, 'black', linewidth=1.5)
        # plot masked regions
        w = np.flatnonzero(np.diff(pp.goodpixels) > 1)
        for wj in w:
            a, b = pp.goodpixels[wj : wj + 2]
            ax[1].axvspan(x[a], x[b], facecolor='lightgray')
            ax[1].plot(x[a : b + 1], resid[a : b + 1], 'blue', linewidth=1.5)
        for k in pp.goodpixels[[0, -1]]:
            ax[1].plot(x[[k, k]], [ref, pp.bestfit[k]], 'lightgray', linewidth=1.5)
        # plot the fit
        ax[1].plot(x, pp.bestfit, 'red', linewidth=2, label='fit')
        ax[1].legend()
        # ax[0].plot(x[pp.goodpixels], pp.goodpixels*0 + ref, '.k', ms=1)

        # plot the residual
        ax[0].plot(x[pp.goodpixels], resid[pp.goodpixels],
                color='LimeGreen', label='residual')
        ax[0].legend()

        # plot the smoothed spectrum
        # with fits.open(smoothed_spec) as hdu:
        #     smoothed = hdu[0].data
        # ax[2].plot(x, smoothed, 'black', linewidth=1.5, label='smoothed spectrum')
        # ax[2].legend()

        # plot the multiplicative polynomial
        # ax[2].plot(x, pp.galaxy, 'black', linewidth=1.5)
        # ax[2].plot(x, pp.mpoly, 'red', linewidth=2, label='mult polynomial')
        # ax[2].legend()

        # plot the average additive polynomial
        # ax[2].plot(x, pp.galaxy, 'black', linewidth=1.5, label='galaxy')
        # print(len(x))
        # print(len(pp.galaxy))
        # print(len(np.sum(pp.matrix[:, :degree], axis=0)/pp.matrix.shape[0]))
        # # ax[2].plot(x, np.sum(pp.matrix, axis=0)/pp.matrix.shape[0], label='av add polynomial')
        # ax[2].legend()

        # draw lines and annotate

        # lines at redshift zero in Angstrom
        h_beta = 4861.35
        mag_b1 = 5167.322
        mag_b2 = 5172.684
        mag_b3 = 5183.604
        fe1_1 = 4918
        fe1_2 = 4921
        fe1_3 = 4933
        fe1_4 = 4957.5967
        fe1_5 = 5042
        fe1_6 = 5167.4883
        o3_1 = 4932.603
        o3_2 = 4960.295
        o3_3 = 5008.24
        ca_k = 3934.777
        ca_h = 3969.588
        lines = np.array([h_beta, mag_b1, mag_b2, mag_b3, fe1_1, fe1_2, fe1_3, fe1_4, fe1_5, fe1_6, o3_1, o3_2, o3_3,
                        ca_k, ca_h])

        if shift_z:
            lines_zshift = lines * (1 + z_best) # lines redshifted
            ax[1].axvline(lines_zshift[0], c='blue')
            ax[1].axvline(lines_zshift[1], c='red')
            ax[1].axvline(lines_zshift[2], c='red')
            ax[1].axvline(lines_zshift[3], c='red')
            ax[1].axvline(lines_zshift[5], c='m')
            ax[1].axvline(lines_zshift[-2], c='c')
            ax[1].axvline(lines_zshift[-1], c='c')
            x_bounds = ax[1].get_xlim()
            ax[1].annotate(text=r'H$_{beta}$', xy = (((lines_zshift[0] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0)
            ax[1].annotate(text='Magnesium b triplet', xy = (((lines_zshift[1] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0, c='red')
            ax[1].annotate(text='', xy = (((lines_zshift[2] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0, c='red')
            ax[1].annotate(text='', xy = (((lines_zshift[3] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0, c='red')
            ax[1].annotate(text='', xy = (((lines_zshift[5] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0, c='m')
            ax[1].annotate(text='Ca K', xy = (((lines_zshift[-2] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0, c='c')
            ax[1].annotate(text='Ca H', xy = (((lines_zshift[-1] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0, c='c')
            # ax[2].axvline(lines_zshift[0], c='blue')
            # ax[2].axvline(lines_zshift[1], c='red')
            # ax[2].axvline(lines_zshift[2], c='red')
            # ax[2].axvline(lines_zshift[3], c='red')
            # ax[2].axvline(lines_zshift[5], c='m')
        else:
            ax.axvline(lines[0])
            ax.axvline(lines[1])
            ax.axvline(lines[2])
            ax.axvline(lines[3])
            ax.axvline(lines[4])
            ax.axvline(lines[5])
            ax.axvline(lines[6])
            ax.axvline(lines[7])
            x_bounds = ax.get_xlim()
            ax.annotate(text=r'H$_{beta}$', xy = (((lines[0] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0)
            ax.annotate(text='Magnesium b triplet', xy = (((lines[1] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0)
            ax.annotate(text='', xy = (((lines[2] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0)
            ax.annotate(text='', xy = (((lines[3] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0)
            ax.annotate(text='Fe I', xy = (((lines[4] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0)
            ax.annotate(text='Fe I', xy = (((lines[5] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0)
            ax.annotate(text='O III', xy = (((lines[4] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0)
            ax.annotate(text='O III', xy = (((lines[5] - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), 1.01),
                xycoords='axes fraction', verticalalignment='center', horizontalalignment='left',
                rotation = 0)



    ax[1].set(xlim = ([ll, rr] + np.array([-0.02, 0.02])*(rr - ll)))
    # ax[2].set(xlim = ([ll, rr] + np.array([-0.02, 0.02])*(rr - ll)))
    ax[0].set(xlim = ([ll, rr] + np.array([-0.02, 0.02])*(rr - ll)))
    # ax[2].set(xlim = ([ll, rr] + np.array([-0.02, 0.02])*(rr - ll)), ylim = (-1, 1))
    # ax[1].set_ylim([mn, mx] + np.array([-0.05, 0.05])*(mx - mn))
    plt.savefig(save_as)
