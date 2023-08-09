'''
Lydia Haacke
05/2023
'''

import numpy as np
from astropy.io import fits


def hdu_wavelength_corection(file_location, save_location, sfxs, sfxs_fixed, corrs):
    '''
    corrs: dictionary with header data in the running part of the code
    '''
    for key in corrs.keys():
        path = ''.join([file_location, '/', key, sfxs])
        save_as = ''.join([save_location, '/', key, sfxs_fixed])
        with fits.open(path) as hdu: # get data from cube
            data = hdu[0].data
        newfile = fits.PrimaryHDU(data, hdu[0].header) # write to new fits file
        newfile.writeto(save_as, overwrite=True)
        print(corrs.keys())
        with fits.open(save_as, 'update') as hdu:
            hdu[0].header['WAVALL0'] = corrs[key]['wavall'][0]
            hdu[0].header['WAVALL1'] = corrs[key]['wavall'][1]
            hdu[0].header['WAVGOOD0'] = corrs[key]['wavgood'][0]
            hdu[0].header['WAVGOOD1'] = corrs[key]['wavgood'][1]
    return 0


def gradient_correction(file_location, save_location, sfxs, sfxs_median, corrs):
    '''
    '''
    for key in corrs.keys():
        path = ''.join([file_location, '/', key, sfxs]) # path where cut icube is
        save_as = ''.join([save_location, '/', key, sfxs_median])
        print(save_as)
        with fits.open(path) as hdu:
            data = hdu[0].data
            med = np.median(data, axis=1) # find median along second axis
            for yind in range(hdu[0].header['NAXIS2']):
                data[:, yind, :] = data[:, yind, :]/med # divide each column by median to remove gradient
        newfile = fits.PrimaryHDU(data, hdu[0].header) # write to new fits file
        newfile.writeto(save_as, overwrite=True)

    return 0
            

def wcs_correction(file_location, save_location, sfxs, sfxs_wcs, corrs):
    '''
    file_location: path to where the kcwi cubes are saved
    sfxs: suffixes apart from kbyymmdd_000xx
    corrs: dictionary of files and cooresponding ra/dec corrections
    '''
    for key in corrs.keys():
        path = ''.join([file_location, '/', key, sfxs])
        save_as = ''.join([save_location, '/', key, sfxs_wcs])
        with fits.open(path) as hdu: # get data from cube
            data = hdu[0].data
        newfile = fits.PrimaryHDU(data, hdu[0].header) # write to new fits file
        newfile.writeto(save_as, overwrite=True)
        with fits.open(save_as, 'update') as hdu:
            hdu[0].header['CRPIX1'] = corrs[key]['xpix']
            hdu[0].header['CRPIX2'] = corrs[key]['ypix']
            hdu[0].header['CRVAL1'] = corrs[key]['xval']
            hdu[0].header['CRVAL2'] = corrs[key]['yval']
    return 0

