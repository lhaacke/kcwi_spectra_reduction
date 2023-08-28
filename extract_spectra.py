'''
Lydia Haacke
08/2023
'''
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

import pyregion
from pyregion.region_to_filter import as_region_filter
from photutils.aperture import SkyEllipticalAperture


class ExtractSpectra:
    def __init__(self, data_stacked_path, var_stacked_path, gc_dict):
        '''
        data_stacked_path: path to stacked data cube, including name.fits
        var_stacked_path: path to stacked variance cube, including name.fits
        gc_dict: dictionary containing the gc, corresponding ra, dec and aperture ellipse for extraction
        '''
        with fits.open(data_stacked_path) as hdu:
            self.data_cube_header = hdu[0].header
            self.data_cube_data = hdu[0].data
        with fits.open(data_stacked_path) as hdu:
            self.data_cube_header = hdu[0].header
            self.data_cube_data = hdu[0].data
        self.wcs = WCS(hdu[0].header)
        self.coords = SkyCoord(gc_dict)


    def get_spec_from_cube(self, mode='median', subtract_sky=True):
        '''
        mode: mode of combining the information from spaxels, default: median
        '''
        # get mask & combine with cube
        pixel_mask = get_pixel_mask(self.region[0])

        # extract spectrum
        spec = np.zeros(self.header['NAXIS3'])
        if mode=='median':
            for i in range(self.header['NAXIS3']):
                spec[i] = np.median(pixel_mask[0].get_values(self.data[i,:,:]))
        elif mode=='sum':
            for i in range(self.header['NAXIS3']):
                spec[i] = np.sum(pixel_mask[0].get_values(self.data[i,:,:]))

        if subtract_sky:
            spec = sky_subtraction(spec)

        # write spectrum to fits file
        spec_header = write_spectrum_header()
        spec_file = fits.PrimaryHDU(spec, spec_header)
        spec_file.writeto(self.save_dir, overwrite=True)

        return 0


    def sky_subtraction(self, spec, sky_spec):
        '''
        subtracts the sky_spec from the spec
        '''
        if spec.shape != sky_spec.shape:
            # throw error
        return spec - sky_spec


    def write_spectrum_header(self):
        '''
        returns a header object that can be used for a spectrum based on the header of the original 
        '''
        h1d = fits.PrimaryHDU()
        h1d[0].header['SIMPLE'] = self.header['SIMPLE']
        h1d[0].header['BITPIX'] = self.header['BITPIX']
        h1d[0].header['NAXIS'] = 1
        h1d[0].header['NAXIS1'] = self.header['NAXIS3']
        h1d[0].header['CTYPE1'] = self.header['CTYPE3']
        h1d[0].header['CUNIT1'] = self.header['CUNIT3']
        h1d[0].header['EXTEND'] = self.header['EXTEND']
        h1d[0].header['EQUINOX'] = self.header['EQUINOX']
        h1d[0].header['CDELT1'] = self.header['CDELT3']
        h1d[0].header['CRPIX1'] = self.header['CRPIX3']
        h1d[0].header['CRVAL1'] = self.header['CRVAL3']
        h1d[0].header['WAVALL0'] = self.header['WAVALL0']
        h1d[0].header['WAVALL1'] = self.header['WAVALL1']
        h1d[0].header['WAVGOOD0'] = self.header['WAVGOOD0']
        h1d[0].header['WAVGOOD1'] = self.header['WAVGOOD1']
        h1d[0].header['BUNIT'] = self.header['BUNIT']
        
        return h1d[0].header


    def get_pixel_mask(self, apertures):
        '''
        returns 3D mask that masks out all of the cube except the specified region
        region: region to not be masked
        shape: tuple with shape of the data cube in pixels
        '''
        mask = np.zeros(np.shape(self.data)) # make mask the same shape as data cube
        wcs_mask = self.get_mask_wcs()

        pixel_aperture = apertures.to_pixel(wcs=wcs_mask)
        pixel_mask = pixel_aperture.to_mask(method='exact') # mask in pixel coordinates

        return pixel_mask


    def get_mask_wcs(self):
        '''
        returns two-dimensional wcs based on self.wcs
        '''
        wcs_mask = self.wcs.dropaxis(dropax=2)
        return wcs_mask
        

class Spectrum:
    def __init__(self, spec_path):
        '''
        spec_path: path to spectrum including name.fits
        '''
        with fits.open(spec_path) as hdu:
            self.spectrum = hdu[0].data
            self.spectrum_header = hdu[0].header
        
    def smooth_spectrum(self, spec, sig=3):
        '''
        Smooths a spectrum using Gaussian kernel smoothing.
        spec (array-like): Array of intensity values corresponding to the wavelengths
        sig (float): Standard deviation of the Gaussian kernel
        returns smoothed spectrum
        '''
        smoothed_spec = gaussian_filter1d(self.spectrum, sig)

        return smoothed_spec

    def plot(self, smoothed_spectrum=True, sig=3):
        '''
        makes and saves various plots of the spectrum

        smoothed_spectrum (bool): whether or not to plot the smoothed spectrum
        sig: sig (float): Standard deviation of the Gaussian kernel when smoothing the spectrum
        '''
        if smoothed_spectrum:
            smoothed_spec = self.smooth_spectrum(self.spectrum, )
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18,9), sharex=True)
            x = np.arange(spec_header['WAVALL0'], spec_header['WAVALL1']+spec_header['CDELT1'], spec_header['CDELT1'])
            ax.plot(x, spec, c='tab:blue', label='full resolution spectrum')
            ax.plot(x, smoothed_spec, c='tab:orange', label='smoothed, sig={}'.format(sig))
            ax.legend(loc='upper right')
            plt.subplots_adjust(hspace=.0)
            

