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

from regions import Regions

import pyregion
from pyregion.region_to_filter import as_region_filter
from photutils.aperture import SkyEllipticalAperture

# import from own repository
from ppxf_fit_kinematics import *


class ExtractSpectra:
    def __init__(self, data_stacked_path, data_cube_name, var_stacked_path,
                 var_cube_name, regions_path, sky_regions_path):
        '''
        data_stacked_path: path to stacked data cube
        data_cube_name: file name of data cube, including file ending (e.g. .fits)
        var_stacked_path: path to stacked variance cube
        var_cube_name: file name of variance cube, including file ending (e.g. .fits)
        
        regions_path: path to file with regions (ds9 format) from which to extract the spectra
        sky_regions_path: path to sky region (careful: at the moment only supports one sky region for all the spectra)
        '''
        with fits.open(''.join([data_stacked_path, '/', data_cube_name])) as hdu:
            self.data_cube_header = hdu[0].header
            self.data_cube_data = hdu[0].data
            self.wcs = WCS(hdu[0].header)
        with fits.open(''.join([var_stacked_path, '/', var_cube_name])) as hdu:
            self.var_cube_header = hdu[0].header
            self.var_cube_data = hdu[0].data
        self.regions = Regions.read(regions_path)
        self.sky_regions = Regions.read(sky_regions_path)
        self.data_spectra_path = ''.join([data_stacked_path, '/data_spectra'])
        self.var_spectra_path = ''.join([var_stacked_path, '/var_spectra'])
        self.plot_path = ''.join([data_stacked_path, '/plots'])
        
    
    def get_masks_from_regions(self, regions):
        '''
        returns a list of masks based on the regions file provided
        '''
        masks = [] # list of all the masks to be returned
        for region in regions:
            aper = SkyEllipticalAperture(region.center, region.width, region.height, theta=region.angle)
            aper_pix = aper.to_pixel(wcs=self.wcs.dropaxis(dropax=2))
            aper_mask = aper_pix.to_mask(method='exact')
            masks.append(aper_mask)
        
        return masks
    
    
    def write_spectrum_header(self, spec_header, data_header, comment=''):
        '''
        updates the header of the spectrum fits file with all necessary keywords
        
        spec_header: header of the spectrum
        data_header: header that contains necessary information
                     default to the header of the data cube the spectrum is taken from
        '''
        spec_header['NAXIS'] = 1
        spec_header['NAXIS1'] = data_header['NAXIS3']
        spec_header['CTYPE1'] = data_header['CTYPE3']
        spec_header['CUNIT1'] = data_header['CUNIT3']
        spec_header['EXTEND'] = data_header['EXTEND']
        spec_header['EQUINOX'] = data_header['EQUINOX']
        spec_header['CRVAL1'] = data_header['CRVAL3']
        spec_header['CRPIX1'] = data_header['CRPIX3']
        spec_header['CDELT1'] = data_header['CDELT3']
        spec_header['WAVALL0'] = data_header['WAVALL0']
        spec_header['WAVALL1'] = data_header['WAVALL1']
        spec_header['WAVGOOD0'] = data_header['WAVGOOD0']
        spec_header['WAVGOOD1'] = data_header['WAVGOOD1']
        spec_header['BUNIT'] = data_header['BUNIT']
        spec_header['COMMENT'] = comment
        
        return spec_header
    
    
    def sky_subtract(self, spec, header, cube, mode='median'):
        '''
        extracts a sky spectrum and subtracts it from spec
        
        spec: spectrum to subtract the sky spectrum from
        '''
        sky_spec = np.zeros(header['NAXIS3']) # initialise spectrum
        sky_mask = self.get_masks_from_regions(self.sky_regions) # make sky mask
        # apply the sky mask
        if mode=='median':
            for i in range(header['NAXIS3']):
                sky_spec[i] = np.median(sky_mask[0].get_values(cube[i,:,:]))
        elif mode=='sum':
            for i in range(header['NAXIS3']):
                sky_spec[i] = np.median(sky_mask[0].get_values(cube[i,:,:]))
        else:
            sys.exit('Unknown method of combining pixels.')
        
        # subtract spectra
        sky_subtracted_spec = spec - sky_spec
        
        if return_mask:
            return sky_mask
        else:
            return sky_subtracted_spec
    
    
    def extract_spectra(self, sky_subtract=True):
        '''
        extracts spectrum from specified regions and save it as fits in a folder
        
        sky_subtract (Boolean type): saved spectra will be locally sky subtracted if true
        '''
        if not os.path.exists(self.data_spectra_path):
            os.mkdir(self.data_spectra_path)
        if not os.path.exists(self.var_spectra_path):
            os.mkdir(self.var_spectra_path)
        
        # get the masks
        masks = self.get_masks_from_regions(self.regions)
        
        # apply the mask
        for i, mask in enumerate(masks):
            data_spec_med = np.zeros(self.data_cube_header['NAXIS3'])
            data_spec_sum = np.zeros(self.data_cube_header['NAXIS3'])
            var_spec_med = np.zeros(self.var_cube_header['NAXIS3'])
            var_spec_sum = np.zeros(self.var_cube_header['NAXIS3'])
            
            for j in range(self.data_cube_header['NAXIS3']):
                data_spec_med[j] = np.median(mask.get_values(self.data_cube_data[j,:,:]))
                data_spec_sum[j] = np.sum(mask.get_values(self.data_cube_data[j,:,:]))
                mask_weighted_data[j,:,:] *= mask_im
                var_spec_med[j] = np.median(mask.get_values(self.var_cube_data[j,:,:]))
                var_spec_sum[j] = np.sum(mask.get_values(self.var_cube_data[j,:,:]))
                mask_weighted_var[j,:,:] *= mask_im
                
            # apply sky subtraction if sky_subtract=True
            if sky_subtract:
                data_spec_med_skysub = self.sky_subtract(data_spec_med, header=self.data_cube_header,
                                                         cube=self.data_cube_data, mode='median')
                data_spec_sum_skysub = self.sky_subtract(data_spec_sum, header=self.data_cube_header,
                                                         cube=self.data_cube_data, mode='sum')
                var_spec_med_skysub = self.sky_subtract(var_spec_med, header=self.var_cube_header,
                                                         cube=self.var_cube_data, mode='median')
                var_spec_sum_skysub = self.sky_subtract(var_spec_sum, header=self.var_cube_header,
                                                         cube=self.var_cube_data, mode='sum')
                # create spectra hdus
                data_spec_med_skysub_hdu = fits.PrimaryHDU(data=data_spec_med_skysub)
                data_spec_sum_skysub_hdu = fits.PrimaryHDU(data=data_spec_sum_skysub)
                var_spec_med_skysub_hdu = fits.PrimaryHDU(data=var_spec_med_skysub)
                var_spec_sum_skysub_hdu = fits.PrimaryHDU(data=var_spec_sum_skysub)
                # update the headers
                data_spec_med_skysub_hdu.header = self.write_spectrum_header(data_spec_med_skysub_hdu.header,
                                                                      data_header=self.data_cube_header,
                                                                      comment='median combined spectrum')
                data_spec_sum_skysub_hdu.header = self.write_spectrum_header(data_spec_sum_skysub_hdu.header,
                                                                      data_header=self.data_cube_header,
                                                                      comment='sum combined spectrum')
                var_spec_med_skysub_hdu.header = self.write_spectrum_header(var_spec_med_skysub_hdu.header,
                                                                      data_header=self.var_cube_header,
                                                                      comment='median combined spectrum')
                var_spec_sum_skysub_hdu.header = self.write_spectrum_header(var_spec_sum_skysub_hdu.header,
                                                                      data_header=self.var_cube_header,
                                                                      comment='sum combined spectrum')
                # save the files
                data_spec_med_skysub_hdu.writeto(''.join([self.data_spectra_path, '/gc{}_spectrum_median.fits'.format(i)]), overwrite=True)
                data_spec_sum_skysub_hdu.writeto(''.join([self.data_spectra_path, '/gc{}_spectrum_sum.fits'.format(i)]), overwrite=True)
                var_spec_med_skysub_hdu.writeto(''.join([self.var_spectra_path, '/gc{}_spectrum_median.fits'.format(i)]), overwrite=True)
                var_spec_sum_skysub_hdu.writeto(''.join([self.var_spectra_path, '/gc{}_spectrum_sum.fits'.format(i)]), overwrite=True)
            # unsubtracted spectra
            else:
                # create spectra hdus
                data_spec_med_hdu = fits.PrimaryHDU(data=data_spec_med)
                data_spec_sum_hdu = fits.PrimaryHDU(data=data_spec_sum)
                var_spec_med_hdu = fits.PrimaryHDU(data=var_spec_med)
                var_spec_sum_hdu = fits.PrimaryHDU(data=var_spec_sum)
                # update the headers
                data_spec_med_hdu.header = self.write_spectrum_header(data_spec_med_hdu.header, comment='median combined spectrum')
                data_spec_sum_hdu.header = self.write_spectrum_header(data_spec_sum_hdu.header, comment='sum combined spectrum')
                var_spec_med_hdu.header = self.write_spectrum_header(var_spec_med_hdu.header, comment='median combined spectrum')
                var_spec_sum_hdu.header = self.write_spectrum_header(var_spec_sum_hdu.header, comment='sum combined spectrum')
                # save the file
                data_spec_med_hdu.writeto(''.join([self.data_spectra_path, '/gc{}_spectrum_median.fits'.format(i)]), overwrite=True)
                data_spec_sum_hdu.writeto(''.join([self.data_spectra_path, '/gc{}_spectrum_sum.fits'.format(i)]), overwrite=True)
                var_spec_med_hdu.writeto(''.join([self.var_spectra_path, '/gc{}_spectrum_median.fits'.format(i)]), overwrite=True)
                var_spec_sum_hdu.writeto(''.join([self.var_spectra_path, '/gc{}_spectrum_sum.fits'.format(i)]), overwrite=True)
         
        return 0
    
        
    def plot_apertures(self, cube, mask, sky_mask, plot_cube=True, plot_mask=True, plot_sky=True):
        '''
        plots the median image of the data cube, applied mask, applied sky aperture

        cube (bool): include median image of the original data cube if True
        mask (bool): include image of the aperture from which spectra are extracted if True
        sky (bool): include image of sky area from which sky spectra is extracted if True
        '''
        if plot_cube and plot_mask and plot_sky:
            with fits.open(cube) as hdu:
                data = hdu[0].data
                cube = 
            cube_median = np.median(cube)
            mask_im = mask.to_image(shape=(cube['NAXIS2'], self.data_cube_header['NAXIS1']))
            
            mask_weighted_data = np.zeros(shape=self.data_cube_data.shape)
            mask_weighted_var = np.zeros(shape=self.var_cube_data.shape)
        
        

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
            smoothed_spec = self.smooth_spectrum(self.spectrum)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18,9), sharex=True)
            x = np.arange(self.spectrum_header['WAVALL0'], self.spectrum_header['WAVALL1']+self.spectrum_header['CDELT1'], self.spectrum_header['CDELT1'])
            ax.plot(x, self.spectrum, c='tab:blue', label='full resolution spectrum')
            ax.plot(x, smoothed_spec, c='tab:orange', label='smoothed, sig={}'.format(sig))
            ax.legend(loc='upper right')
            plt.subplots_adjust(hspace=.0)
            

