'''
Lydia Haacke
08/2023
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from fabada import fabada
from astropy import units as u

from regions import Regions

from photutils.aperture import SkyEllipticalAperture


class ExtractSpectra:
    def __init__(self, cube_path, stacked_cubes_name, regions_path, sky_regions_path):
        '''
        data_stacked_path: path to stacked data cube
        data_cube_name: file name of data cube, including file ending (e.g. .fits)
        var_stacked_path: path to stacked variance cube
        var_cube_name: file name of variance cube, including file ending (e.g. .fits)
        
        regions_path: path to file with regions (ds9 format) from which to extract the spectra
        sky_regions_path: path to sky region (careful: at the moment only supports one sky region for all the spectra)
        '''
        with fits.open(''.join([cube_path, 'data_', stacked_cubes_name])) as hdu:
            self.data_cube_header = hdu[0].header
            self.data_cube_data = hdu[0].data
            self.wcs = WCS(hdu[0].header)
        with fits.open(''.join([cube_path, 'var_', stacked_cubes_name])) as hdu:
            self.var_cube_header = hdu[0].header
            self.var_cube_data = hdu[0].data
        self.regions = Regions.read(regions_path)
        self.sky_regions = Regions.read(sky_regions_path) # make this into sky coords?
        self.data_spectra_path = ''.join([cube_path, '/data_spectra'])
        self.var_spectra_path = ''.join([cube_path, '/var_spectra'])
        self.plot_path = ''.join([cube_path, '/plots'])
        
    
    def get_masks_from_regions(self, regions, sky_regions):
        '''
        returns a list of masks based on the regions file provided
        '''
        masks = [] # list of all the masks to be returned
        sky_masks = [] # list of corresponding sky masks
        for region, sky_region in zip(regions, sky_regions):
            aper = SkyEllipticalAperture(region.center, region.width, region.height, theta=region.angle)
            aper_pix = aper.to_pixel(wcs=self.wcs.dropaxis(dropax=2))
            aper_mask = aper_pix.to_mask(method='exact')
            masks.append(aper_mask)
            
            sky = SkyEllipticalAperture(sky_region.center, sky_region.width,
                                        sky_region.height, theta=sky_region.angle)
            sky_pix = sky.to_pixel(wcs=self.wcs.dropaxis(dropax=2))
            sky_mask = sky_pix.to_mask(method='exact')
            sky_masks.append(sky_mask)

        return masks, sky_masks
    
    
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
    
    
    def sky_subtract(self, spec, header, cube, sky_mask, mode='median'):
        '''
        extracts a sky spectrum and subtracts it from spec
        
        spec: spectrum to subtract the sky spectrum from
        '''
        sky_spec = np.zeros(header['NAXIS3']) # initialise spectrum
        # apply the sky mask
        if mode=='median':
            for i in range(header['NAXIS3']):
                sky_spec[i] = np.median(sky_mask.get_values(cube[i,:,:]))
        elif mode=='sum':
            for i in range(header['NAXIS3']):
                sky_spec[i] = np.median(sky_mask.get_values(cube[i,:,:]))
        else:
            sys.exit('Unknown method of combining pixels.')
        
        # subtract spectra
        sky_subtracted_spec = spec - sky_spec
        
        return sky_subtracted_spec
    
    
    def extract_spectra(self, sky_subtract=True, plot=True):
        '''
        extracts spectrum from specified regions and save it as fits in a folder
        
        sky_subtract (Bool): saved spectra will be locally sky subtracted if true
        plot (Bool): plot image of the cube, mask and sky mask as sanity check
        '''
        if not os.path.exists(self.data_spectra_path):
            os.mkdir(self.data_spectra_path)
        if not os.path.exists(self.var_spectra_path):
            os.mkdir(self.var_spectra_path)
        
        # get the masks
        masks, sky_masks = self.get_masks_from_regions(self.regions, self.sky_regions)
        
        # apply the mask
        for i, (mask, sky_mask) in enumerate(zip(masks, sky_masks)):
            data_spec_med = np.zeros(self.data_cube_header['NAXIS3'])
            data_spec_sum = np.zeros(self.data_cube_header['NAXIS3'])
            var_spec_med = np.zeros(self.var_cube_header['NAXIS3'])
            var_spec_sum = np.zeros(self.var_cube_header['NAXIS3'])
            
            # initialise arrays for mask-weighted cubes
            mask_weighted_data = np.zeros(shape=self.data_cube_data.shape)
            mask_weighted_var = np.zeros(shape=self.var_cube_data.shape)
            
            # get mask images
            mask_im = mask.to_image(shape=(self.data_cube_header['NAXIS2'], self.data_cube_header['NAXIS1']))
            sky_mask_im = sky_mask.to_image(shape=(self.data_cube_header['NAXIS2'], self.data_cube_header['NAXIS1']))
            
            for j in range(self.data_cube_header['NAXIS3']):
                data_spec_med[j] = np.median(mask.get_values(self.data_cube_data[j,:,:]))
                data_spec_sum[j] = np.sum(mask.get_values(self.data_cube_data[j,:,:]))
                mask_weighted_data[j,:,:] *= mask_im

                var_spec_med[j] = np.median(mask.get_values(self.var_cube_data[j,:,:]))
                var_spec_sum[j] = np.sum(mask.get_values(self.var_cube_data[j,:,:]))
                mask_weighted_var[j,:,:] *= mask_im
            
            if plot:
                self.plot_apertures(self.data_cube_data, self.var_cube_data, mask_im, sky_mask_im, i) 
                
            # apply sky subtraction if sky_subtract=True
            if sky_subtract:
                data_spec_med_skysub = self.sky_subtract(data_spec_med, header=self.data_cube_header,
                                                         cube=self.data_cube_data, sky_mask=sky_mask, mode='median')
                data_spec_sum_skysub = self.sky_subtract(data_spec_sum, header=self.data_cube_header,
                                                         cube=self.data_cube_data, sky_mask=sky_mask, mode='sum')
                var_spec_med_skysub = self.sky_subtract(var_spec_med, header=self.var_cube_header,
                                                         cube=self.var_cube_data, sky_mask=sky_mask, mode='median')
                var_spec_sum_skysub = self.sky_subtract(var_spec_sum, header=self.var_cube_header,
                                                         cube=self.var_cube_data, sky_mask=sky_mask, mode='sum')
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
    
        
    def plot_apertures(self, cube, var_cube, mask_im, sky_mask_im, i=''):
        '''
        plots the median image of the data cube, applied mask, applied sky aperture

        cube (bool): include median image of the original data cube if True
        mask (bool): include image of the aperture from which spectra are extracted if True
        sky (bool): include image of sky area from which sky spectra is extracted if True
        '''
        if not os.path.exists(self.plot_path):
            os.mkdir(self.plot_path)
        
        cube_median = np.median(cube, axis=0)
        var_cube_median = np.median(var_cube, axis=0)
        
        fig, ax = plt.subplots(nrows=1, ncols = 4, figsize=(24,16), subplot_kw={'projection':self.wcs, 'slices':('y', 'x', 200)})
        im0 = ax[0].imshow(cube_median)
        im1 = ax[1].imshow(var_cube_median)
        im2 = ax[2].imshow(mask_im, interpolation='nearest', origin='lower')
        im3 = ax[3].imshow(sky_mask_im, interpolation='nearest', origin='lower')
        
        plt.savefig(''.join([self.plot_path, '/gc{}_extraction_mask.png'.format(i)]))
        plt.close()
        
        return 0


class Noise:
    def __init__(self, cube_path, spec_ending, noise_spec_ending):
        '''
        cube_path (string): path to stacked data and variance cube
        spec_ending (string): filename ending of data_spectra (*file_ending.fits)
        noise_spec_ending (string): filename ending of noise spectra (*file_ending.fits)
        '''
        self.specs = np.sort(glob.glob(''.join([cube_path, '/data_spectra/', spec_ending])))
        self.noise_specs = np.sort(glob.glob(''.join([cube_path, '/var_spectra/', noise_spec_ending])))
        self.denoised_specs_path = ''.join([cube_path, '/denoised_spectra'])
        
        
    def apply_fabada(self, spec, noise_spec):
        '''
        uses fabada to estimate noise and recover noise-reduced spectrum
        '''
        return(fabada(spec, noise_spec))

    
    def save_spec(self, spectrum, header, path):
        '''
        saves a spectrum with the header as fits
        
        spectrum (array): 1D array with spectrum
        header (fits.hdu): fits header data unit
        gc_number: internal index of the gc this spectrum is from
        '''
        if not os.path.exists(self.denoised_specs_path):
            os.mkdir(self.denoised_specs_path)
        
        spec_hdu = fits.PrimaryHDU(spectrum, header)
        spec_hdu.writeto(path, overwrite=True)
        
        return 0

    
    def denoise(self):
        '''
        removes the noise from spectra using fabada
        '''
        for spec, noise_spec in zip(self.specs, self.noise_specs):
            gc_num_spec = re.findall(r'gc[0-9]+', spec)
            gc_num_noise_spec = re.findall(r'gc[0-9]+', noise_spec)
            if gc_num_spec[0] != gc_num_noise_spec[0]:
                sys.exit('Data and variance spectrum do not match.')
            # get the data and noise spectrum
            with fits.open(spec) as hdu:
                spec_data = hdu[0].data
                spec_header = hdu[0].header
            with fits.open(noise_spec) as hdu:
                noise_spec_data = hdu[0].data
                noise_spec_header = hdu[0].header  
            # denoise and save spectrum
            denoised_spec = self.apply_fabada(spec_data, noise_spec_data)
            self.save_spec(denoised_spec, spec_header,
                           ''.join([self.denoised_specs_path, '/{}_denoised.fits'.format(gc_num_spec[0])]))
        
        return 0
    
        

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
            

