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

    def __init__(self, path_to_cube, path_to_save, ra_list, dec_list, size):
        '''
        path_to_cube: filepath to data cube
        path_to_save: path to directory where files are saved. Subdirectories added in corresponding function where needed
        ra/dec_list: iterable containing ra and dec of wanted targets
        size: (semimajor, semiminor, position angle of semimajor) all in angular units
        '''
        with fits.open(path_to_cube) as hdu:
            self.cube = hdu[0].data
            self.header = hdu[0].header
            self.wcs = WCS(hdu[0].header)
        # initialise dictionary with information on each object
        # make the regions
        self.coords = SkyCoord(ra=ra_list, dec=dec_list, unit=(u.degree, u.degree))
        a, b, theta = size
        self.regions = SkyEllipticalAperture(self.coords, a, b, theta)
        # get path where to save files
        self.save_dir = path_to_save


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
        wcs_mask = get_mask_wcs()

        pixel_aperture = apertures.to_pixel(wcs=wcs_mask)
        pixel_mask = pixel_aperture.to_mask(method='exact') # mask in pixel coordinates

        return pixel_mask


    def get_mask_wcs(self):
        '''
        returns two-dimensional wcs based on self.wcs
        '''
        wcs_mask = self.wcs.dropaxis(dropax=2)
        return wcs_mask
        
