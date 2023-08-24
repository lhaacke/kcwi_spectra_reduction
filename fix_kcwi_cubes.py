'''
Lydia Haacke
05/2023
'''
import glob
import os
import math
import numpy as np
from astropy.io import fits


class manipulate_icubes:
    def __init__(self, icubes_path, cube_dict):
        self.cube_path = icubes_path
        self.cut_cubes_path = ''.join([self.cube_path, 'icubes_cut'])
        self.gradient_corrected_path = ''.join([self.cube_path, 'icubes_gradient_corrected'])
        self.rebinned_cubes_path = ''.join([self.cube_path, 'icubes_rebinned'])
        self.cube_dict = cube_dict

    def cut_cubes(self):
        '''
        cube_path: path to cubes that need to be cut
        cube_dict: dictionary with info on cubes (pixels to be included)
        '''
        # check if the directory for the cut cubes is there
        # make cut cube directory if it is not
        if not os.path.exists(self.cut_cubes_path):
            os.mkdir(self.cut_cubes_path)
            
        # cut the cubes and save the cut cube into the cut cube directory 
        cubes = glob.glob(''.join([self.cube_path, '*icubes.fits']))
        for cube in cubes:
            key = cube[-26:-12]
            with fits.open(cube) as hdu:
                hdu.info()
                data = hdu[0].data
                data_header = hdu[0].header
                var = hdu[2].data
                var_header = hdu[2].header
                
                # cut data and variance cube
                cut_cube = data[self.cube_dict[key]['z_border'][0]-1:self.cube_dict[key]['z_border'][1],
                                self.cube_dict[key]['y_border'][0]-1:self.cube_dict[key]['y_border'][1],
                                self.cube_dict[key]['x_border'][0]-1:self.cube_dict[key]['x_border'][1]]
                cut_cube_variance = var[self.cube_dict[key]['z_border'][0]-1:self.cube_dict[key]['z_border'][1],
                                        self.cube_dict[key]['y_border'][0]-1:self.cube_dict[key]['y_border'][1],
                                        self.cube_dict[key]['x_border'][0]-1:self.cube_dict[key]['x_border'][1]]
                
                # correct header keywords to avoid confusing qfits view
                data_header['NAXIS1'] = data.shape[0]
                data_header['NAXIS2'] = data.shape[1]
                data_header['NAXIS3'] = data.shape[2]
                data_header['WAVALL0'] = math.ceil(data_header['WAVGOOD0'])
                data_header['WAVALL1'] = math.floor(data_header['WAVGOOD1'])
                var_header['NAXIS1'] = var.shape[0]
                var_header['NAXIS2'] = var.shape[1]
                var_header['NAXIS3'] = var.shape[2]
                
                # save cubes to new directory
                cut_cube_hdu = fits.PrimaryHDU(cut_cube, data_header)
                cut_cube_vdu = fits.ImageHDU(cut_cube_variance, var_header)
                cut_cube_hdul = fits.HDUList([cut_cube_hdu, cut_cube_vdu])
                cut_cube_hdul.writeto(''.join([self.cut_cubes_path, '/', key, '_icubes_cut.fits']), overwrite=True)
        return 0


    def gradient_correction(self):
        '''
        cut_cube_path: path where to find the cut icubes that need to be corrected
        '''
        # check if gradient_corrected path exists or not
        if not os.path.exists(self.gradient_corrected_path):
            os.mkdir(self.gradient_corrected_path)

        # correct the gradient along the x_axis
        cut_suff = '/*icubes_cut.fits'
        key_len = 14
        cubes = glob.glob(''.join([self.cut_cubes_path, cut_suff]))
        for cube in cubes:
            key = cube[-(len(cut_suff) + key_len):-(len(cut_suff) - 1)]
            with fits.open(cube) as hdu:
                data = hdu[0].data
                data_header = hdu[0].header
                data_med = np.median(data, axis=1)
                var = hdu[1].data
                var_header = hdu[1].header
#                 var_med = np.median(var, axis=1)
            for yind in range(data_header['NAXIS2']):
                data[:, yind, :] = data[:, yind, :]/data_med # data cube
#                 var[:, yind, :] = var[:, yind, :]/var_med # variance cube

            # save cubes to new directory
            cube_hdu = fits.PrimaryHDU(data, data_header)
            cube_vdu = fits.ImageHDU(var, var_header)
            cube_hdul = fits.HDUList([cube_hdu, cube_vdu])
            cube_hdul.writeto(''.join([self.gradient_corrected_path, '/', key, '_gradient_corrected.fits']), overwrite=True)

        return 0


    def compare_central_wavelengths(self):
        '''
        checks if all the central wavelengths are the same
        '''
        # get all the cubes in gradient corrected folder  
        cut_suff = '/*gradient_corrected.fits'
        key_len = 14
        cubes = glob.glob(''.join([self.gradient_corrected_path, cut_suff]))
        
        # get the central wavelength of one cube as reference
        with fits.open(cubes[0]) as hdu:
            crval = hdu[0].header['CRVAL3']
            crpix = hdu[0].header['CRPIX3']
        
        # compare the central wavelengths for each file, change if slightly different
        for cube in cubes:
            key = cube[-(len(cut_suff) + key_len - 1):-(len(cut_suff)-1)]
            with fits.open(cube, 'update') as hdu:
                h1 = hdu[0].header
                if h1['CRVAL3'] == crval:
                    continue
                else:
                    hdu[0].header['CRVAL3'] = crval
                    hdu[0].header['CRPIX3'] = crpix
        return 0


    def rebin_cubes(self):
        '''
        '''
        # check if the directory for the rebinned cubes is there
        # make cut cube directory if it is not
        if not os.path.exists(self.rebinned_cubes_path):
            os.mkdir(self.rebinned_cubes_path)
            
        # Montage pre-processing
        imlist = mImgtbl(self.gradient_corrected_path, ''.join([self.rebinned_cubes_path, '/icubes.tbl']), showCorners=True)
        print(imlist)

        # use mMakeHdr
        hdr_temp = mMakeHdr(''.join([self.rebinned_cubes_path, '/icubes.tbl']), ''.join([self.rebinned_cubes_path '/icubes.hdr']))
        print(hdr_temp)

        # rebin cubes
        cut_suff = '/*gradient_corrected.fits'
        key_len = 14
        cubes = glob.glob(''.join([self.gradient_corrected_path, cut_suff]))
        arearatio = self.get_area_ratio(cubes[0])
        for cube in cubes:
            key = cube[-(len(cut_suff) + key_len - 1):-(len(cut_suff)-1)]
            rep_cube = mProjectCube(cube,
                                    ''.join([rebinned_cubes_path, '/', key, '_reproj.fits']),
                                    ''.join([self.rebinned_cubes_path '/icubes.hdr']),
                                    drizzle=1.0, energyMode=False, fluxScale=arearatio)
            print(''.join([rebinned_cubes_path, '/', key, '_reproj.fits']))
            print(''.join([self.rebinned_cubes_path '/icubes.hdr']))
            print(rep_cube)
    

    def stack_cubes(self, stacked_cubes_name):
        '''
        stacks all rebinned cubes in self.rebinned_cubes_path

        stacked_cubes_name: filename of the stacked cubes fits file (e.g. 'stacked.fits')
        '''
        # create image metadata table for reprojected cubes
        im_meta = mImgtbl(self.rebinned_cubes_path,
                        ''.join([self.rebinned_cubes_path, '/icubes-proj.tbl']), showCorners=True)
        print(im_meta)
        
        # actually add reprojected cubes
        print(''.join([path, '/reproj/', output_name, '.fits']))
        added_cubes = mAddCube(''.join([self.rebinned_cubes_path, '/']),
                            ''.join([self.rebinned_cubes_path, '/icubes-proj.tbl']),
                            ''.join([self.rebinned_cubes_path, '/icubes.hdr']),
                            ''.join([self.rebinned_cubes_path, '/', stacked_cubes_name]),
                            shrink=True)
        print(added_cubes)


    def wcs_correction(self):
        '''
        '''
        # check if wcs_corrected path exists or not
        if not os.path.exists(self.wcs_corrected_path):
            os.mkdir(self.wcs_corrected_path)
            
        # correct the wcs according to the values in self.cube_dict   
        cut_suff = '/*gradient_corrected.fits'
        key_len = 14
        cubes = glob.glob(''.join([self.gradient_corrected_path, cut_suff]))
        for cube in cubes:
            key = cube[-(len(cut_suff) + key_len - 1):-(len(cut_suff)-1)]
            with fits.open(cube) as hdu:
                data = hdu[0].data
                data_header = hdu[0].header
                data_med = np.median(data, axis=1)
                var = hdu[1].data
                var_header = hdu[1].header
            
            # change the header values in data cube to correct wcs reference
            data_header['CRPIX1'] = self.cube_dict[key]['xpix']
            data_header['CRPIX2'] = self.cube_dict[key]['ypix']
            data_header['CRVAL1'] = self.cube_dict[key]['xval']
            data_header['CRVAL2'] = self.cube_dict[key]['yval']
            
            # add necessary keywords to variance cube to 'add' a wcs
            var_header['CRPIX1'] = self.cube_dict[key]['xpix']
            var_header['CRPIX2'] = self.cube_dict[key]['ypix']
            var_header['CRPIX3'] = data_header['CRPIX3']
            var_header['CRVAL1'] = self.cube_dict[key]['xval']
            var_header['CRVAL2'] = self.cube_dict[key]['yval']
            var_header['CRVAL3'] = data_header['CRVAL3']
            #!!!!!! the ones below need to be adjusted to work with the rebinned cubes !!!!!!!!!!!!!
#             var_header['CRDELT1'] = data_header['CRDELT1']
#             var_header['CRDELT2'] = data_header['CRDELT2']
#             var_header['CRDELT3'] = data_header['CRDELT3']
            var_header['WCSDIM'] = data_header['WCSDIM']
            var_header['WCSNAME'] = data_header['WCSNAME']
            var_header['RADESYS'] = data_header['RADESYS']
        
                
        # save cubes to new directory
        wcs_cube_hdu = fits.PrimaryHDU(data, data_header)
        wcs_cube_vdu = fits.ImageHDU(var, var_header)
        wcs_cube_hdul = fits.HDUList([wcs_cube_hdu, wcs_cube_vdu])
        wcs_cube_hdul.writeto(''.join([self.wcs_corrected_path, '/', key, '_wcs_corrected.fits']), overwrite=True)

        return 0

        