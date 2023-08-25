'''
Lydia Haacke
05/2023
'''
import glob
import os
import math
import re
import numpy as np
from astropy.io import fits
from MontagePy.main import *


class manipulate_icubes:
    def __init__(self, cube_path, cube_dict):
        self.cube_path = cube_path
        self.cut_cubes_path = ''.join([self.cube_path, 'icubes_cut_1'])
        
        self.data_gradient_corrected_path = ''.join([self.cube_path, 'icubes_gradient_corrected_data_2'])
        self.var_gradient_corrected_path = ''.join([self.cube_path, 'icubes_gradient_corrected_var_2'])
        self.gradient_corrected_path = ''.join([self.cube_path, 'icubes_gradient_corrected_2'])
        
        self.data_rebinned_path = ''.join([self.cube_path, 'icubes_rebinned_data_3'])
        self.var_rebinned_path = ''.join([self.cube_path, 'icubes_rebinned_var_3'])
        self.rebinned_joint_path = ''.join([self.cube_path, 'icubes_rebinned_3'])
        
        self.data_wcs_corrected_path = ''.join([self.cube_path, 'wcs_corrected_data_4'])
        self.var_wcs_corrected_path = ''.join([self.cube_path, 'wcs_corrected_var_4'])
        self.wcs_corrected_path = ''.join([self.cube_path, 'wcs_corrected_4'])
        
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
            key = self.get_key(cube)
            with fits.open(cube) as hdu:
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

            # correct header keywords
            data_header['NAXIS1'] = data.shape[0]
            data_header['NAXIS2'] = data.shape[1]
            data_header['NAXIS3'] = data.shape[2]
            data_header['CRPIX3'] -= self.cube_dict[key]['z_border'][0]
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


    def compare_central_wavelengths(self):
        '''
        checks if all the central wavelengths are the same
        '''
        # get all the cubes in gradient corrected folder  
        cubes = glob.glob(''.join([self.cut_cubes_path, cut_suff]))
        
        # get the central wavelength of one cube as reference
        with fits.open(cubes[0]) as hdu:
            crval = hdu[0].header['CRVAL3']
            crpix = hdu[0].header['CRPIX3']
        
        # compare the central wavelengths for each file, change if slightly different
        for cube in cubes:
            key = self.get_key(cube)
            with fits.open(cube, 'update') as hdu:
                h1 = hdu[0].header
                if h1['CRVAL3'] == crval:
                    continue
                else:
                    hdu[0].header['CRVAL3'] = crval
                    hdu[0].header['CRPIX3'] = crpix
        return 0


    def gradient_correction(self):
        '''
        cut_cube_path: path where to find the cut icubes that need to be corrected
        '''
        # check if gradient_corrected path exists or not
        if not os.path.exists(self.data_gradient_corrected_path):
            os.mkdir(self.data_gradient_corrected_path)
        if not os.path.exists(self.var_gradient_corrected_path):
            os.mkdir(self.var_gradient_corrected_path)

        # correct the gradient along the x_axis
        cubes = glob.glob(''.join([self.cut_cubes_path, cut_suff]))
        for cube in cubes:
            key = self.get_key(cube)
            with fits.open(cube) as hdu:
                data = hdu[0].data
                data_header = hdu[0].header
                data_med = np.median(data, axis=1)
                var = hdu[1].data
                var_header = hdu[1].header
            for yind in range(data_header['NAXIS2']):
                data[:, yind, :] = data[:, yind, :]/data_med # data cube
                var[:, yind, :] = var[:, yind, :]/data_med # variance cube
                
            var_header_wcs = self.add_var_wcs_header(var_header, data_header, key)

            # save cubes to new directory
            cube_hdu = fits.PrimaryHDU(data, data_header)
            cube_vdu = fits.PrimaryHDU(var, var_header_wcs)
            cube_hdu.writeto(''.join([self.data_gradient_corrected_path, '/', key, '_gradient_corrected.fits']), overwrite=True)
            cube_vdu.writeto(''.join([self.var_gradient_corrected_path, '/', key, '_gradient_corrected.fits']), overwrite=True)
            
        # join cubes
        self.join_cubes(self.data_gradient_corrected_path, self.var_gradient_corrected_path,
                        self.gradient_corrected_path, '/*gradient_corrected.fits')
            
        return 0

    
    def join_cubes(self, data_path, var_path, joint_path, suff):
        '''
        joins data and variance cube after rebinning
        takes cubes from self.data_rebinned_path and self.var_rebinned_path
        '''
        # check if joint path exists
        if not os.path.exists(joint_path):
            os.mkdir(joint_path)
        
        # data and var cubes have the exact same name
        # gobble and sort both groups results in two matching list where index matches index
        data_cubes = np.sort(glob.glob(''.join([data_path, suff])))
        var_cubes = np.sort(glob.glob(''.join([var_path, suff])))
        for i, cube in enumerate(data_cubes):
            with fits.open(cube) as hdu:
                data = hdu[0].data
                data_header = hdu[0].header
            with fits.open(var_cubes[i]) as hdu:
                var = hdu[0].data
                var_header = hdu[0].header
        
            # find key using regex
            key = self.get_key(cube)
        
            # save cubes to new directory in joint format
            cube_hdu = fits.PrimaryHDU(data, data_header)
            cube_vdu = fits.ImageHDU(var, var_header)
            cube_hdul = fits.HDUList([cube_hdu, cube_vdu])
            cube_hdul.writeto(''.join([joint_path, '/', key, '_rebinned_joint.fits']), overwrite=True)  
        
        return 0


    def get_area_ratio(self, file, header=False):
        '''
        calculate the area ratio of data cube pixels
        file: path to one of the fits files
        header: whether or not to return the header of the fits file
        '''
        with fits.open(file) as hdu:
            h1 = hdu[0].header
            ratio = h1['SLSCL']/h1['PXSCL']
        if header:
            return ratio, h1
        else:
            return ratio


    def fix_rebinned_hdr(self, cube, hdr0):
        '''
        add necessary keywords to stacked cubes' header
        (heavily based on Nikki Nielsen's function)

        cube: cube where keywords need to be added to the header
        hdr0: header of an original cube containing the keywords
        '''
        with fits.open(cube, 'update') as hdr1:
                hdr1[0].header['WAVALL0'] = hdr0['WAVALL0']
                hdr1[0].header['WAVALL1'] = hdr0['WAVALL1']
                hdr1[0].header['WAVGOOD0'] = hdr0['WAVGOOD0']
                hdr1[0].header['WAVGOOD1'] = hdr0['WAVGOOD1']
                hdr1[0].header['CRVAL3'] = hdr0['CRVAL3']
                hdr1[0].header['CRPIX3'] = hdr0['CRPIX3']
                hdr1[0].header['CUNIT3'] = hdr0['CUNIT3']
                hdr1[0].header['CTYPE3'] = hdr0['CTYPE3']
                hdr1[0].header['CDELT3'] = hdr0['CD3_3']
                hdr1[0].header['BUNIT'] = hdr0['BUNIT']
                hdr1[0].header['WCSDIM'] = hdr0['WCSDIM']
                hdr1[0].header['WCSNAME'] = hdr0['WCSNAME']
                hdr1[0].header['RADESYS'] = hdr0['RADESYS']
        return 0


    def rebin_cubes(self, header=True):
        '''
        rebin the cubes from rectangular to square pixels

        header: whether or not to return the header from get_area_ratio
        '''
        # check if the directory for the rebinned cubes is there
        # make cut cube directory if it is not
        if not os.path.exists(self.rebinned_cubes_path):
            os.mkdir(self.rebinned_cubes_path)
            
        # Montage pre-processing
        imlist = mImgtbl(self.gradient_corrected_path, ''.join([self.rebinned_cubes_path, '/icubes.tbl']), showCorners=True)
        print(imlist)

        # use mMakeHdr
        hdr_temp = mMakeHdr(''.join([self.rebinned_cubes_path, '/icubes.tbl']), ''.join([self.rebinned_cubes_path, '/icubes.hdr']))
        print(hdr_temp)

        # rebin cubes
        cut_suff = '/*gradient_corrected.fits'
        key_len = 14
        cubes = glob.glob(''.join([self.gradient_corrected_path, cut_suff]))
        if header:
            arearatio, orig_header = self.get_area_ratio(cubes[0], header=True)
        else:
            arearatio = self.get_area_ratio(cubes[0])
        for cube in cubes:
            key = cube[-(len(cut_suff) + key_len - 1):-(len(cut_suff)-1)]
            # reproject and fix the header of the data cube
            rep_cube = mProjectCube(cube,
                                    ''.join([self.rebinned_cubes_path, '/', key, '_reproj.fits']),
                                    ''.join([self.rebinned_cubes_path, '/icubes.hdr']),
                                    drizzle=1.0, energyMode=False, fluxScale=arearatio)
            if header:
                self.fix_rebinned_hdr(cube, orig_header)

            print(rep_cube)

        # fix the header of rebinned cubes
    

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


    def get_key(self, cube):
        '''
        get the kbyymmdd_xxxxx string out of a file path
        
        cube: filepath to a certain data cube
        '''
        num = re.findall(r'[0-9]+_[0-9]+', cube)
        key = ''.join(['kb', num[0]])
        
        return key

        