'''
Lydia Haacke
08/2023
'''
import glob
import os
import sys
import math
import re
import numpy as np
from astropy.io import fits
from MontagePy.main import *


class Manipulate_icubes:
    def __init__(self, cube_path, cube_dict):
        self.cube_path = cube_path
        self.cut_cubes_path = ''.join([self.cube_path, 'icubes_cut_1'])
        
        self.data_gradient_corrected_path = ''.join([self.cube_path, 'icubes_gradient_corrected_data_2'])
        self.var_gradient_corrected_path = ''.join([self.cube_path, 'icubes_gradient_corrected_var_2'])
        self.gradient_corrected_path = ''.join([self.cube_path, 'icubes_gradient_corrected_2'])
        
        self.data_wcs_corrected_path = ''.join([self.cube_path, 'icubes_wcs_corrected_data_3'])
        self.var_wcs_corrected_path = ''.join([self.cube_path, 'icubes_wcs_corrected_var_3'])
        self.wcs_corrected_path = ''.join([self.cube_path, 'icubes_wcs_corrected_3'])
        
        self.data_rebinned_path = ''.join([self.cube_path, 'icubes_rebinned_data_4'])
        self.var_rebinned_path = ''.join([self.cube_path, 'icubes_rebinned_var_4'])
        self.rebinned_path = ''.join([self.cube_path, 'icubes_rebinned_4'])
        
        self.cube_dict = cube_dict
        
        self.reference_header = None


    def cut_cubes(self):
        '''
        Cuts overhang pixels and bad wavelengths from FITS data cubes.
        Uses class attributes.

        Returns:
        -----------
        int: Status code (0 on successful completion).

        Notes:
        -----------
        Creates output directory if it doesn't exist.
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
            data_header['CRPIX3'] -= (self.cube_dict[key]['z_border'][0]-1)
            data_header['WAVALL0'] = self.cube_dict[key]['WAVALL0']
            data_header['WAVALL1'] = self.cube_dict[key]['WAVALL1']
            
            var_header['NAXIS1'] = var.shape[0]
            var_header['NAXIS2'] = var.shape[1]
            var_header['NAXIS3'] = var.shape[2]

            # save cubes to new directory
            cut_cube_hdu = fits.PrimaryHDU(cut_cube, data_header)
            cut_cube_vdu = fits.ImageHDU(cut_cube_variance, var_header)
            cut_cube_hdul = fits.HDUList([cut_cube_hdu, cut_cube_vdu])
            cut_cube_hdul.writeto(''.join([self.cut_cubes_path, '/', key, '_icubes_cut.fits']), overwrite=True)
        
        return 0


    def compare_central_wavelengths(self, cut_suff='/*icubes_cut.fits'):
        '''
        Compare central wavelengths across FITS cubes.

        Parameters
        ----------
        cut_suff : str, optional
            Suffix pattern to identify cube files. Default is '/*icubes_cut.fits'.
            Used to glob files in self.cut_cubes_path.
        
        Returns
        -------
        int
            Returns 0 on successful completion.
        
        Raises
        ------
        SystemExit
            If the difference in central wavelengths between any cube and the reference
            exceeds ±1 Angstrom.

        Notes
        -----
        Only use this if you don't expect more than ~1 AA offset in central lambda.
        Sensitivity changes with central wavelength and otherwise you really should take that into consideration.
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
                elif (h1['CRVAL3'] > (crval+1)) or (h1['CRVAL3'] < (crval-1)):
                    sys.exit('Difference in central wavelengths too big')
                else:
                    # this is actually a bad idea, unless you want to only get recessional v
                    # and don't mind the fact that sensitivity changes with central wavelength
                    # depending on observing settings
                    hdu[0].header['CRVAL3'] = crval
                    hdu[0].header['CRPIX3'] = crpix
        return 0


    def add_var_wcs_header(self, var_header, data_header, key):
        '''
        Add mostly WCS related header to variance cube.
        
        Parameters
        ----------
        var_header : astropy.io.fits.Header
            Header object of the variance cube to be updated with WCS information.
        data_header : astropy.io.fits.Header
            Header object of the corresponding data cube containing reference WCS keywords.
        key : str
            Exposure identifier in 'kbyymmdd_xxxxx' format used to access calibration data from self.cube_dict.
        
        Returns
        -------
        astropy.io.fits.Header
            The updated variance cube header with WCS keywords added.
        '''
        # add necessary keywords to variance cube to 'add' a wcs
        var_header['CRPIX1'] = self.cube_dict[key]['xpix']
        var_header['CRPIX2'] = self.cube_dict[key]['ypix']
        var_header['CRPIX3'] = data_header['CRPIX3']
        var_header['CRVAL1'] = self.cube_dict[key]['xval']
        var_header['CRVAL2'] = self.cube_dict[key]['yval']
        var_header['CRVAL3'] = data_header['CRVAL3']
        var_header['CUNIT1'] = data_header['CUNIT1']
        var_header['CUNIT2'] = data_header['CUNIT2']
        var_header['CUNIT3'] = data_header['CUNIT3']
        var_header['CTYPE1'] = data_header['CTYPE1']
        var_header['CTYPE2'] = data_header['CTYPE2']
        var_header['CTYPE3'] = data_header['CTYPE3']
        var_header['CD1_1'] = data_header['CD1_1']
        var_header['CD2_1'] = data_header['CD2_1']
        var_header['CD1_2'] = data_header['CD1_2']
        var_header['CD2_2'] = data_header['CD2_2']
        var_header['CD3_3'] = data_header['CD3_3']
        var_header['BUNIT'] = data_header['BUNIT']
        var_header['WCSDIM'] = data_header['WCSDIM']
        var_header['WCSNAME'] = data_header['WCSNAME']
        var_header['RADESYS'] = data_header['RADESYS']
        var_header['SLSCL'] = data_header['SLSCL']
        var_header['PXSCL'] = data_header['PXSCL']
        var_header['WAVALL0'] = self.cube_dict[key]['WAVALL0']
        var_header['WAVALL1'] = self.cube_dict[key]['WAVALL1']
        var_header['WAVGOOD0'] = data_header['WAVGOOD0']
        var_header['WAVGOOD1'] = data_header['WAVGOOD1']
        
        return var_header


    def gradient_correction(self, cut_suff='/*icubes_cut.fits'):
        '''
        Applies gradient correction to KCWI data cubes along the x-axis.
        (Because there's a natural gradient across the IFU)
       
        Parameters
        ----------
        cut_suff : str, optional
            Glob pattern suffix for identifying cut cubes to process. 
            Default is '/*icubes_cut.fits'.
        
        Returns
        -------
        int
            Returns 0 upon successful completion.
        
        Notes
        -----
        Creates output directories if they do not exist:
          - self.data_gradient_corrected_path for corrected data cubes
          - self.var_gradient_corrected_path for corrected variance cubes
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

    
    def join_cubes(self, data_path, var_path, joint_path, cut_suff):
        '''
        Joins data and variance FITS cubes into a single HDU list after rebinning.
        Reads rebinned data and variance cubes from separate directories, combines them
        into a single FITS file with data in the primary HDU and variance in an image HDU,
        and writes the combined cubes to the joint output directory.
        
        Parameters
        ----------
        data_path : str
            Path to directory containing rebinned data cubes.
        var_path : str
            Path to directory containing rebinned variance cubes.
        joint_path : str
            Path to output directory where combined cubes will be saved (created if not existing).
        cut_suff : str
            Glob pattern suffix to match cube filenames (e.g., '/*.fits').
        
        Returns
        -------
        int
            Returns 0 on successful completion.
        
        Notes
        -----
        Data and variance cubes must have matching names.
        Overwrites existing output files.
        '''
        # check if joint path exists
        if not os.path.exists(joint_path):
            os.mkdir(joint_path)
        
        # data and var cubes have the exact same name
        # gobble and sort both groups results in two matching list where index matches index
        data_cubes = np.sort(glob.glob(''.join([data_path, cut_suff])))
        var_cubes = np.sort(glob.glob(''.join([var_path, cut_suff])))
        for data_cube, var_cube in zip(data_cubes, var_cubes):
            with fits.open(data_cube) as hdu:
                data = hdu[0].data
                data_header = hdu[0].header
            with fits.open(var_cube) as hdu:
                var = hdu[0].data
                var_header = hdu[0].header
        
            # find key using regex
            data_key, var_key = self.get_key(data_cube), self.get_key(var_cube)
        
            # save cubes to new directory in joint format
            cube_hdu = fits.PrimaryHDU(data, data_header)
            cube_vdu = fits.ImageHDU(var, var_header)
            cube_hdul = fits.HDUList([cube_hdu, cube_vdu])
            cube_hdul.writeto(''.join([joint_path, '/', data_key, '_rebinned_joint.fits']), overwrite=True)  
        
        return 0


    def get_area_ratio(self, file, header=True):
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


    def wcs_correction(self, cut_suff='/*gradient_corrected.fits'):
        '''
        Correct the WCS (World Coordinate System) system for FITS cubes based on pixel values.

        Parameters
        ----------
        cut_suff : str, optional
            Glob pattern suffix for identifying input FITS files.
            Default is '/*gradient_corrected.fits'
        
        Returns
        -------
        int
            Returns 0 upon successful completion.
        
        Raises
        ------
        SystemExit
            If data and variance cube keys do not match during processing.
        
        Notes
        -----
        Creates output directories if they do not exist.
        '''
        # check data, var_wcs_corrected directories exist
        # make them if not
        if not os.path.exists(self.data_wcs_corrected_path):
            os.mkdir(self.data_wcs_corrected_path)
        if not os.path.exists(self.var_wcs_corrected_path):
            os.mkdir(self.var_wcs_corrected_path)
            
        # correct the wcs according to the values in self.cube_dict   
        data_cubes = np.sort(glob.glob(''.join([self.data_gradient_corrected_path, cut_suff])))
        var_cubes = np.sort(glob.glob(''.join([self.var_gradient_corrected_path, cut_suff])))
        for data_cube, var_cube in zip(data_cubes, var_cubes):
            data_key, var_key = self.get_key(data_cube), self.get_key(var_cube)
            if data_key != var_key:
                sys.exit('Keys must be the same.')
            with fits.open(data_cube) as hdu:
                data = hdu[0].data
                data_header = hdu[0].header
            with fits.open(var_cube) as hdu:
                var = hdu[0].data
                var_header = hdu[0].header
            
            # save cubes to new directory
            wcs_cube_hdu = fits.PrimaryHDU(data, data_header)
            wcs_cube_vdu = fits.PrimaryHDU(var, var_header)
            wcs_cube_hdu.writeto(''.join([self.data_wcs_corrected_path, '/', data_key, '_wcs_corrected.fits']), overwrite=True)
            wcs_cube_vdu.writeto(''.join([self.var_wcs_corrected_path, '/', var_key, '_wcs_corrected.fits']), overwrite=True)
            
            with fits.open(''.join([self.data_wcs_corrected_path, '/', data_key, '_wcs_corrected.fits']), 'update') as hdu:
                data = hdu[0].data
                data_header = hdu[0].header
                # change the header values in data cube to correct wcs reference
                data_header['CRPIX1'] = self.cube_dict[data_key]['xpix']
                data_header['CRPIX2'] = self.cube_dict[data_key]['ypix']
                data_header['CRVAL1'] = self.cube_dict[data_key]['xval']
                data_header['CRVAL2'] = self.cube_dict[data_key]['yval']
                
            with fits.open(''.join([self.var_wcs_corrected_path, '/', var_key, '_wcs_corrected.fits']), 'update') as hdu:
                var = hdu[0].data
                var_header = hdu[0].header
                # change the header values in variance cube to correct wcs reference
                var_header['CRPIX1'] = self.cube_dict[var_key]['xpix']
                var_header['CRPIX2'] = self.cube_dict[var_key]['ypix']
                var_header['CRVAL1'] = self.cube_dict[var_key]['xval']
                var_header['CRVAL2'] = self.cube_dict[var_key]['yval']
            
            
        # join cubes
        self.join_cubes(self.data_wcs_corrected_path, self.var_wcs_corrected_path,
                        self.wcs_corrected_path, '/*wcs_corrected.fits')

        return 0


    def rebin_cubes(self, header=True, cut_suff='/*wcs_corrected.fits'):
        """
        Rebin KCWI data and variance cubes to a common WCS grid using Montage.

        Parameters
        ----------
        header : bool, optional
            If True, preserves original header information and fixes headers in rebinned 
            cubes. If False, skips header preservation. Default is True.
        cut_suff : str, optional
            Glob pattern suffix to identify WCS-corrected cube files. 
            Default is '/*wcs_corrected.fits'.
       
         Returns
        -------
        int
            Returns 0 upon successful completion.
        
        Notes
        -----
        Based on montagepy using mImgtbl, mMakeHdr, and mProjectCube functions for rebinning
        author confused by the join cubes bit in the end, don't think that's necessary
        will create joint variance and data cubes where each is an extension
        
        Raises
        ------
        SystemExit
            If data cube cannot be matched to corresponding variance cube.
        """
        
        # check if the directories for rebinned cubes exist
        # make if they don't
        if not os.path.exists(self.data_rebinned_path):
            os.mkdir(self.data_rebinned_path)
        if not os.path.exists(self.var_rebinned_path):
            os.mkdir(self.var_rebinned_path)
            
        # Montage pre-processing for data and var separately
        imlist_data = mImgtbl(self.data_wcs_corrected_path, ''.join([self.data_rebinned_path, '/icubes.tbl']), showCorners=True)
        imlist_var = mImgtbl(self.var_wcs_corrected_path, ''.join([self.var_rebinned_path, '/icubes.tbl']), showCorners=True)

        # use mMakeHdr
        hdr_temp_data = mMakeHdr(''.join([self.data_rebinned_path, '/icubes.tbl']), ''.join([self.data_rebinned_path, '/icubes.hdr']))
        hdr_temp_var = mMakeHdr(''.join([self.var_rebinned_path, '/icubes.tbl']), ''.join([self.var_rebinned_path, '/icubes.hdr']))

        # rebin cubes
        data_cubes = np.sort(glob.glob(''.join([self.data_wcs_corrected_path, cut_suff])))
        var_cubes = np.sort(glob.glob(''.join([self.var_wcs_corrected_path, cut_suff])))
        
        # get arearatio and header data
        if header:
            arearatio_data, orig_header_data = self.get_area_ratio(data_cubes[0], header=True)
            arearatio_var, orig_header_var = self.get_area_ratio(var_cubes[0], header=True)
        else:
            arearatio_data = self.get_area_ratio(data_cubes[0])
            arearatio_var = self.get_area_ratio(var_cubes[0])
        for data_cube, var_cube in zip(data_cubes, var_cubes):
            data_key, var_key = self.get_key(data_cube), self.get_key(var_cube)
            if data_key != var_key:
                sys.exit('Matching data cube to wrong variance cube.')
            # reproject data cube
            rep_cube_data = mProjectCube(data_cube,
                                    ''.join([self.data_rebinned_path, '/', data_key, '_reproj.fits']),
                                    ''.join([self.data_rebinned_path, '/icubes.hdr']),
                                    drizzle=1.0, energyMode=False, fluxScale=arearatio_data)
            # reproject variance cube
            rep_cube_var = mProjectCube(var_cube,
                                    ''.join([self.var_rebinned_path, '/', var_key, '_reproj.fits']),
                                    ''.join([self.var_rebinned_path, '/icubes.hdr']),
                                    drizzle=1.0, energyMode=False, fluxScale=arearatio_var)                        
            if header:
                self.fix_rebinned_hdr(''.join([self.data_rebinned_path, '/', data_key, '_reproj.fits']), orig_header_data)
                self.fix_rebinned_hdr(''.join([self.var_rebinned_path, '/', var_key, '_reproj.fits']), orig_header_var)

        # fix the header of rebinned cubes

        return 0


        # join cubes
        # self.join_cubes(self.data_rebinned_path, self.var_rebinned_path,
        #                 self.rebinned_path, '/*reproj.fits')

        # return 0


    def stack_cubes(self, stacked_cubes_name):
        '''
        Stack all rebinned data and variance cubes into single FITS files.

        Parameters
        ----------
        stacked_cubes_name : str
            Filename of the output stacked cubes FITS file (e.g., 'stacked.fits').
        
        Returns
        -------
        int
            Returns 0 upon successful completion.
        
        Notes
        -----
        commenting in join_cubes will create joint data and variance file
        '''
        # create image metadata table for reprojected data cubes
        im_meta_data = mImgtbl(self.data_rebinned_path,
                        ''.join([self.data_rebinned_path, '/icubes-proj.tbl']), showCorners=True)
        print(im_meta_data)
        # create image metadata table for reprojected variance cubes
        im_meta_var = mImgtbl(self.var_rebinned_path,
                        ''.join([self.var_rebinned_path, '/icubes-proj.tbl']), showCorners=True)
        print(im_meta_var)
        
        # actually add reprojected data cubes
        added_cubes_data = mAddCube(''.join([self.data_rebinned_path, '/']),
                            ''.join([self.data_rebinned_path, '/icubes-proj.tbl']),
                            ''.join([self.data_rebinned_path, '/icubes.hdr']),
                            ''.join([self.cube_path, 'data_', stacked_cubes_name]),
                            shrink=True)
        print(added_cubes_data)
        added_cubes_var = mAddCube(''.join([self.var_rebinned_path, '/']),
                            ''.join([self.var_rebinned_path, '/icubes-proj.tbl']),
                            ''.join([self.var_rebinned_path, '/icubes.hdr']),
                            ''.join([self.cube_path, 'var_', stacked_cubes_name]),
                            shrink=True)
        print(added_cubes_var)


        # update stacked data cube hdr
        orig_headers = glob.glob(''.join([self.data_wcs_corrected_path, '/*.fits'])) # get header with original axis data
        with fits.open(orig_headers[0]) as hdu:
            h1_stacked_template = hdu[0].header
        self.fix_stacked_hdr(''.join([self.cube_path, 'var_', stacked_cubes_name]), h1_stacked_template)
        self.fix_stacked_hdr(''.join([self.cube_path, 'data_', stacked_cubes_name]), h1_stacked_template)

        # join cubes
        # self.join_cubes(''.join([self.cube_path, 'data_', stacked_cubes_name]), ''.join([self.cube_path, 'var_', stacked_cubes_name]),
        #                 self.cube_path, stacked_cubes_name)

        return 0


    def fix_stacked_hdr(self, cube, hdr0):
        """
        Fix and synchronize header keywords of a stacked FITS cube.

        This method updates specific header keywords in a stacked data cube FITS file
        by copying corresponding values from a reference header. It ensures that the
        stacked cube contains all necessary wavelength and unit-related metadata.

        Parameters
        ----------
        cube : str
            Filepath to the stacked FITS cube file to be updated.
        hdr0 : astropy.io.fits.Header
            Reference header object containing source keyword values.

        Returns
        -------
        int
            Returns 0 upon successful completion.

        Notes
        -----
        The method updates the following header keywords in the stacked cube:
        - CUNIT3, CTYPE3: Wavelength axis unit and type information
        - WAVALL0, WAVALL1: Wavelength coverage range
        - CDELT3: Wavelength pixel scale (copied from CD3_3)
        - WAVGOOD0, WAVGOOD1: Valid wavelength range
        - BUNIT: Brightness unit of the data

        The file is opened in 'update' mode to preserve other header information
        while modifying only the specified keywords.

        Examples
        --------
        >>> from astropy.io import fits
        >>> hdr_ref = fits.getheader('reference_cube.fits')
        >>> fixer = CubeFixerClass()  # Assuming this method belongs to a class
        >>> fixer.fix_stacked_hdr('stacked_cube.fits', hdr_ref)
        0
        """
        '''
        add all necessary keywords to header of stacked cube
        '''
        with fits.open(cube, 'update') as hdr1:
            hdr1[0].header['CUNIT3'] = hdr0['CUNIT3']
            hdr1[0].header['CTYPE3'] = hdr0['CTYPE3']
            hdr1[0].header['WAVALL0'] = hdr0['WAVALL0']
            hdr1[0].header['WAVALL1'] = hdr0['WAVALL1']
            hdr1[0].header['CDELT3'] = hdr0['CD3_3']
            hdr1[0].header['WAVGOOD0'] = hdr0['WAVGOOD0']
            hdr1[0].header['WAVGOOD1'] = hdr0['WAVGOOD1']
            hdr1[0].header['BUNIT'] = hdr0['BUNIT']

        return 0


    def get_key(self, cube):
        '''
        get the kbyymmdd_xxxxx string out of a file path
        
        cube: filepath to a certain data cube
        '''
        num = re.findall(r'[0-9]+_[0-9]+', cube)
        key = ''.join(['kb', num[0]])
        
        return key

        