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
        self.cut_cube_path = ''.join([self.cube_path, 'icubes_cut'])
        self.gradient_corrected_path = ''.join([self.cube_path, 'icubes_gradient_corrected'])
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

    def compare_central_wavelengths(self, file_list):
        '''
        checks if all the central wavelengths are the same
        '''
        # get the central wavelength of one cube as reference
        with fits.open(file_list[0]) as hdu:
            crval = hdu[0].header['CRVAL3']
            crpix = hdu[0].header['CRPIX3']
            print(crval)
        # compare the central wavelengths for each file, change if slightly different
        print(file_list)
        for file in file_list:
            with fits.open(file, 'update') as hdu:
                h1 = hdu[0].header
                if h1['CRVAL3'] == crval:
                    continue
                else:
                    hdu[0].header['CRVAL3'] = crval
                    hdu[0].header['CRPIX3'] = crpix
                    # sys.exit('Central wavelengths do not match.')
        return 0


def hdu_wavelength_correction_dict(file_location, save_location, sfxs, sfxs_fixed, corrs, owrite=True):
    '''
    corrs: dictionary with header data in the running part of the code
    # this needs some adjustment to work without the dictionary structure
    '''
    for key in corrs.keys():
        path = ''.join([file_location, '/', key, sfxs])
        save_as = ''.join([save_location, '/', key, sfxs_fixed])
        with fits.open(path) as hdu: # get data from cube
            data = hdu[0].data
        newfile = fits.PrimaryHDU(data, hdu[0].header) # write to new fits file
        newfile.writeto(save_as, overwrite=owrite)
        print(corrs.keys())
        with fits.open(save_as, 'update') as hdu:
            hdu[0].header['WAVALL0'] = corrs[key]['wavall'][0]
            hdu[0].header['WAVALL1'] = corrs[key]['wavall'][1]
            hdu[0].header['WAVGOOD0'] = corrs[key]['wavgood'][0]
            hdu[0].header['WAVGOOD1'] = corrs[key]['wavgood'][1]
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

