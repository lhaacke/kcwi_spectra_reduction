'''
Lydia Haacke
05/2023
'''
import glob
import os
import numpy as np
from astropy.io import fits


class manipulate_icubes:
    def __init__(self, icubes_path, cube_dict):
        self.cube_path = icubes_path
        self.cut_cube_path = ''.join([self.cube_path, 'icubes_cut'])
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
                var_header['NAXIS1'] = var.shape[0]
                var_header['NAXIS2'] = var.shape[1]
                var_header['NAXIS3'] = var.shape[2]
                
                # save cubes to new directory
                cut_cube_hdu = fits.PrimaryHDU(cut_cube, data_header)
                cut_cube_vdu = fits.ImageHDU(cut_cube_variance, var_header)
                cut_cube_hdul = fits.HDUList([cut_cube_hdu, cut_cube_vdu])
                cut_cube_hdul.writeto(''.join([self.cut_cubes_path, '/', key, '_icubes_cut.fits']), overwrite=True)
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


    



def hdu_wavelength_correction(file_location, wavelengths=(), save_location=None, owrite=True):
    '''
    file_location: location of the file with header to be edited
    save_location: location where to save file with edited header, this can be the same file as file_location if owrite=True
    wavelengths: the wavelengths that should be in WAVALL0 and WAVALL1 in format (WAVALL0, WAVALL1) or other iterable
    owrite: whether to overwrite the 
    '''
    with fits.open(file_location, 'update') as hdu:
        hdu[0].header['WAVALL0'] = wavelengths[0]
        hdu[0].header['WAVALL1'] = wavelengths[1]
    if save_location:
        # tba: saving in different location than original
        i = 0
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

