'''
Lydia Haacke
05/2023
'''
import glob
import sys
from astropy.io import fits
from MontagePy.main import *

########################## FUNCTIONS #########################################
def compare_central_wavelengths(file_list):
    '''
    Compare central wavelengths across multiple FITS files.
    Does not automatically fix wavelengths.

    Parameters
    ----------
    file_list : list
        list of file paths (to FITS files) to be compared.

    Returns
    -------
    int
        Returns 0 upon successful completion.

    Notes
    -----
    - Files are opened in 'update' mode to allow modifications to headers, but doesn't currently update anything.
    - CRVAL3 is the central wavelength value (wavelength units).
    - CRPIX3 is the reference pixel for the wavelength axis.
    '''
    # get the central wavelength of one cube as reference
    with fits.open(file_list[0]) as hdu:
        crval = hdu[0].header['CRVAL3']
        crpix = hdu[0].header['CRPIX3']
    # compare the central wavelengths for each file, change if slightly different
    for file in file_list:
        with fits.open(file, 'update') as hdu:
            h1 = hdu[0].header
            if h1['CRVAL3'] == crval:
                continue
            else:
                # hdu[0].header['CRVAL3'] = crval
                # hdu[0].header['CRPIX3'] = crpix
                sys.exit('Central wavelengths do not match.')
    return 0


def get_area_ratio(file):
    '''
    Calculate the area ratio of data cube pixels.

    This function computes the ratio between the slice scale (SLSCL) and pixel scale (PXSCL)
    from the FITS file header. This ratio represents how the physical area of a slice
    compares to the area of a single pixel in the data cube.

    Parameters
    ----------
    file : str
        Path to a FITS file containing the data cube.

    Returns
    -------
    float
        The area ratio calculated as SLSCL / PXSCL.

    Notes
    -----
    Requires the 'SLSCL' and 'PXSCL' keywords in primary HDU header
    '''
    with fits.open(file) as hdu:
        h1 = hdu[0].header
        ratio = h1['SLSCL']/h1['PXSCL']
    return ratio


def stack_cubes(path, file_list, output_name):
    '''
    Stack and reproject KCWI data cubes onto a common pixel grid (does bulk of the whole purpose).
    Reprojects data cubes onto a square pixel grid (flux preserving), then combines them into a final mosaic.
    Input cubes should be pre-processed (cut and gradient corrected).

    Parameters
    ----------
    path : str
        Path to directory containing the cubes and subdirectories 
        ('/reproj' subdirectory will be used for intermediate and output files)
    file_list : list of str
        List of file paths to KCWI cubes to reproject and stack
    output_name : str
        Name for the output stacked cube file (without .fits extension)

    Returns
    -------
    int
        Returns 0 on successful completion

    Notes
    -----
    - Requires 'reproj' subdirectory in path for storing reprojected cubes
    - Input cubes should be pre-processed (cut and gradient corrected)
    - Uses Montage mImgtbl, mMakeHdr, mProjectCube, and mAddCube functions
    - Path indexing is configured for specific KCWI file naming conventions
    - Output file saved to: {path}/reproj/{output_name}.fits
    '''
    # original instructions:
    # Create a directory to hold the reprojected images: $ mkdir proj-narrow
    # Create a directory to hold the shrunken images: $ mkdir narrow-shrunk
    # Create a directory to hold the final image mosaic: $ mkdir final

    # use mImgtbl
    imlist = mImgtbl(path, ''.join([path, '/reproj/', 'icubes.tbl']), showCorners=True)
    # imlist = mImgtbl(path, ''.join([path, 'icubes.tbl']), showCorners=True)
    print(imlist)

    # use mMakeHdr
    hdr_temp = mMakeHdr(''.join([path, '/reproj/', 'icubes.tbl']), ''.join([path, '/reproj/', 'icubes.hdr']))
    # hdr_temp = mMakeHdr(''.join([path, 'icubes.tbl']), ''.join([path, 'icubes.hdr']))
    print(hdr_temp)

    # set grid ratio
    arearatio = get_area_ratio(file_list[0])

    # use mProjectCube (for each cube to be added to the final mosaic)
    # !!!! check right formatting for cube & saving path
    for cube in file_list:
        # # 53 for bh3l
        # rep_cube = mProjectCube(cube, ''.join([cube[:53], '/reproj/', cube[54:-5], '_reproj.fits']), ''.join([path, '/reproj/', 'icubes.hdr']), drizzle=1.0, energyMode=False, fluxScale=arearatio)
        # print(''.join([cube[:53], '/reproj/', cube[54:-5], '_reproj.fits']))
        # print(''.join([path, '/reproj/', 'icubes.hdr']))

        # # 57 for bh3l in old folder
        # rep_cube = mProjectCube(cube, ''.join([cube[:57], '/reproj/', cube[58:-5], '_reproj.fits']), ''.join([path, '/reproj/', 'icubes.hdr']), drizzle=1.0, energyMode=False, fluxScale=arearatio)
        # print(''.join([cube[:57], '/reproj/', cube[58:-5], '_reproj.fits']))
        # print(''.join([path, '/reproj/', 'icubes.hdr']))

        # for bh3l with position angle 0
        rep_cube = mProjectCube(cube, ''.join([cube[:71], '/reproj/', cube[72:-5], '_reproj.fits']), ''.join([path, '/reproj/', 'icubes.hdr']), drizzle=1.0, energyMode=False, fluxScale=arearatio)
        print(''.join([cube[:71], '/reproj/', cube[72:-5], '_reproj.fits']))
        print(''.join([path, '/reproj/', 'icubes.hdr']))

        print(rep_cube)
        

    # use mImgtbl again (create image metadata file)
    im_meta = mImgtbl(''.join([path, '/reproj']), ''.join([path, '/reproj/', 'icubes-proj.tbl']), showCorners=True)
    # im_meta = mImgtbl(path, ''.join([path, 'icubes-proj.tbl']), showCorners=True)
    print(im_meta)

    # use mAddCube (coadd reprojected cubes)
    print(''.join([path, '/reproj/', output_name, '.fits']))
    added_cubes = mAddCube(''.join([path, '/reproj/']),
                           ''.join([path, '/reproj/', 'icubes-proj.tbl']),
                           ''.join([path, '/reproj/', 'icubes.hdr']),
                           ''.join([path, '/reproj/', output_name, '.fits']),
                           shrink=True)
    # added_cubes = mAddCube(path, ''.join([path, 'icubes-proj.tbl']),
    #                        ''.join([path, 'icubes.hdr']),
    #                        ''.join([path, '/', output_name, '.fits']),
    #                        shrink=True)
    print(added_cubes)


    # optional: use mViewer (create image of mosaic)


    return 0


def fix_hdr(cube_new, cube_orig):
    '''
    Copy missing keywords from old to new cube (montagepy drops some!!)

    Parameters
    ----------
    cube_new : str
        File path to the new stacked FITS cube whose header will be updated.
    cube_orig : str
        File path to the original FITS cube from which header keywords will be copied.

    Returns
    -------
    int
        Returns 0 upon successful completion.

    Notes
    -----
    The new cube file is opened in 'update' mode and modified in-place.
    Without this code will not recognise that the file is 3 dimensional
    '''
    with fits.open(cube_orig) as hdrorig:
        hdr0 = hdrorig[0].header
        with fits.open(cube_new, 'update') as hdr1:
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

    return 0


############################## RUN ########################################
# !!! possibly very outdated

# # input
# # bh3m
# path = 'path_to_wcs_corrected_cubes'
# file_list = glob.glob(''.join([path, '/', '*_wcscorr.fits']))
# # print(file_list)
# stacked_name = 'stacked_cube_name'

# # # check if all the central wavelengths are the same
# compare_central_wavelengths(file_list)

# # # stack the cubes
# stack_cubes(path, file_list, stacked_name)

# # fix the header of the resulting stack of cubes
# new_cube = ''.join([path, '/reproj/', stacked_name, '.fits'])
# orig_cube = file_list[0]
# fix_hdr(new_cube, orig_cube)

# # fix the header of the wcscorr cubes
# new_cube = ''.join([path,'/reproj/',  ])


