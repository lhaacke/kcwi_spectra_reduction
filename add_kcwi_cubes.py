'''
Lydia Haacke
05/2023
'''
import glob
from astropy.io import fits
from MontagePy.main import *

########################## FUNCTIONS #########################################
def compare_central_wavelengths(file_list):
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


def get_area_ratio(file):
    '''
    calculate the area ratio of data cube pixels
    file: path to one of the fits files
    '''
    with fits.open(file) as hdu:
        h1 = hdu[0].header
        ratio = h1['SLSCL']/h1['PXSCL']
    # with fits.open(file) as hdu:
    #     h1 = hdu[0].header
    #     ratio = h1['PXSCL']/h1['SLSCL']
    return ratio


def stack_cubes(path, file_list, output_name):
    '''
    reproject KCWI cubes onto a square pixel grid, flux preserving
    cubes should be cut and gradient corrected

    path: path to directory containing cubes
    file_list: list of cubes to reproject and stack
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
    add necessary keywords to stacked cubes' header
    (heavily based on Nikki Nielsen's function)
    '''
    with fits.open(cube_orig) as hdrorig:
        hdr0 = hdrorig[0].header
        print(hdr0['CRVAL3'])
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

    # newcube = fits.PrimaryHDU(cube_new,hdr1[0].header)
    # newcube.writeto(cube_new,overwrite=True)

    return 0


############################## RUN ########################################
# !!! 

# # input
# # bh3m
# path = '../stacked_cubes_spectra/bh3m_cut_cubes/wcs_corrected'
# file_list = glob.glob(''.join([path, '/', '*_wcscorr.fits']))
# # print(file_list)
# stacked_name = 'NGC5846_UDG1_BH3M_mosaic_invratio'

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


# input
# # bh3l
# path = '../stacked_cubes_spectra/bh3l_cut_cubes/old/wcs_corrected/reproj_posang'
# file_list = glob.glob(''.join([path, '/', '*_wcscorr.fits']))
# # print(file_list)
# stacked_name = 'NGC5846_UDG1_BH3L_mosaic'

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

