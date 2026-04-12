# kcwi_spectra_reduction
Semi-automatic pipeline to extract and process spectra from KCWI cubes

# Use
Run based on an input file which contains dictionary-style information on the data cubes to be combined. Keywords needed:

'x_border': the x-pixels to be included in the combined cube, python style (inclusive, exclusive)

'y_border': the y-pixels to be included in the combined cube, python style (inclusive, exclusive)

'z_border' (wavelength axis): the z-pixels to be included in the combined cube, python style (inclusive, exclusive)

'WAVALL0': lower wavelength range limit

'WAVALL1': upper wavelength range limit

'xpix': centre pixel value x dimension

'ypix': centre pixel value y dimension

'xval': actual RA value of the pixel specified in xpix

'yval': actual DEC value of the pixel specified in ypix

### Dict example
bh3m_cube_dict = {

    'cube1':{
    
        'x_border':(1, 24), 'y_border':(4, 70), 'z_border':(212, 2112),
        'WAVALL0':4800, 'WAVALL1':5300,
        'xpix':6.7238, 'ypix':32.065, 'xval':221.3356655, 'yval':1.811620
    },
    
    'cube1':{
        'x_border':(1, 24), 'y_border':(4, 70), 'z_border':(212, 2112),
        'WAVALL0':4800, 'WAVALL1':5300,
        'xpix':6.7339, 'ypix':33.4, 'xval':226.3356655, 'yval':1.811620
    }
    
}
