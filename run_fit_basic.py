from ppxf_fit_kinematics_4 import *

import glob

########################## input ################################
z = [0., 0.001, 0.0015, 0.0064, 0.0068, 0.0072, 0.0076, 0.008] # 8 redshifts a ~ 2 hrs each
grating = ('BH3_Large', 'yale')
gc_list = ['gc14']
shift_spec = True
fit = True

######################### run ###################################
for gc in gc_list:
    spectrum = '../air_wavelengths/bh3l_yale/udg1_1/{}_spectra.fits'.format(gc)
    save_as = '../air_wavelengths/bh3l_yale/udg1_1/{}_spectra.txt'.format(gc)
    fit_vel_sigma(spectrum, save_as, z, grating, shift_spec=shift_spec, fit=fit)
  
