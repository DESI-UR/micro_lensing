import os
import fitsio
import random
import numpy as np
import healpy as hp
from glob import glob
from collections import defaultdict
from desitarget import desi_mask
import matplotlib.pyplot as plt

os.environ["DESI_SPECTRO_REDUX"] = "/home/tyapici/data/DESI/spectro/redux/"
os.environ["SPECPROD"] = "dc17a2"

basedir = os.path.join(os.getenv("DESI_SPECTRO_REDUX"),os.getenv("SPECPROD"),"spectra-64")
specfilenames = glob(basedir+"/*/*/spectra*")

def get_spectrum(file_idx=0, source_idx=0, output=False):
    specfilename = specfilenames[file_idx]
    bwave = fitsio.read(specfilename,"B_WAVELENGTH")
    rwave = fitsio.read(specfilename,"R_WAVELENGTH")
    zwave = fitsio.read(specfilename,"Z_WAVELENGTH")
    wave = np.hstack([bwave,rwave,zwave])
    bflux = fitsio.read(specfilename,"B_FLUX")[source_idx]
    rflux = fitsio.read(specfilename,"R_FLUX")[source_idx]
    zflux = fitsio.read(specfilename,"Z_FLUX")[source_idx]
    flux = np.hstack([bflux,rflux,zflux])
    
    from scipy.interpolate import interp1d
    extrapolator = interp1d(wave, flux, fill_value='extrapolate')
    wavelengths  = []
    fluxvalues   = []
    if output:
        fd = open("{}_spectrum_{}.dat".format(source_type, i), "w")
        fd.write("#   WAVELENGTH        FLUX\n#------------- -----------\n")
    for i in np.arange(3600., 9000.):
        wavelength = i
        fluxvalue = extrapolator(wavelength)
        if fluxvalue < -10:
            continue
        wavelengths.append(wavelength)
        fluxvalues.append(fluxvalue)
        if output:
            fd.write("     {0:.3f}     {1:.4f}\n".format(wavelength, fluxvalue))
    if output:
        fd.close()
    return wavelengths, fluxvalues

def get_random_spectrum(source_type, output=False):
    num_objs = 0
    while num_objs<=0:
        file_idx = random.randint(0, len(specfilenames)-1)
        fm       = fitsio.read(specfilenames[file_idx],1)
        stds     = np.where(fm["DESI_TARGET"] & desi_mask[source_type])[0]
        num_objs = len(stds)
    random_obj = random.randint(0, num_objs-1)
    source_idx  = stds[random_obj]
    return get_spectrum(file_idx, source_idx, output)

if __name__=="__main__":
    print(get_random_spectrum("QSO_SOUTH", 0))
