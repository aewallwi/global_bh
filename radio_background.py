import numpy as np
from settings import COSMO,C,KBOLTZMANN,KPC,LITTLEH
from settings import DH,PI,SPLINE_DICT,ERG,HPLANCK_KEV
#from settings import ERG, HPLANCK_KS
import scipy.integrate as integrate
from cosmo_utils import dict2tuple
import scipy.interpolate as interp

def background_intensity(z,zmin,zmax,freq,emissivity_function):
    '''
    Given the rest-frame emissivity function, compute the background
    at redshift z.
    Args:
        z, redshift of observer
        freq, observer frequency
        emissivity_function, the comoving emissivity function
        which takes args (redshift,frequency) and returns comoving emissivity
        at rest-frame frequency in Joules/Hz/sec/(Mpc/h)^3
    Returns:
        Specific Intensity of background radiation at frequency freq in Watts/
        Hz/m^2
    '''
    if z<=zmax:
        g = lambda x: emissivity_function(x,freq*(1+x)/(1+z))\
        /(1.+x)/COSMO.Ez(x)
        output=integrate.quad(g,max([z,zmin]),zmax)[0]
        output=output*DH/4./PI/(1e3*KPC)**2.*LITTLEH**3.*(1.+z)**3.
        return output
    else:
        return 0.

def background_intensity_Xrays(z,zmin,zmax,ex,emissivity_function,
tau_function=lambda x,y:0.,freq_unit='keV',energy_unit='erg',area_unit='sqcm',saunit='sr'):
    '''
    Given the rest-frame emissivity function, compute the background of X-rays
    observed at redshift z.
    Args:
        z, redshift of observer
        ex, x-ray energy (KeV)
        emissivity_function, the comoving emissivity function
        which takes args (redshift,frequency) and returns comoving emissivity
        at rest-frame frequency in Joules/keV/sec/(Mpc/h)^3
        cgs, if true, return cgs units (erg cm^-2) deg^-2
    Returns:
        Specific Intensity of background radiation at frequency freq in Watts/
        Hz/m^2/sr
    '''
    if z<=zmax:
        g = lambda x: emissivity_function(x,ex*(1+x)/(1+z))\
        /(1.+x)/COSMO.Ez(x)
        output=integrate.quad(g,zmin,zmax)[0]
        output=output*DH/4./PI/(1e3*KPC)**2.*LITTLEH**3.*(1.+z)**3.
        #This gives Watts/m^2/keV
        if freq_unit=='Hz':#convert from keV^-1 to Hz^-1
            output=output*HPLANCK_KEV
        if energy_unit=='erg':#convert from J to erg
            output=output/ERG
        if area_unit=='sqcm':
            output=output*1e-4 #convert form m^-2 to cm^-2
        elif area_unit=='sqMpc':
            output=output*(1e3*KPC)**2.
        if saunit=='sqdeg':
            output=output*(PI/180.)**2.
        return output
    else:
        return 0.

def background_intensity_ir(z,emissivity_function,lambda_min=3.2e-6,lambda_max=3.9e-6,**kwargs):
    '''
    calculate the integrated background intensity from lambda_min to lambda_max
    '''
    freq_min=C*1e3/lambda_max
    freq_max=C*1e3/lambda_min
    g=lambda x: background_intensity(z,zmin=kwargs['zmin'],zmax=kwargs['zmax'],
    freq=C*1e3/x,emissivity_function=emissivity_function)*C*1e3/x**2
    return integrate.quad(g,lambda_min,lambda_max)[0]

def integrated_Xrays_2p5(z,zlow,zhigh,emissivity_function,**kwargs):
    '''
    calculate the integral of \int_0.5keV^2keV J_nu
    '''
    j2=background_intensity_Xrays(z,zlow,zhigh,2.,emissivity_function,freq_unit='keV',area_unit='sqcm',saunit='sqdeg')
    coeff=j2*2.**kwargs['alphaX']
    return coeff*(2.**(1-kwargs['alphaX'])-.5**(1.-kwargs['alphaX']))/(1.-kwargs['alphaX'])


def brightness_temperature(z,zmin,zmax,freq,emissivity_function,recompute=False,**kwargs):
    '''
    Give the rest-frame brightness temperature for a radio background
    Args:
        z, redshift of observer
        freq, observer frequency
        emissivity_function, the comoving emissivity function
        which takes args (redshift,frequency,kwargs)
    Returns:
        Brightness temperature of radio-background (Kelvin)
    '''
    #splkey=('brightness_temperature',zmin,zmax,freq)+dict2tuple(kwargs)
    #if not SPLINE_DICT.has_key(splkey) or recompute:
    #    zvals=np.linspace(0,zmax,25)
    #    tvals=np.zeros_like(zvals)
    #    for znum,zval in enumerate(zvals):
    #        tvals[znum]=background_intensity(zval,zmin,zmax,freq,emissivity_function)\
    #        *(C/freq*1e3)**2./2./KBOLTZMANN
    #    SPLINE_DICT[splkey]=interp.interp1d(zvals,tvals)
    #return SPLINE_DICT[splkey](z)

    return background_intensity(z,zmin,zmax,freq,emissivity_function)\
    *(C/freq*1e3)**2./2./KBOLTZMANN
