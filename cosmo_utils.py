#
#cosmology utility functions
#
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from settings import COSMO, MP, MSOL, LITTLEH,PI
from colossus.lss import mass_function
from colossus.lss import bias as col_bias
from settings import SPLINE_DICT

#*********************************************************
#utility functions
#*********************************************************
def dict2tuple(dictionary):
    '''
    Convert a dictionary to a tuple of alternating key values
    and referenced values.
    Example: {'apple':0,'beta':2000} -> ('apple',0,'beta',20000)
    Args:
        dict, dictionary
    Returns:
        tuple of (key0, value0, key1, value1, key2, value2, ...)
    '''
    output=()
    for key in dictionary.keys():
        if isinstance(dictionary[key],dict):
            output=output+dict2tuple(dictionary[key])
        else:
            output=output+(key,dictionary[key])
    return output
#**********************************************************
#cosmology utility functions
#**********************************************************
def massfunc(m,z,model='tinker08',mdef='200m'):
    '''
    dark matter mass function dN/dlog_10M/dV (h^3 Mpc^-3)
    Args:
        mass, float, msolar/h
        z, float, redshift
        model, string, dark matter prescription (see colossus documentation)
        mdef, string, mass definition.
    Returns:
        dark matter mass function dN/dlog_10M/dV (h^3 Mpc^-3)
    '''
    return mass_function.massFunction(m,z,mdef=mdef,model=model,
    q_out='dndlnM')*np.log(10.)


def bias(mvir,z,mode='tinker10',mdef='200m'):
    return col_bias.haloBias(mvir, model = 'tinker10', z = z,mdef=mdef)

def delta(z):
    '''
    critical density from Bryan and Norman 1998
    Args:
         z, float, redshift
    Returns:
         collapsed virial density relative to mean density
    '''
    x=COSMO.Om(z)-1.
    return 18.*PI*PI+82.*x-39.*x*x

def tvir(mhalo,z,mu=1.22):
    '''
    virial temperature of a halo with dark matter mass mhalo
    Args:
        mhalo: (msolar/h)
        z: redshift
        mu: mean molecular weight of hydrogen, 1.22 for neutral,
        0.59 for ionized,
        0.61 for ionized gas and singly ionized helium
    Returns:
        Virial temperature of halo in Kelvin using equation 26 from Barkana+2001
    '''
    return 1.98e4*(mu/.6)*(mhalo/1e8)**(2./3.)\
    *(COSMO.Om(0.)/COSMO.Om(z)*delta(z)/18./PI/PI)**(1./3.)*((1.+z)/10.)
def tvir2mvir(t,z,mu=1.22):
    '''
    convert from virial temperature to halo mass
    Args:
        t, temperature (K)
        z, redshift
        mu, mean molecular weight of gas
    Returns:
        dark-matter halo mass corresponding to virial temperature tvir in Msol/h
    '''
    return 1e8*(t/tvir(1e8,z,mu))**(3./2.)


def vVir(m,z):
    '''
    virial velocity of halo in km/sec
    from Maller and Bullock 2004
    Args:
         m,float,mass of halo in solar masses/h
         z,float,redshift
    Returns:
        float, circular virial velocity of dark
        matter halo (km/sec)
    '''
    return 144.*(COSMO.Om(z)*delta(z)/97.2)**(1./6.)\
    *(m/1e12)**(1./3.)*(1.+z)**.5
