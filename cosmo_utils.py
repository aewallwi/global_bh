#
#cosmology utility functions
#
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from settings import COSMO, MP, MSOL, LITTLEH,PI,BARN
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

def rVir(m,z):
    '''
    virial radius (coMpc/h)
    from Maller and Bullock 2004
    Args:
        m,float,mass from halo in msolar/h
        z,float,redshift
    Returns:
        virial radius of DM halo. (coMpc/h)
    '''
    return 206.*(COSMO.Om(z)*delta(z)/97.2)**(-1./3.)\
    *(m/1e12)**(1./3.)*(1.+z)**-1.*1e-3*(1.+z)


def nH(m,z):
    '''
    Number of Hydrogen atoms in a halo with mass m_h
    Args:
    m, float, mass of halo in msolar/h
    z,float,redshift
    '''
    return m*Ob(0.)/Om(0.)*(1.-YP)/MP*MSOL*LITTLEH
def nHe(m,z):
    '''
    Number of Hydrogen atoms in a halo with mass m_h
    Args:
    m, float, mass of halo in msolar/h
    z,float,redshift
    '''
    return nH(m,z)*YP/4./(1-YP)

#*******************************************
#Ionization parameters
#*******************************************
def sigma_HLike(e,z=1.):
    '''
    gives ionization cross section in cm^2 for an X-ray with energy ex (keV)
    for a Hydrogen-like atom with atomic number z
    '''
    ex=e*1e3 #input energy in keV, convert to eV
    e1=13.6*z**2.
    if ex<e1:
        return 0.
    else:
        eps=np.sqrt(ex/e1-1.)
        return 6.3e-18*(e1/ex)**4.*np.exp(4.-(4.*np.arctan(eps)/eps))/\
        (1-np.exp(-2.*PI/eps))/z/z
def sigma_HeI(e):
    '''
    gives ionization cross section in cm^2 for X-ray of energy ex (keV)
    for HeI atom.
    '''
    ex=e*1e3#input energy is in keV, convert to eV
    if ex>=2.459e1:
        e0,sigma0,ya,p,yw,y0,y1=1.361e1,9.492e2,1.469,3.188,2.039,4.434e-1,2.136
        x=ex/e0-y0
        y=np.sqrt(x**2.+y1**2.)
        fy=((x-1.)**2.+yw**2.)*y**(0.5*p-5.5)\
        *np.sqrt(1.+np.sqrt(y/ya))**(-p)
        return fy*sigma0*BARN*1e6#Last factor converts from Mbarnes to cm^-2
    else:
        return 0.


#*******************************************
#initialize interpolation x_int_tables
#*******************************************
def init_interpolation_tables():
    '''
    Initialize interpolation tables for number of ionizing electrons produced
    per ionization of H, He, HI, fraction of energy deposited in heating and
    fraction deposited in ionization.
    '''
    table_names=['x_int_tables/log_xi_-4.0.dat',
                   'x_int_tables/log_xi_-3.6.dat',
                   'x_int_tables/log_xi_-3.3.dat',
                   'x_int_tables/log_xi_-3.0.dat',
                   'x_int_tables/log_xi_-2.6.dat',
                   'x_int_tables/log_xi_-2.3.dat',
                   'x_int_tables/log_xi_-2.0.dat',
                   'x_int_tables/log_xi_-1.6.dat',
                   'x_int_tables/log_xi_-1.3.dat',
                   'x_int_tables/log_xi_-1.0.dat',
                   'x_int_tables/xi_0.500.dat',
                   'x_int_tables/xi_0.900.dat',
                   'x_int_tables/xi_0.990.dat',
                   'x_int_tables/xi_0.999.dat']

    SPLINE_DICT['xis']=np.array([1e-4,2.318e-4,4.677e-4,1.0e-3,2.318e-3,
    4.677e-3,1.0e-2,2.318e-2,4.677e-2,1.0e-1,.5,.9,.99,.999])
    for tname,xi in zip(table_names,SPLINE_DICT['xis']):
        dirname,filename=os.path.split(os.path.abspath(__file__))
        itable=np.loadtxt(dirname+'/'+tname,skiprows=3)
        SPLINE_DICT[('f_ion',xi)]=interp.interp1d(itable[:,0]/1e3,itable[:,1])
        SPLINE_DICT[('f_heat',xi)]=interp.interp1d(itable[:,0]/1e3,itable[:,2])
        SPLINE_DICT[('f_exc',xi)]=interp.interp1d(itable[:,0]/1e3,itable[:,3])
        SPLINE_DICT[('n_Lya',xi)]=interp.interp1d(itable[:,0]/1e3,itable[:,4])
        SPLINE_DICT[('n_{ion,HI}',xi)]=interp.interp1d(itable[:,0]/1e3,itable[:,5])
        SPLINE_DICT[('n_{ion,HeI}',xi)]=interp.interp1d(itable[:,0]/1e3,itable[:,6])
        SPLINE_DICT[('n_{ion,HeII}',xi)]=interp.interp1d(itable[:,0]/1e3,itable[:,7])
        SPLINE_DICT[('shull_heating',xi)]=interp.interp1d(itable[:,0]/1e3,itable[:,8])
        SPLINE_DICT[('min_e_kev',xi)]=(itable[:,0]/1e3).min()
        SPLINE_DICT[('max_e_kev',xi)]=(itable[:,0]/1e3).max()

def interp_heat_val(e_kev,xi,mode='f_ion'):
    '''
    get the fraction of photon energy going into ionizations at energy e_kev
    and ionization fraction xi
    Args:
    ex, energy of X-ray in keV
    xi, ionized fraction of IGM
    '''
    if xi>SPLINE_DICT['xis'].min() and xi<SPLINE_DICT['xis'].max():
        ind_high=int(np.arange(SPLINE_DICT['xis'].shape[0])[SPLINE_DICT['xis']>=xi].min())
        ind_low=int(np.arange(SPLINE_DICT['xis'].shape[0])[SPLINE_DICT['xis']<xi].max())

        x_l=SPLINE_DICT['xis'][ind_low]
        x_h=SPLINE_DICT['xis'][ind_high]
        min_e=np.max([SPLINE_DICT[('min_e_kev',x_l)],SPLINE_DICT[('min_e_kev',x_h)]])
        max_e=np.min([SPLINE_DICT[('max_e_kev',x_l)],SPLINE_DICT[('max_e_kev',x_h)]])
        if e_kev<=min_e: e_kev=min_e
        if e_kev>=max_e: e_kev=max_e
        vhigh=SPLINE_DICT[(mode,x_h)](e_kev)
        vlow=SPLINE_DICT[(mode,x_l)](e_kev)
        a=(vhigh-vlow)/(x_h-x_l)
        b=vhigh-a*x_h
        return b+a*xi #linearly interpolate between xi values.
    elif xi<=SPLINE_DICT['xis'].min():
        xi=SPLINE_DICT['xis'].min()
        if e_kev<SPLINE_DICT[('min_e_kev',xi)]:
            e_kev=SPLINE_DICT[('min_e_kev',xi)]
        if e_kev>SPLINE_DICT[('max_e_kev',xi)]:
            e_kev=SPLINE_DICT[('max_e_kev',xi)]
        return SPLINE_DICT[(mode,xi)](e_kev)
    elif xi>=SPLINE_DICT['xis'].max():
        xi=SPLINE_DICT['xis'].max()
        if e_kev<SPLINE_DICT[('min_e_kev',xi)]:
            e_kev=SPLINE_DICT[('min_e_kev',xi)]
        if e_kev>SPLINE_DICT[('max_e_kev',xi)]:
            e_kev=SPLINE_DICT[('max_e_kev',xi)]
        return SPLINE_DICT[(mode,xi)](e_kev)


def ion_sum(ex,xe):
    '''
    \sum_{i,j}(hnu-E^th_j)*(fion,HI/E^th_j)f_i x_i \sigma_i
    for hun=ex
    and ionized IGM fraction xe
    '''
    e_hi=ex-E_HI_ION
    e_hei=ex-E_HEI_ION
    e_heii=ex-E_HEII_ION

    fi_hi=interp_heat_val(e_hi,xe,'n_{ion,HI}')\
    +interp_heat_val(e_hi,xe,'n_{ion,HeI}')\
    +interp_heat_val(e_hi,xe,'n_{ion,HeII}')+1.
    fi_hi=fi_hi*(F_H*(1-xe))*sigma_HLike(ex)

    fi_hei=interp_heat_val(e_hei,xe,'n_{ion,HI}')\
    +interp_heat_val(e_hei,xe,'n_{ion,HeI}')\
    +interp_heat_val(e_hei,xe,'n_{ion,HeII}')+1.
    fi_hei=fi_hei*(F_HE*(1-xe))*sigma_HeI(ex)

    fi_heii=interp_heat_val(e_heii,xe,'n_{ion,HI}')\
    +interp_heat_val(e_heii,xe,'n_{ion,HeI}')\
    +interp_heat_val(e_heii,xe,'n_{ion,HeII}')+1.
    fi_heii=fi_heii*(F_HE*xe*sigma_HLike(ex,z=2.))

    return fi_hi+fi_hei+fi_heii



def alpha_B(T4):
    '''
    Case-B recombination coefficient
    for ionized gas at T4x10^4K
    Returns cm^3, sec^-1. Applies to situation
    where photon is not re-absorbed by nearby Hydrogen
    '''
    return 2.6e-13*(T4)**-.7

def alpha_A(T4):
    '''
    Case-A recombination coefficient
    for neutral gas at T4x10^4K
    Returns cm^3, sec^-1. Applies to situation
    where photon is re-absorbed by nearby Hydrogen
    '''
    4.2e-13*(Tks[tnum-1]/1e4)**-.7

def clumping_factor(z):
    '''
    Clumping factor at redshift z
    '''
    return 2.9*((1.+z)/6.)**(-1.1)
