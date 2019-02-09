import scipy.optimize as op
import numpy as np
import yaml
import scipy.integrate as integrate
import h5py
from settings import COSMO,TEDDINGTON,MBH_INTERP_MAX,MBH_INTERP_MIN,SPLINE_DICT
from settings import M_INTERP_MIN,LITTLEH,PI,JY,DH,MP,MSOL,TH,KBOLTZMANN
from settings import N_INTERP_Z,N_INTERP_MBH,Z_INTERP_MAX,Z_INTERP_MIN,ERG
from settings import M_INTERP_MAX,KPC,F_HE,F_H,YP,BARN,YR,EV,ERG,F21,E2500_KEV
from settings import N_TSTEPS,E_HI_ION,E_HEI_ION,E_HEII_ION,SIGMAT,RY_KEV
from settings import KBOLTZMANN_KEV,NH0,NH0_CM,NHE0_CM,NHE0,C,LEDD,DIRNAME
from settings import N_INTERP_X,TCMB0,ARAD,ME,TCMB0,HPLANCK_EV,NB0_CM,KEV,NSPEC_MAX
from cosmo_utils import *
import scipy.interpolate as interp
import copy
import datetime
#import radio_background as RB
#mport camb
import os
from settings import DEBUG
from joblib import Parallel, delayed


'''
Aaron Ewall-Wice
root.zarantan@gmail.com
core methods for computing the global 21cm signal under the influence of radio
loud black holes
Nov-01-2018
'''

def get_m_minmax(z,mode='BH',**kwargs):
    '''
    Method to compute the minimum and maximum masses of halos hosting black hole
    seeds or star formation.
    Args:
        z, float, redshift
        mode,string (BH, POPII, or POPIII)
        **kwargs, dictionary with arguments that include
        {'MASSLIMUNITS': string set to either 'KELVIN' or 'MSOL' determining
                         whether virial temperature or mass should be used
                         for halo mass limits.
        ,'TMIN_HALO':,'TMAX_HALO':, two entries that specify minimum and max virial temperature
                                    use 'TMIN_POPII/POPIII' etc... for stellar halos.
        ,'MMIN_HALO':,'MMAX_HALO':, two entries that specify minimum/max halo masses
                                    (if MASSLIMUNITS is MSOL). use MMIN_POPII etc...
                                    for stellar halos.
        }
    Returns:
        mmin, miniumum halos mass, float (msol/h)
        mmax, maximum halo mass, float (msol/h)

    '''
    assert mode in ['BH','POPII','POPIII']
    if mode=='BH':
        pf='HALO'
    elif mode=='POPII':
        pf='POPII'
    elif mode=='POPIII':
        pf='POPIII'
    if kwargs['MASSLIMUNITS']=='KELVIN':
        mmin=tvir2mvir(kwargs['TMIN_'+pf],z)
        mmax=tvir2mvir(kwargs['TMAX_'+pf],z)
    else:
        mmin=kwargs['MMIN_'+pf]
        mmax=kwargs['MMAX_'+pf]
    return mmin,mmax


def rho_stellar(z,pop,derivative=False,fractional=False,verbose=False,dt=1e-3,**kwargs):
    '''
    matter density in halos hosting stellar populations
    Args:
        z, float, redshift
        pop, string, population label (either II or III)
        derivative, bool, if true, return time derivativey in Gyr^-1
        fractional, bool, if true, return mass density in collapsed halos hosting Stars
         as a fraction of total comoving density.
         verbose, bool, if true return detailed commentary
         dt, float, time interval too take derivative over
         kwargs, simulation parameter dictionary.
    Returns:
        float, mass density in collapsed halos hosting stars of population pop.
        if not fractional, is in units of h^2 Mpc^-3 msolar
    '''
    splkey = ('rho_stellar','analytic',pop)+dict2tuple(kwargs)
    if not splkey in SPLINE_DICT:
        taxmin=np.max([.9*COSMO.age(kwargs['ZMAX'])-kwargs['TAU_DELAY_POP'+pop],.0015])
        taxis=np.linspace(taxmin,
        1.1*COSMO.age(kwargs['ZMIN']),kwargs['NTIMES'])
        stellar_densities = np.zeros_like(taxis)
        for tnum,tval in enumerate(taxis):
            zval = COSMO.age(tval,inverse=True)
            mmin,mmax=get_m_minmax(zval,mode='POP'+pop,**kwargs)
            if verbose: print('mmin=%.2e, mmax=%.2e'%(mmin,mmax))
            stellar_densities[tnum]=rho_collapse_st(mmin,mmax,zval,verbose=verbose,fractional=True)
            if verbose: print('z=%.2f, rho_*=%.2e'%(zval,stellar_densities[tnum]))
        stellar_densities[stellar_densities<=0.]=np.exp(-90.)
        SPLINE_DICT[splkey] = interp.interp1d(taxis,np.log(stellar_densities),bounds_error=False,fill_value=-90.)
    t = COSMO.age(z)-kwargs['TAU_DELAY_POP'+pop]
    if not derivative:
        output = np.exp(SPLINE_DICT[splkey](t))
    else:
        if t+dt <=COSMO.age(kwargs['ZMIN']) and t-dt >= COSMO.age(kwargs['ZMAX']):
            output = (np.exp(SPLINE_DICT[splkey](t+dt))\
            - np.exp(SPLINE_DICT[splkey](t-dt)))\
            /(2.*dt)
        else:
            output = 0.
    if not fractional:
        output = output * COSMO.rho_m(0)*1e9
    return output

def rho_bh(z, quantity='accreting',verbose=False, dt = 1e-3, derivative = False,**kwargs):
    '''
    Black hole mass density
    Args:
        z, float, redshift
        quantity, float, 'accreting', 'quiescent', 'seed', 'seednumber' to specify
        density of accreting, quiescent, seeds, or number density of seeds respectively.
        verbose, bool, if True, give detailed narrative
        dt, float, time interval over which to approximate derivative
        derivative, if true, compute time derivative of density
    Returns: comoving black hole mass density of specified population in units of
        h^2 msol / Mpc^3
    '''
    splkey=('rho_bh','analytic')+dict2tuple(kwargs)
    if not splkey in SPLINE_DICT:
        #define integrand
        taxmin=np.max([.9*COSMO.age(kwargs['ZMAX'])-kwargs['TAU_DELAY'],.0015])
        taxis=np.linspace(taxmin,
        1.1*COSMO.age(kwargs['ZMIN']),kwargs['NTIMES'])
        t_seed_max=COSMO.age(kwargs['Z_SEED_MIN'])
        rho_bh_accreting=np.zeros_like(taxis)
        rho_bh_quiescent=np.zeros_like(taxis)
        rho_bh_seeds=np.zeros_like(taxis)
        n_halos=np.zeros_like(taxis)
        dt=taxis[1]-taxis[0]
        qlist=['accreting','quiescent']
        t0list=[taxis[0],taxis[0]+kwargs['TAU_FEEDBACK']]
        mbh_min = kwargs['MSEED']
        mbh_max = mbh_min * np.exp(kwargs['TAU_FEEDBACK']/kwargs['TAU_GROW'])
        tmin = COSMO.age(kwargs['Z_SEED_MIN'])
        for tnum,tval in enumerate(taxis):
            #compute seed density
            if tval >= 1.35e-3:
                zval = COSMO.age(tval,inverse=True)
                mmin,mmax=get_m_minmax(zval,**kwargs)
                rho_bh_seeds[tnum] = mbh_min * rho_collapse_st(mmin,mmax,
                COSMO.age(tval,inverse=True),number=True) * kwargs['FHALO']
        SPLINE_DICT[splkey] = {}
        rho_bh_seeds[rho_bh_seeds<=0.] = np.exp(-90)
        n_halos = rho_bh_seeds/mbh_min
        SPLINE_DICT[splkey]['seednumber'] = interp.interp1d(taxis,np.log(n_halos),bounds_error=False,fill_value=-90.)
        SPLINE_DICT[splkey]['seed'] = interp.interp1d(taxis,np.log(rho_bh_seeds),bounds_error=False,fill_value=-90.)
        for tnum,tval in enumerate(taxis):
            #compute bh_accreting
            def accreting_integrand(mbh):
                tv = tval - kwargs['TAU_GROW']*np.log(mbh/mbh_min)
                if tv-dt>1.35e-3 and tv+dt <= tmin:
                    return (np.exp(SPLINE_DICT[splkey]['seednumber'](tv+dt))-np.exp(SPLINE_DICT[splkey]['seednumber'](tv-dt)))/(2.*dt)
                else:
                    return 0.
            rho_bh_accreting[tnum] =  integrate.quad(accreting_integrand,mbh_min,mbh_max)[0]\
            * kwargs['TAU_GROW']
            #compute quiescent density
            zval = COSMO.age(tval,inverse=True)
            tfb = np.min([tval - kwargs['TAU_FEEDBACK'],COSMO.age(kwargs['Z_SEED_MIN'])])
            if tfb > 1.35e-3:
                zfb = np.max([COSMO.age(tfb,inverse=True),kwargs['Z_SEED_MIN']])
                mmin,mmax=get_m_minmax(zfb,**kwargs)
                #if not expnint:
                rho_bh_quiescent[tnum] = rho_collapse_st(mmin,mmax,zfb, number = True)\
                * mbh_max * kwargs['FHALO']
            else:
                rho_bh_quiescent[tnum] = 0.

        rho_bh_quiescent[rho_bh_quiescent<=0.] = np.exp(-90.)
        rho_bh_accreting[rho_bh_accreting<=0.] = np.exp(-90.)
        SPLINE_DICT[splkey]['accreting'] = interp.interp1d(taxis,np.log(rho_bh_accreting),bounds_error=False,fill_value=-90.)
        SPLINE_DICT[splkey]['quiescent'] = interp.interp1d(taxis,np.log(rho_bh_quiescent),bounds_error=False,fill_value=-90.)
        SPLINE_DICT[splkey]['total'] = interp.interp1d(taxis,np.log(rho_bh_accreting + rho_bh_quiescent),bounds_error=False,fill_value=-90.)
    t = COSMO.age(z)-kwargs['TAU_DELAY']
    if not derivative:
        return np.exp(SPLINE_DICT[splkey][quantity]\
    (t))
    else:
        if quantity == 'seednumber' and t+dt > COSMO.age(kwargs['Z_SEED_MIN']):
            return 0.
        else:
            return (np.exp(SPLINE_DICT[splkey][quantity](t+dt))\
            - np.exp(SPLINE_DICT[splkey][quantity](t-dt)))\
            /(2.*dt)


def xray_integral_norm(alpha,emin,emax):
    '''
    Function to return the 1/integral of a power law
    Args:
        alpha, float, power law index: f(E)~E^-\alpha
        emin, float, lower integration limit
        emax, float, upper integration limit
    Returns:
        [  \int f(E) dE ] ^-1
    '''
    return (1-alpha)/(emax**(1-alpha)-emin**(1-alpha))

def log_normal_moment(mu,sigma,pow,base=10.):
    '''
    moment of a log-normal distribution
    Args:
        mu, mean of log
        sigma, std of log
        pow, order of moment
        base, base of log
    Returns:
        pow moment of log-normal distribution
    '''
    powbase=pow*np.log(base)
    return np.exp(mu*powbase+.5*(powbase*sigma)**2.)

def emissivity_radio(z,freq,**kwargs):
    '''
    emissivity of radio emission from accreting black holes at redshift z
    in W/Hz*(h/Mpc)^3
    Args:
        z, redshift
        freq, co-moving frequency
        kwargs, model dictionary parameters
    Returns:
        emissivity of radio emission from accreting black holes at redshift z
        in W/Hz*(h/Mpc)^3
    '''
    if z<=kwargs['ZMAX'] and z>=kwargs['ZMIN']:
        return 3.8e23\
        *(log_normal_moment(kwargs['R_MEAN'],kwargs['R_STD'],1.)/1.6e4)\
        *(kwargs['FLOUD']/.1)\
        *(kwargs['GBOL']/.003)\
        *(.045/kwargs['TAU_GROW'])\
        *(xray_integral_norm(kwargs['ALPHA_X'],2.,10.)/.53)\
        *2.**(0.9-kwargs['ALPHA_X'])\
        *(2.48e-3)**(1.6-kwargs['ALPHA_OX'])\
        *(2.8)**(kwargs['ALPHA_R']-.6)\
        *(4.39)**(-(kwargs['ALPHA_O1']-.61))\
        *(rho_bh(z,**kwargs)/1e4)\
        *(freq/1e9)**(-kwargs['ALPHA_R'])
    else:
        return 0.

def emissivity_xrays(z,E_x,obscured=True,**kwargs):
    '''
    emissivity of X-rays from accreting black holes at redshift z
    in (keV)/sec/keV/(h/Mpc)^3
    Args:
        z, redshift
        E_x, x-ray energy (keV)
        kwargs, model parameters dictionary
    Returns:
        X-ray emissivity (keV)/sec/keV/(h/Mpc)^3
    '''
    if z<=kwargs['ZMAX'] and z>=kwargs['ZMIN']:
        output=2.0e49*((1.+kwargs['FLOUD']\
        *log_normal_moment(kwargs['R_MEAN'],kwargs['R_STD'],1./6.)/1.3))\
        *(.045/kwargs['TAU_GROW'])\
        *(kwargs['GBOL']/.003)*(xray_integral_norm(kwargs['ALPHA_X'],2.,10.)/.53)\
        *(rho_bh(z,**kwargs)/1e4)*(E_x)**(-kwargs['ALPHA_X'])\
        *np.exp(-E_x/300.)
        #output=2.322e48*(kwargs['FX']/2e-2)*E_x**(-kwargs['ALPHA_X'])\
        #*(rho_bh_runge_kutta(z,**kwargs)/1e4)*(1.-kwargs['ALPHA_X'])\
        #/(10.**(1.-kwargs['ALPHA_X'])-2.**(1.-kwargs['ALPHA_X']))\
        #*np.exp(-E_x/300.)#include 300 keV exponential cutoff typical of AGN
        if obscured:
            output=output*np.exp(-10.**kwargs['LOG10_N']*(sigma_HLike(E_x)\
            +F_HE/F_H*sigma_HeI(E_x)))
        return output
    else:
        return 0.

def q_h(z,**kwargs):
    '''
    absorbed hydrogen-ionizing photons per second per cubic megaparsec.
    (sec^-1) Yue 2013.
    Args:
        z, redshift
        kwargs, simulation parameter dictionary
    Returns:
        Absorbed ionizing photons per second
        (h/Mpc)^3 sec^-1
    '''
    splkey = ('absorbed_ionizations',1)+dict2tuple(kwargs)
    if not splkey in SPLINE_DICT:
        eabs = lambda x: np.exp(-10.**kwargs['LOG10_N']*sigma_HLike(x))
        g = lambda x: x**(-kwargs['ALPHA_X'])*np.exp(-x/300.)*(1.-eabs(x))
        h = lambda x: (x/.2)**(-kwargs['ALPHA_O2'])*(1.-eabs(x))*(.2)**(-kwargs['ALPHA_X'])
        j = lambda x: (1. + (1./3.)*(x/RY_KEV-1.))*g(x)/x
        k = lambda x: h(x)/x
        q_u = integrate.quad(k,RY_KEV,.2)[0]+integrate.quad(j,.2,np.inf)[0]
        q_norm = q_u
        SPLINE_DICT[splkey] = q_norm
    return SPLINE_DICT[splkey] * emissivity_xrays(z,1.,obscured=False,**kwargs)

def emissivity_uv_reprocessed(z,E_uv,**kwargs):
    '''
    emissivity in UV-photons from the reprocessed emission from accreting
    black holes at redshift z
    in (eV)/sec/eV (h/Mpc)^3
    Args:
        z, redshift
        E_uv, energy of uv photon (eV)
        kwargs, model parameter dictionary
    Returns:
        UV emissivity (eV)/sec/eV (h/Mpc)^3
    '''
    tg = solve_t(**kwargs)
    E_kev = E_uv / 1e3
    output = q_h(z, **kwargs) * (  4.*PI*gamma_c(tg)/tg**.5/alpha_B(tg/1e4)\
    * np.exp(-E_kev/KBOLTZMANN_KEV/tg)\
    + 2.*(E_kev/RY_KEV/.75)*p_lya(E_kev/RY_KEV/.75)\
    * (1.+(1-.5)/.5*c_coll(tg)/alpha_B(tg/1e4)))
    return output

def emissivity_uv(z,E_uv,mode='energy',obscured=True,**kwargs):
    '''
    emissivity in UV-photons from accreting black holes at redshift z
    in (eV)/sec/eV (h/Mpc)^3
    Args:
        z, redshift
        E_uv, energy of uv photon (eV)
        kwargs, model parameter dictionary
    Returns
        UV emissivity (eV)/sec/eV (h/Mpc)^3
    '''
    if z>=kwargs['ZMIN'] and z<=kwargs['ZMAX']:
        #power_select=np.sqrt(np.sign(E_uv-13.6),dtype=complex)
        #output=3.5e3*emissivity_xrays(z,2.,obscured=False,**kwargs)*(2500./912.)**(-.61)\
        #*(E_uv/13.6)**(-0.61*np.imag(power_select)-1.71*(np.real(power_select)))
        #add stellar contribution
        output=7.8e52*(kwargs['GBOL']/.003)\
        *(.045/kwargs['TAU_GROW'])\
        *(xray_integral_norm(kwargs['ALPHA_X'],2.,10.)/.53)\
        *(rho_bh(z,**kwargs)/1e4)\
        *2.**(.9-kwargs['ALPHA_X'])\
        *(2500./912.)**(0.61-kwargs['ALPHA_O1'])\
        *(2.48e-3)**(1.6-kwargs['ALPHA_OX'])
        if E_uv>=13.6:
            output=output*(E_uv/13.6)**(-kwargs['ALPHA_O2'])
        else:
            output=output*(E_uv/13.6)**(-kwargs['ALPHA_O1'])
        #*(E_uv/13.6)**(-kwargs['ALPHA_O1']*np.imag(power_select)-kwargs['ALPHA_O2']\
        #*(np.real(power_select)))
        if obscured and E_uv >= 13.6:
            esc_key = ('UV','F_ESC')+dict2tuple(kwargs)
            if not esc_key in SPLINE_DICT:
                #print(kwargs['F_ESC_FROM_LOGN'])
                if kwargs['F_ESC_FROM_LOGN']:
                    g=lambda x: emissivity_uv(z,x*1e3,mode='energy',obscured=False,**kwargs)\
                    *np.exp(-10.**kwargs['LOG10_N']\
                    *(sigma_HLike(x)+F_HE/F_H*sigma_HeI(x)))
                    h=lambda x: emissivity_uv(z,x*1e3,mode='energy',obscured=False,**kwargs)
                    SPLINE_DICT[esc_key] = integrate.quad(g,13.6e-3,24.59e-3)[0]/integrate.quad(h,13.6e-3,24.59e-3)[0]
                else:
                    SPLINE_DICT[esc_key]=kwargs['F_ESC']
            #print(SPLINE_DICT[esc_key])
            output=output*SPLINE_DICT[esc_key]
        if kwargs['INCLUDE_REPROCESSED']:
            output = output + emissivity_uv_reprocessed(z,E_uv,**kwargs)
        if mode=='number':
            output=output/E_uv
    else:
        output=0.
    return output

def emissivity_lyalpha_stars(z,E_uv,mode,pop,verbose=False,**kwargs):
    '''
    Number of photons per second per eV emitted from stars
    between 912 Angstroms and Ly-alpha transition
    Usese Barkana 2005 emissivity model with interpolation table from 21cmFAST
    Args:
        z, float, redshift
        E_uv,photon energy (eV)
        mode, determine whether to give number per second or energy per second
        pop, string 'II' or 'III'
        kwargs, parameter dictionary
    Returns:
        UV emissivity (eV)/sec/eV/(Mpc/h)^3
    '''
    output=rho_stellar(z,pop=pop,derivative=True,**kwargs)\
    *kwargs['F_STAR_POP'+pop]*COSMO.Ob(0.)/COSMO.Om(0.)*MSOL/MP\
    *(1.-.75*YP)
    if verbose: print('rho_stellar_factor=%.2e'%output)
    output=output*stellar_spectrum(E_uv,pop=pop,**kwargs)/YR/1e9\
    *kwargs['N_ION_POP'+pop]#convert from per Gyr to
    if mode=='energy':
        output=output*E_uv
    return output

def emissivity_xrays_stars(z,E_x,pop,obscured=True,**kwargs):
    '''
    X-ray emissivity for stellar mass black holes in XRB at redshift z
    in (keV)/sec/keV/(Mpc/h)^3
    use 3e39 erg/sec *f_X*(msolar/yr)^-1 X-ray emissivit (see mesinger 2011).
    Args:
        z, float, redshift
        E_x, float, x-ray energy
        pop, string, 'II' or 'III'
        kwargs, parameter dictionary
    Returns:
        X-ray emissivity (keV)/sec/keV/(Mpc/h)^3
    '''
    if z<=kwargs['ZMAX'] and z>=kwargs['ZMIN_POP'+pop]:
        output=rho_stellar(z,pop=pop,derivative=True,**kwargs)\
        *kwargs['F_STAR_POP'+pop]*COSMO.Ob(0.)/COSMO.Om(0.)\
        *kwargs['FX_POP'+pop]\
        *ERG*3e39/KEV*1e-9/LITTLEH\
        *xray_integral_norm(kwargs['ALPHA_X_POP'+pop],0.5,8.)\
        *E_x**(-kwargs['ALPHA_X_POP'+pop])
        if obscured:
            output=output*np.exp(-10.**kwargs['LOG10_N_POP'+pop]\
            *(sigma_HLike(E_x)+F_HE/F_H*sigma_HeI(E_x)))
        return output
    else:
        return 0.

def background_intensity(z,x,mode='radio',**kwargs):
    '''
    background intensity from accreting black holes in
    radio, x-rays, or uv
    Args:
        z, redshift
        x, frequency (radio), energy (eV) (uv), energy (keV) (xrays)
        mode, 'radio', 'uv', or 'xrays'
    Returns:
        radio: W/m^2/Hz/Sr
        uv: ev/sec/cm^2/ev/Sr
        xrays: kev/sec/cm^2/kev/sr
    '''
    if z<=kwargs['ZMAX']:
        if mode=='radio':
            area_factor=1.
            emissivity_function=emissivity_radio
        elif mode=='uv':
            area_factor=1e4
            emissivity_function=emissivity_uv
        elif mode=='xrays':
            area_factor=1e4
            emissivity_function=emissivity_xrays
        g=lambda zp:emissivity_function(zp,x*(1+zp)/(1.+z),**kwargs)/(1.+zp)\
        /COSMO.Ez(zp)
        return DH/4./PI/(1e3*KPC)**2.*LITTLEH**3.*(1.+z)**3./area_factor\
        *integrate.quad(g,z,kwargs['ZMAX'])[0]
    else:
        return 0.

def brightness_temperature(z,freq,**kwargs):
    '''
    background brightness temperature at frequency freq
    in radio (Kelvin)
    Args:
        z, observation redshift
        freq, radiation frequency (Hz)
    Returns:
        brightness temperature in Kelvin
    '''
    return background_intensity(z,freq,mode='radio',**kwargs)*(C*1e3/freq)**2.\
    /2./KBOLTZMANN

def ndot_uv(z,E_low=13.6,E_high=24.6,**kwargs):
    '''
    number of photons per Gyr per (h/Mpc)^3 at redshift z
    emitted between E_low and E_high from black holes
    Args:
        z, redshift
        E_low, lower photon energy (eV)
        E_high, higher photon energy (eV)
    Returns:
        float, number of phjotons per second per unit volume in Gyr^-1 (h/Mpc)^3
    '''
    return (emissivity_uv(z,E_low,**kwargs)\
    -emissivity_uv(z,E_high,**kwargs))/(kwargs['ALPHA_O2'])\
    *YR*1e9

def ndot_uv_stars(z,pop,**kwargs):
    '''
    number of ionizing photons per Gyr per (h/Mpc)^3
    emitted at redshift z from stars
    Args:
        z, redshift
        kwargs, model parameters
    Returns:
        float, number of phjotons per second per unit volume in Gyr^-1 (h/Mpc)^3
    '''
    return kwargs['N_ION_POP'+pop]*kwargs['F_STAR_POP'+pop]\
    *kwargs['F_ESC_POP'+pop]\
    *rho_stellar(z,pop=pop,derivative=True,**kwargs)\
    *MSOL/MP/LITTLEH*(1.-.75*YP)

#******************************************************************************
#Simulation functions
#******************************************************************************
def q_ionize(zlow,zhigh,ntimes=int(1e4),T4=1.,verbose=False,**kwargs):
    '''
    Compute the HII filling fraction over ntimes different
    redshifts between zlow and zhigh
    Args:
        zlow, minimum redshift to evolve calculatino to
        zhigh, maximum redshift to start calculation
        ntimes, number of time (redshift) steps
        T4, temperature of ionized regions
        kwargs, model parameters
    Returns:
        ndarray (ntimes) HII volumetric filling factor as a function of redshift
    '''
    tmax=COSMO.age(zlow)
    tmin=COSMO.age(zhigh)
    taxis=np.linspace(tmin,tmax,ntimes)
    dt=(taxis[1]-taxis[0])#dt in Gyr
    zaxis=COSMO.age(taxis,inverse=True)
    qvals=np.zeros_like(taxis)
    qvals_He=np.zeros_like(qvals)
    dtau_vals=np.zeros_like(qvals)
    chi=YP/4./(1.-YP)
    #define integrand for helium ionization
    #define integrand for hydrogen ionization
    def qdot(t,q):
        '''
        ionizations per Gyr
        '''
        zval=COSMO.age(t,inverse=True)
        zeta_popii=kwargs['F_ESC_POPII']*kwargs['F_STAR_POPII']\
        *kwargs['N_ION_POPII']\
        *4./(4.-3.*YP)
        zeta_popiii=kwargs['F_ESC_POPIII']*kwargs['F_STAR_POPIII']\
        *kwargs['N_ION_POPIII']\
        *4./(4.-3.*YP)
        #add black holes
        rec_term=q*clumping_factor(zval)*alpha_A(T4)\
        *NH0_CM*(1+chi)*(1.+zval)**3.*1e9*YR
        dq=-rec_term
        dq=dq+zeta_popii*rho_stellar(zval,pop='II',derivative=True,
        fractional=True,verbose=verbose,**kwargs)\
        +zeta_popiii*rho_stellar(zval,pop='III',derivative=True,
        fractional=True,verbose=verbose,**kwargs)
        #print('dq=%e'%dq)
        if verbose: print('ndot_uv=%e, dot{rho}_*II = %.2e, dot{rho}_*III = %.2e'\
        %(ndot_uv(zval,**kwargs)/NH0,rho_stellar(zval,pop='II',derivative=True,fractional=True,**kwargs),
        rho_stellar(zval,pop='III',derivative=True,fractional=True,**kwargs)))
        dq=dq+ndot_uv(zval,**kwargs)/NH0
        if q<=1.:
        #add black Holes
            return dq
        else:
            return 0.
    integrator=integrate.ode(qdot)
    #integrator.set_initial_value(0.,taxis[0])
    tnum=1
    qvals=np.zeros(len(taxis))
    dtau_vals=np.zeros(len(taxis)+1)#store tau increments
    #last value is integrated tau to the lowest redshift.
    #while integrator.successful and tnum<len(taxis):
    #    integrator.integrate(integrator.t+dt)
    #    qvals[tnum]=integrator.y[0]
    #    tnum+=1
    for tnum in range(1,len(taxis)):
        qvals[tnum]=qvals[tnum-1]+dt*qdot(taxis[tnum-1],qvals[tnum-1])
    #now compute tau
    qvals[qvals>1.]=1.
    qspline=interp.interp1d(np.append(taxis,2*taxis[-1]),
    np.append(qvals,1.))
    #print(qspline(taxis[-1]))
    def tau_integrand(zv):
        return DH*1e3*KPC*1e2*NH0_CM*SIGMAT*(1.+zv)**2./COSMO.Ez(zv)*\
        qspline(COSMO.age(zv))*(1.+chi)
    for tnum in range(0,len(taxis)):
        zval=COSMO.age(taxis[tnum],inverse=True)
        dz=-COSMO.Ez(zval)*(1.+zval)/TH*dt
        dtau_vals[tnum]=dz*tau_integrand(zval)
    dtau_vals[-1]=SIGMAT*NH0_CM*DH*KPC*1e3*1e2*integrate.quad(lambda x: (1.+x)**2./COSMO.Ez(x),0.,zaxis.min())[0]
    return taxis,zaxis,qvals,dtau_vals

def Jalpha_summand(n,z,mode='agn',pop=None,verbose=False,**kwargs):
    '''
    Sum of UV flux at various Ly-n transitions
    Args:
        n, integer, order of transition
        z, float, redshift
        mode, string, 'agn' or 'stars': specifies source of photons to return
        pop, if 'stars' specified, should be 'II' or 'III'
        kwargs, dictionary of simulation parameters
    '''
    if mode=='agn':
        return integrate.quad(lambda x:
        emissivity_uv(x,e_ly_n(n)*(1.+x)/(1.+z),mode='number',
        obscured=True,**kwargs)/COSMO.Ez(x),z,zmax(z,n))[0]*pn_alpha(n)
    elif mode=='stars':
        return integrate.quad(lambda x:
        emissivity_lyalpha_stars(x,e_ly_n(n)*(1.+x)/(1.+z),
        mode='number',pop=pop,verbose=verbose,**kwargs)/COSMO.Ez(x),z,zmax(z,n))[0]*pn_alpha(n)

def J_Xrays_obs(Eobs,**kwargs):
    '''
    Compute the X-ray background from black holes and stars.
    Args:
        Eobs, observed X-ray energy
    Returns:
        X-ray flux in keV/sec/keV/m^2/Sr
    '''
    if isinstance(Eobs,np.ndarray):
        splkey=('J','Xrays')+dict2tuple(kwargs)
        if not splkey in SPLINE_DICT:
            jfactor=DH/4./PI/(1e3*KPC)**2.*LITTLEH**3.
            intfactor_bh=integrate.quad(lambda x: \
            emissivity_xrays(x,(1.+x),obscured=False,**kwargs)/(1.+x)/COSMO.Ez(x),kwargs['ZLOW'],kwargs['ZHIGH'])[0]
            intfactor_popii=integrate.quad(lambda x: \
            emissivity_xrays_stars(x,(1.+x),obscured=False,pop='II',**kwargs)/(1.+x)/COSMO.Ez(x),kwargs['ZLOW'],kwargs['ZHIGH'])[0]
            intfactor_popiii=integrate.quad(lambda x: \
            emissivity_xrays_stars(x,(1.+x),obscured=False,pop='III',**kwargs)/(1.+x)/COSMO.Ez(x),kwargs['ZLOW'],kwargs['ZHIGH'])[0]
            intfactor_popii=integrate.quad(lambda x:emissivity_xrays(x,(1.+x),**kwargs),kwargs['ZLOW'],kwargs['ZHIGH'])[0]
            SPLINE_DICT[splkey+('BH','X')]=intfactor_bh*jfactor
            SPLINE_DICT[splkey+('POPII','X')]=intfactor_popii*jfactor
            SPLINE_DICT[splkey+('POPIII','X')]=intfactor_popiii*jfactor
        return SPLINE_DICT[splkey+('BH','X')]*(Eobs)**(-kwargs['ALPHA_X'])+\
               SPLINE_DICT[splkey+('POPII','X')]*(Eobs)**(-kwargs['ALPHA_X_POPII'])+\
               SPLINE_DICT[splkey+('POPIII','X')]*(Eobs)**(-kwargs['ALPHA_X_POPIII'])
    else:
        raise(ValueError('Must provide frequencies as numpy array or float.'))

#def s2m(sjy,freq_obs,fhalo,**kwargs):
#    '''
#    Convert a flux to a black-hole mass.
#    args:
#        sjy, observed flux of source in Jy
#        freq_obs, observed frequency of source
#        fhalo, fraction of dm halos that host black-hole seeds
#
#    '''

def dn_ds_domega(sjy,freq_obs,**kwargs):
    '''
    Compute differential number of sources per flux bin
    '''
    s_mks = sjy*JY #convert Jy to SI units.
    s2m = lambda s,z:  s*4.*PI*COSMO.luminosityDistance(z)**2./(1+z)\
    *(rho_bh(z,**kwargs)/emissivity_radio(z,freq_obs*(1.+z),**kwargs))\
    * (1e3*KPC/LITTLEH)**2.*kwargs['FLOUD'] #from (Mpc/h)^2 to meter^2
    dvc = lambda z: DH*(COSMO.angularDiameterDistance(z)*(1.+z))**2./ COSMO.Ez(z)*LITTLEH
    def dnds(s,z):
        mratio = s2m(s,z)/kwargs['MSEED']
        tv = COSMO.age(z)-kwargs['TAU_GROW']*np.log(mratio)
        if mratio>1. and mratio <= np.exp(kwargs['TAU_FEEDBACK']/kwargs['TAU_GROW']) and tv>1.35e-3:
            zv = COSMO.age(tv,inverse=True)
            return rho_bh(zv, quantity='seednumber',derivative=True,**kwargs)\
            *kwargs['TAU_GROW']/(s/JY)*kwargs['FLOUD']
        else:
            return 0.

    return integrate.quad(lambda x: dnds(s_mks,x)*dvc(x),kwargs['ZLOW'],kwargs['ZHIGH'])[0]


def T_radio_obs(fobs,**kwargs):
    '''
    Compute the radio background from models of black holes and stars.
    Args:
        fobs, observed radio frequency
    Returns:
        radio background in Kelvin at frequency fobs at redshift 0.
    '''
    if isinstance(fobs,np.ndarray) or isinstance(fobs,float):
        splkey=('Temp','radio')+dict2tuple(kwargs)
        if not splkey in SPLINE_DICT:
            tfactor=DH/4./PI/(1e3*KPC)**2.*LITTLEH**3.*(C*1e3/1e9)**2./2./KBOLTZMANN
            intfactor=integrate.quad(lambda x: emissivity_radio(x,1e9*(1.+x),
            **kwargs)/(1.+x)/COSMO.Ez(x),
            kwargs['ZLOW'],kwargs['ZHIGH'])[0]
            SPLINE_DICT[splkey]=tfactor*intfactor
        return SPLINE_DICT[splkey]*(fobs/1e9)**(-2.-kwargs['ALPHA_R'])
    else:
        raise(ValueError('Must provide frequencies as numpy array or float.'))


def delta_Tb(zlow,zhigh,ntimes=int(1e3),T4_HII=1.,verbose=False,diagnostic=False
,**kwargs):
    '''
    Compute the 21cm signal as as a function of redshift for a model.
    Args:
        zlow, minimum redshift to evolve calculation to
        zhigh, maximum redshift to evolve calculation to
        ntimes, number of redshift steps
        T4_HII, temperature of HII regions
        kwargs, dictionary of model parameters.
    Returns:
        dictionary of brightness tempeature results (see below).
    '''
    #begin by calculating HII history
    if verbose: print('Computing Ionization History')
    taxis,zaxis,q_ion,tau_values=q_ionize(zlow,zhigh,ntimes,T4_HII,verbose=verbose,**kwargs)
    aaxis=1./(1.+zaxis)
    xray_axis=np.logspace(-1,3,N_INTERP_X)
    radio_axis=np.logspace(6,12,N_INTERP_X)#radio frequencies
    dlogx=np.log10(xray_axis[1]/xray_axis[0])
    Tks=np.zeros(ntimes)
    xes=np.zeros(ntimes)
    xalphas=np.zeros(ntimes)
    xcolls=np.zeros(ntimes)
    jxs=np.zeros(N_INTERP_X)#X-ray flux
    Jalphas=np.zeros(ntimes)#Lyman-Alpha flux
    Jalphas_stars=np.zeros(ntimes)
    Jxrays=np.zeros(ntimes)
    Jxrays_stars=np.zeros(ntimes)
    jrad=np.zeros(N_INTERP_X)#background 21-cm brightness temperature
    Trads=np.zeros(ntimes)#21-cm background temperature
    Tspins=np.zeros(ntimes)
    Talphas=np.zeros(ntimes)
    tb=np.zeros(ntimes)
    if diagnostic:
        jx_matrix=[]
        xray_matrix=[]
        dtau_matrix=[]
    recfast=np.loadtxt(DIRNAME+'/../tables/recfast_LCDM.dat')
    if verbose: print('Initialzing xe and Tk with recfast values.')
    xes[0]=interp.interp1d(recfast[:,0],recfast[:,1])(zhigh)
    Tks[0]=interp.interp1d(recfast[:,0],recfast[:,-1])(zhigh)
    xcolls[0]=x_coll(Tks[0],xes[0],zaxis[0],Trads[0])
    Tspins[0],Talphas[0],xalphas[0]=tspin(xcolls[0],Jalphas[0]+Jalphas_stars[0],
    Tks[0],Trads[0],zaxis[0],xes[0])
    if verbose: print('Initializing Interpolation.')
    init_interpolation_tables()
    if verbose: print('Starting Evolution at z=%.2f, xe=%.2e, Tk=%.2f'\
    %(zaxis[0],xes[0],Tks[0]))
    for tnum in range(1,len(taxis)):
        zval,tval,aval=zaxis[tnum-1],taxis[tnum-1],aaxis[tnum-1]
        xe,tk,qval=xes[tnum-1],Tks[tnum-1],q_ion[tnum-1]
        tc=Talphas[tnum-1]
        xalpha=xalphas[tnum-1]
        xcoll=xcolls[tnum-1]
        ts=Tspins[tnum-1]
        tcmb=TCMB0/aval
        xrays=xray_axis*aaxis[0]/aval
        freqs=radio_axis*aaxis[0]/aval
        dz=zaxis[tnum]-zval
        if xe<1.:
            if verbose: print('Computing Ly-alpha flux')
            for n in range(2,NSPEC_MAX):
                Jalphas[tnum]=Jalphas[tnum]\
                +Jalpha_summand(n,zaxis[tnum],verbose=verbose,**kwargs)
                for pop in ['II','III']:
                    Jalphas_stars[tnum]=Jalphas_stars[tnum]\
                    +Jalpha_summand(n,zaxis[tnum],mode='stars',
                    pop=pop,vebose=verbose,**kwargs)
            Jalphas[tnum]=1.\
            *Jalphas[tnum]*DH/aval**2./4./PI\
            /(1e3*KPC*1e2)**2.*LITTLEH**3.*HPLANCK_EV
            Jalphas_stars[tnum]=Jalphas_stars[tnum]*DH/aval**2./4./PI\
            /(1e3*KPC*1e2)**2.*LITTLEH**3.*HPLANCK_EV
            if verbose: print('Jalphas=%.2e, Jalpha_*=%.2e'%(Jalphas[tnum],Jalphas_stars[tnum]))
            dtaus=-dz*(DH*1e3*KPC*1e2)/aval**2./COSMO.Ez(zval)\
            *np.max([(1.-qval),0.])\
            *((1.-xe)*NH0_CM*sigma_HLike(xrays)\
            +(1.-xe)*NHE0_CM*sigma_HeI(xrays)\
            +xe*NHE0_CM*sigma_HLike(xrays,z=2.))
            #dtaus[xrays<kwargs['EX_MIN']]=9e99
            dtaus[xrays<.0136]=9e99
            #subtract emissivity since dz is negative
            jxs=(jxs*aval**3.\
            -(emissivity_xrays(zval,xrays,**kwargs)+
              emissivity_xrays_stars(zval,xrays,pop='II',**kwargs)+\
              emissivity_xrays_stars(zval,xrays,pop='III',**kwargs))\
            *DH/4./PI*aval/COSMO.Ez(zval)\
            /(1e3*KPC*1e2)**2.*LITTLEH**3.*dz)\
            *np.exp(-dtaus)/aaxis[tnum]**3.
            if diagnostic:
                jx_matrix=jx_matrix+[jxs]
                xray_matrix=xray_matrix+[xrays]
                dtau_matrix=dtau_matrix+[dtaus]
            jrad=(jrad*aval**3.\
            -emissivity_radio(zval,freqs,**kwargs)*DH/4./PI*aval/COSMO.Ez(zval)\
            /(1e3*KPC)**2.*LITTLEH**3.*dz)/aaxis[tnum]**3.
            Trads[tnum]=interp.interp1d(freqs,jrad*(C*1e3/freqs)**2./2./KBOLTZMANN)\
            (F21)
            jxfunc=interp.interp1d(xrays,jxs,fill_value=0.,
            kind='linear',bounds_error=False)
            g=lambda x:heating_integrand(np.exp(x),xe,jxfunc)
            lxmin=np.log(xrays.min())
            lxmax=np.log(xrays.max())
            epsx=integrate.quad(g,lxmin,lxmax)[0]
            dtdz=-TH*1e9*YR/COSMO.Ez(zval)*aval
            dt=dtdz*dz
            g=lambda x:ionization_integrand(np.exp(x),xe,jxfunc)
            gamma_ion=integrate.quad(g,lxmin,lxmax)[0]
            dxe=(gamma_ion\
            -alpha_B(tk/1e4)*xe**2.*NH0_CM/aval**3.)*dt#assuming no neutral clump
            #Compute secondary Ly-alpha photons from X-rays
            g=lambda x:xray_lyalpha_integrand(np.exp(x),x,jxfunc)
            Jalphas[tnum]=Jalphas[tnum]+integrate.quad(g,lxmin,lxmax)[0]\
            *(C*1e5)/4./PI/COSMO.Ez(zaxis[tnum])*NB0_CM/aaxis[tnum]\
            /e_ly_n(2.)*HPLANCK_EV*1e9*YR*TH
            dTk_a=2.*tk*aval*dz
            dTk_x=2./3./KBOLTZMANN_KEV/(1.+xe)*epsx*dt
            dTk_i=-tk*dxe/(1.+xe)
            dTk_c=xe/(1.+xe+F_HE)*8.*SIGMAT*(tcmb-tk)/3./ME/(C*1e3)\
            *ARAD*(tcmb)**4.*dt*1e-4
            if verbose: print('dTk_a=%.2e,dTk_c=%.2e,dTk_x=%.2e,dTk_i=%.2e'\
            %(dTk_a,dTk_c,dTk_x,dTk_i))
            Tks[tnum]=tk+dTk_a+dTk_c+dTk_x+dTk_i
            xes[tnum]=xe+dxe
            #compute Tsping
            xcolls[tnum]=x_coll(Tks[tnum],xes[tnum],zaxis[tnum],Trads[tnum])
            Tspins[tnum],Talphas[tnum],xalphas[tnum]=tspin(xcolls[tnum],
            Jalphas[tnum]+Jalphas_stars[tnum],
            Tks[tnum],Trads[tnum],zaxis[tnum],xes[tnum])
            # CMB coupling to radio background
            #ts[tnum]=Tks[tnum]
            if verbose: print('z=%.2f,Tk=%.2f,xe=%.2e,QHII=%.2f'%(zaxis[tnum],
            Tks[tnum],xes[tnum],q_ion[tnum]))
    for tnum in range(ntimes):
        tb[tnum]=27.*(1.-xes[tnum])*np.max([(1.-q_ion[tnum]),0.])\
        *(1.-(TCMB0/aaxis[tnum]+Trads[tnum])/Tspins[tnum])\
        *(COSMO.Ob(0.)*LITTLEH**2./0.023)*(0.15/COSMO.Om(0.)/LITTLEH**2.)\
        *(1./10./aaxis[tnum])**.5
    #generate radio background between 10MHz and 10 GHz
    faxis_rb=np.logspace(7,10,100)
    saxis = np.logspace(-13,-1,100)
    if kwargs['COMPUTEBACKGROUNDS']:
        tradio_obs=T_radio_obs(faxis_rb,**kwargs)
        dndsvals = np.array([dn_ds_domega(s,1.4e9,**kwargs) for s in saxis])
    else:
        tradio_obs=np.zeros_like(faxis_rb)
        dndsvals = np.zeros_like(saxis)
    #compute soft X-ray background by redshifting latest jx
    jx_obs=jxs/(1.+zaxis[tnum])**3.
    eaxis_xb=xrays/(1.+zaxis[tnum])

    output={'T':taxis,'Z':zaxis,'Tk':Tks,'Xe':xes,'Q':q_ion,'Trad':Trads,
    'Ts':Tspins,'Tb':tb,'Xalpha':xalphas,'Tc':Talphas,
    'Jalpha':Jalphas,'Xcoll':xcolls,'Jalpha*':Jalphas_stars,'Talpha':Talphas,
    'rho_bh_a':rho_bh(zaxis,quantity='accreting',**kwargs),
    'rho_bh_q':rho_bh(zaxis,quantity='quiescent',**kwargs),
    'rho_bh_s':rho_bh(zaxis,quantity='seed',**kwargs),
    'rb_obs':np.vstack([faxis_rb,tradio_obs]).T,
    'ex_obs':np.vstack([eaxis_xb,jx_obs]).T,
    'radio_counts':np.vstack([saxis,dndsvals]).T,
    'tau_ion_vals':tau_values}
    if kwargs['INCLUDE_REPROCESSED']:
        print('saving ratio.')
        ev_array=np.logspace(-2,2,1000)
        er = np.vectorize(lambda x: emissivity_uv_reprocessed(kwargs['Z_SEED_MIN'],x,**kwargs))(ev_array)
        et = np.vectorize(lambda x: emissivity_uv(kwargs['Z_SEED_MIN'],x,**kwargs))(ev_array)
        rep_ratio = er/(et-er)
        output['reprocess_ratio'] = np.array([ev_array,
                                              rep_ratio]).T
    if diagnostic:
        output['jxs']=np.array(jx_matrix)
        output['xrays']=np.array(xray_matrix)
        output['taus']=np.array(dtau_matrix)
    #save output dictionary
    return output



class GlobalSignal():
    '''
    This class stores global signal and runs simulations
    '''
    def __init__(self,config_file):
        '''
        Initialize a global signal object.
        Args:
            config_file: string pointing to .yaml file with model parameters
        Returns:
            nothing, reads in config file and initializes simulation with params
            specified in config file.
        '''
        SPLINE_DICT = {}
        self.config_file=config_file
        with open(config_file,'r') as yamlfile:
            yamldict=yaml.load(yamlfile)
            self.config=yamldict['CONFIG']
            self.params=yamldict['PARAMS']
            del yamldict
            self.param_vals={}
            for key in self.params:
                if self.params[key]['TYPE']=='FLOAT':
                    self.params[key]['P0']=float(self.params[key]['P0'])
                self.param_vals[key]=self.params[key]['P0']
        self.param_vals['MASSLIMUNITS']=self.config['MASSLIMUNITS']
        self.param_vals['FEEDBACK']=self.config['FEEDBACK']
        self.param_vals['NTIMESGLOBAL']=self.config['NTIMESGLOBAL']
        self.param_vals['ZLOW']=self.config['ZLOW']
        self.param_vals['ZHIGH']=self.config['ZHIGH']
        self.param_vals['COMPUTEBACKGROUNDS']=self.config['COMPUTEBACKGROUNDS']
        if 'INCLUDE_REPROCESSED' not in self.config:
            self.config['INCLUDE_REPROCESSED']=True
        self.param_vals['INCLUDE_REPROCESSED']=self.config['INCLUDE_REPROCESSED']
        if not 'SELFCONSISTENTO2' in self.config:
            self.config['SELFCONSISTENTO2'] = True
        if self.config['SELFCONSISTENTO2']:
            self.param_vals['ALPHA_O2'] = (self.param_vals['ALPHA_X']*np.log(.2/2.)\
            +self.param_vals['ALPHA_OX']*np.log(2./E2500_KEV)\
            +self.param_vals['ALPHA_O1']*np.log(E2500_KEV/RY_KEV))/np.log(.2/E2500_KEV)
        self.param_vals['SELFCONSISTENTO2'] = self.config['SELFCONSISTENTO2']
        self.param_vals['F_ESC_FROM_LOGN']=self.config['F_ESC_FROM_LOGN']
        self.param_history={}   #dictionary of parameters for each run
        self.global_signals={}  #dictionary of global signal files for each run
        self.run_dates=[]       #list of dates of each run.

    def set(self,key,value):
        '''
        set a parameter value
        Args:
            key, name of the parameter ot set.
            value: value to set the parameter to.
        '''
        if key in self.param_vals:
            self.param_vals[key]=value
        else:
            print('Warning: Invalid Parameter Supplied')

    def increment(self,key,dvalue,log=False,base=10.):
        '''
        increment the value of a parameter
        Args:
            key, name of parameter to set
            value, value to set the parameter to.
        '''
        if not log:
            self.param_vals[key]=self.param_vals[key]+dvalue
        else:
            self.param_vals[key]=self.param_vals[key]*10.**dvalue
        self.param_history.append(copy.deepcopy(self.param_vals))

    def calculate_global(self,verbose=False):
        '''
        wrapper for computing the global 21-cm signal.
        '''
        rundate=str(datetime.datetime.now())
        self.run_dates.append(rundate)
        if self.config['LW_FEEDBACK']:
            dtbfunc=delta_Tb_feedback
        else:
            dtbfunc=delta_Tb
        self.global_signals[rundate]=dtbfunc(\
        zlow=self.config['ZLOW'],zhigh=self.config['ZHIGH'],
        ntimes=self.param_vals['NTIMESGLOBAL'],**self.param_vals,diagnostic=True,verbose=verbose)
        self.param_history[rundate]=copy.deepcopy(self.param_vals)
    def save_to_disk(self):
        hf = h5py.File(self.config['OUTPUTNAME']+'.h5', 'w')
        for rundate in self.run_dates:
            #check if group already exists if not, create a new group.
            if not rundate in hf:
                gf=hf.create_group(rundate)
                param_set=self.param_history[rundate]
                data_set=self.global_signals[rundate]
                for data_key in data_set:
                    #print(data_key)
                    gf.create_dataset(data_key,data=data_set[data_key])
                for param_key in param_set:
                    gf.attrs[param_key]=param_set[param_key]
        hf.close()
    def read_from_disk(self):
        hf = h5py.File(self.config['OUTPUTNAME']+'.h5','r')
        #prepend run dates, param history, and global signals
        newdates=[]
        for rundate in hf:
            newdates=newdates+[rundate]
            params={}
            signals={}
            for param in hf[rundate].attrs:
                params[param]=hf[rundate].attrs[param]
            for signalname in hf[rundate]:
                signals[signalname]=np.array(hf[rundate][signalname])
            self.global_signals[rundate]=signals
            self.param_history[rundate]=params
        self.run_dates=newdates+self.run_dates
        hf.close()
