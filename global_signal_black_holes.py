import scipy.optimize as op
import numpy as np
import os
import scipy.integrate as integrate
from settings import COSMO,TEDDINGTON,MBH_INTERP_MAX,MBH_INTERP_MIN,SPLINE_DICT
from settings import M_INTERP_MIN,LITTLEH,PI,JY,DH,MP,MSOL,TH,KBOLTZMANN
from settings import N_INTERP_Z,N_INTERP_MBH,Z_INTERP_MAX,Z_INTERP_MIN,ERG
from settings import M_INTERP_MAX,KPC,F_HE,F_H,YP,BARN,YR,EV,ERG
from settings import N_TSTEPS,E_HI_ION,E_HEI_ION,E_HEII_ION,SIGMAT
from settings import KBOLTZMANN_KEV,NH0,NH0_CM,NHE0_CM,NHE0,C,LEDD
from cosmo_utils import *
import scipy.interpolate as interp
import copy
import matplotlib.pyplot as plt
import radio_background as RB
import recfast4py.recfast as recfast
from settings import DEBUG



def rho_bh(z,mode='both',**kwargs):
    '''
    density of black holes in msolar h^2/Mpc^3
    at redshift z given model in **kwargs
    '''
    assert mode in ['accretion','seeding','both']
    splkey=('rho','gridded',mode)+dict2tuple(kwargs)
    if not SPLINE_DICT.has_key(splkey):
        print('Growing Black Holes')
        taxis=np.linspace(.9*COSMO.age(kwargs['ZMAX']),
        1.1*COSMO.age(kwargs['ZMIN']),kwargs['NTIMES'])
        zaxis=COSMO.age(taxis,inverse=True)
        #compute density of halos in mass range
        rho_halos=np.zeros_like(zaxis)
        rho_bh=np.zeros_like(zaxis)
        for tnum in range(len(taxis)):
            g=lambda x: massfunc(10.**x,zaxis[tnum])*10.**x
            if kwargs['MASSLIMUNITS']=='KELVIN':
                limlow=np.log10(tvir2mvir(kwargs['TMIN_HALO'],
                zaxis[tnum]))
                limhigh=np.log10(tvir2mvir(kwargs['TMAX_HALO'],
                zaxis[tnum]))
            else:
                limlow=np.log10(kwargs['MMIN_HALO'])
                limhigh=np.log10(kwargs['MMAX_HALO'])
            rho_halos[tnum]=integrate.quad(g,limlow,limhigh)[0]
        rho_bh[0]=kwargs['FS']*rho_halos[0]
        dt=(taxis[tnum]-taxis[tnum-1])
        for tnum in range(1,len(taxis)):
            rho_bh[tnum]=rho_bh[tnum-1]
            if mode=='seeding' or mode=='both':
                rho_bh[tnum]=rho_bh[tnum]+kwargs['FS']*(rho_halos[tnum]-rho_halos[tnum-1])
            if mode=='accretion' or mode=='both':
                rho_bh[tnum]=rho_bh[tnum]+rho_bh[tnum-1]*dt/kwargs['TAU_GROW']
        tfunc=interp.interp1d(taxis,rho_bh)
        zv=np.linspace(zaxis.min(),zaxis.max(),N_INTERP_Z)#[1:-1]
        rhoz=tfunc(np.hstack([taxis.max(),COSMO.age(zv[1:-1]),taxis.min()]))
        SPLINE_DICT[splkey]=interp.interp1d(zv,rhoz,fill_value=0.,
        bounds_error=False)
    return SPLINE_DICT[splkey](z)



def emissivity_radio(z,freq,**kwargs):
    '''
    emissivity of radio emission from accreting black holes at redshift z
    in W/Hz*(h/Mpc)^3
    Args:
        z, redshift
        freq, co-moving frequency
        kwargs, model dictionary parameters
    '''
    if z<=kwargs['ZMAX'] and z>=kwargs['ZMIN']:
        return 1.0e22*(kwargs['FR']/250.)*(kwargs['FX']/2e-2)**0.86\
        *(rho_bh(z,**kwargs)/1e4)*(freq/1.4e9)**(-kwargs['ALPHA_R'])\
        *((2.4**(1.-kwargs['ALPHA_X'])-0.1**(1.-kwargs['ALPHA_X']))/\
        (10.**(1.-kwargs['ALPHA_X']-2.**(1.-kwargs['ALPHA_X']))))
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
    '''
    if z<=kwargs['ZMAX'] and z>=kwargs['ZMIN']:
        output=2.322e48*(kwargs['FX']/2e-2)*E_x**(-kwargs['ALPHA_X'])\
        *(rho_bh(z,**kwargs)/1e4)*(1.-kwargs['ALPHA_X'])\
        /(10.**(1.-kwargs['ALPHA_X'])-2.**(1.-kwargs['ALPHA_X']))
        if obscured:
            output=output*np.exp(-10.**kwargs['LOG10_N']*(F_H*sigma_HLike(E_x)\
            +F_HE*sigma_HeI(E_x)))
        return output
    else:
        return 0.

def emissivity_uv(z,E_uv,**kwargs):
    '''
    emissivity in UV-photons from accreting black holes at redshift z
    in (eV)/sec/eV/(h/Mpc)^3
    Args:
        z, redshift
        E_uv, energy of uv photon (eV)
        kwargs, model parameter dictionary
    '''
    power_select=np.sqrt(np.sign(E_uv-13.6),dtype=complex)
    return 3.5e3*emissivity_xrays(z,2.,obscured=False,**kwargs)*(2500./912.)**(-.61)\
    *(E_uv/13.6)**(-0.61*np.imag(power_select)-1.71*(np.real(power_select)))\
    *kwargs['F_ESC']



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
    '''
    return background_intensity(z,freq,mode='radio',**kwargs)*(C*1e3/freq)**2.\
    /2./KBOLTZMANN

def ndot_uv(z,E_low=13.6,E_high=np.infty,**kwargs):
    '''
    number of photons per second per (h/Mpc)^3 at redshift z
    emitted between E_low and E_high
    Args:
        z, redshift
        E_low, lower photon energy (eV)
        E_high, higher photon energy (eV)
    '''
    return (emissivity_uv(z,E_low,**kwargs)\
    -emissivity_uv(z,E_high,**kwargs))/(1.7)

#******************************************************************************
#Simulation functions
#******************************************************************************
def q_ionize(zlow,zhigh,ntimes=int(1e4),T4=1.,**kwargs):
    '''
    Compute the HII filling fraction over ntimes different
    redshifts between zlow and zhigh
    Args:
        zlow, minimum redshift to evolve calculatino to
        zhigh, maximum redshift to start calculation
        ntimes, number of time (redshift) steps
        T4, temperature of ionized regions
        kwargs, model parameters
    '''
    tmax=COSMO.age(zlow)
    tmin=COSMO.age(zhigh)
    taxis=np.linspace(tmin,tmax,ntimes)
    dt=(taxis[1]-taxis[0])*YR*1e9#dt in seconds (convert from Gyr)
    zaxis=COSMO.age(taxis,inverse=True)
    qvals=np.zeros_like(taxis)
    qvals_He=np.zeros_like(qvals)
    tau_vals=np.zeros_like(qvals)
    chi=YP/4./(1.-YP)
    for tnum in range(1,len(qvals)):
        dz=-zaxis[tnum]+zaxis[tnum-1]
        tval,zval=taxis[tnum-1],zaxis[tnum-1]
        trec_inv=alpha_B(T4)*NH0_CM*(1.+chi)*(1.+zval)*clumping_factor(zval)
        trec_He_inv=alpha_B(T4)*NH0_CM*(1.+2.*chi)*(1.+zval)**3.\
        *clumping_factor(zval)
        dq=-qvals[tnum-1]*trec_inv
        dq_He=-qvals_He[tnum-1]*trec_He_inv*dt
        if zval>=kwargs['ZMIN'] and zval<=kwargs['ZMAX']:
            dq=dq+ndot_uv(zval,E_low=13.6,E_high=4.*13.6,**kwargs)/NH0*dt
            dq_He=dq_He+ndot_uv(zval,E_low=13.6*4.,E_high=np.inf,**kwargs)/NHE0
        dtau=DH*1e3*KPC*1e2*NH0_CM*SIGMAT*(1.+zval)**2./COSMO.Ez(zval)*\
        (qvals[tnum-1]*(1.+chi)+qvals_He[tnum-1]*chi)*dz
        tau_vals[tnum]=tau_vals[tnum-1]+dtau
        qvals[tnum]=qvals[tnum-1]+dq
        qvals_He[tnum]=qvals_He[tnum-1]+dq_He
    return taxis,zaxis,qvals,tau_vals


def run_heating(zlow,zhigh,ntimes=int(1e2),T4=1.,**kwargs):
    '''
    Compute the kinetic temperature and electron fraction in the neutral IGM
    (xe) as a function of redshift.
    Args:
        zlow, minimum redshift to evolve calculation to
        zhigh, maximum redshift to evolve calculation to
        ntimes, number of redshift steps
        T4, temperature of HII regions
        kwargs, dictionary of model parameters.
    '''
