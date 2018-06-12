import scipy.optimize as op
import numpy as np
import yaml
import scipy.integrate as integrate
from settings import COSMO,TEDDINGTON,MBH_INTERP_MAX,MBH_INTERP_MIN,SPLINE_DICT
from settings import M_INTERP_MIN,LITTLEH,PI,JY,DH,MP,MSOL,TH,KBOLTZMANN
from settings import N_INTERP_Z,N_INTERP_MBH,Z_INTERP_MAX,Z_INTERP_MIN,ERG
from settings import M_INTERP_MAX,KPC,F_HE,F_H,YP,BARN,YR,EV,ERG,F21
from settings import N_TSTEPS,E_HI_ION,E_HEI_ION,E_HEII_ION,SIGMAT
from settings import KBOLTZMANN_KEV,NH0,NH0_CM,NHE0_CM,NHE0,C,LEDD,DIRNAME
from settings import N_INTERP_X,TCMB0,ARAD,ME,TCMB0,HPLANCK_EV,NB0_CM,KEV,NSPEC_MAX
from cosmo_utils import *
import scipy.interpolate as interp
import copy
#import radio_background as RB
#mport camb
import os
from settings import DEBUG
from joblib import Parallel, delayed

def get_m_minmax(z,mode='BH',**kwargs):
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

def rho_bh(z,**kwargs):
    if kwargs['DENSITYMETHOD']=='RUNGEKUTTA':
        return rho_bh_runge_kutta(z,**kwargs)
    else:
        return rho_bh_analytic(z,**kwargs)

def rho_stellar_z(z,pop,**kwargs):
    g=lambda x: massfunc(10.**x,z,model=kwargs['MFMODEL'],
    mdef=kwargs['MFDEFF'])*10.**x
    if kwargs['MASSLIMUNITS']=='KELVIN':
        limlow=np.log10(tvir2mvir(kwargs['TMIN_POP'+pop],
        z))
        limhigh=np.min([20.,np.log10(tvir2mvir(kwargs['TMAX_POP'+pop],z))])
    else:
        limlow=np.log10(kwargs['MMIN_POP'+pop])
        limhigh=np.min([20.,np.log10(kwargs['MMAX_POP'+pop])])
    return integrate.quad(g,limlow,limhigh)[0]

def rho_bh_seeds_z(z,**kwargs):
    g=lambda x: massfunc(10.**x,z,model=kwargs['MFMODEL'],
    mdef=kwargs['MFDEFF'])*10.**x
    limlow,limhigh=get_m_minmax(z,**kwargs)
    limlow,limhigh=np.log10(limlow),np.log10(limhigh)
    output=integrate.quad(g,limlow,limhigh)[0]*kwargs['FS']
    return output

def rho_stellar_analytic(z,pop,mode='derivative',fractional=False,verbose=False,**kwargs):
    '''
    time derivative of collapsed matter density in star forming halos.
    '''
    if mode=='derivative':
        d=True
    else:
        d=False
    mmin,mmax=get_m_minmax(z,mode='POP'+pop,**kwargs)
    return rho_collapse_eps(mmin,mmax,z,derivative=d,fractional=fractional)



def rho_stellar_runge_kutta(z,pop,mode='derivative',fractional=False,verbose=False,**kwargs):
    '''
    time-derivative of collapsed matter density in msolar h^2/Mpc^3
    at redshift z for **kwargs model
    '''
    splkey=('rho','coll',pop)+dict2tuple(kwargs)
    if not splkey+('integral','stellar') in SPLINE_DICT:
        if verbose: print('Calculating Collapsed Fraction')
        taxmin=np.max([COSMO.age(kwargs['ZMAX']+.5)-kwargs['TAU_DELAY_POP'+pop],.0015])
        taxis=np.linspace(taxmin,
        COSMO.age(kwargs['ZMIN_POP'+pop]-.5),kwargs['NTIMES'])
        zaxis=COSMO.age(taxis,inverse=True)
        rho_halos=np.zeros_like(zaxis)
        if kwargs['MULTIPROCESS']:
            rho_halos=np.array(Parallel(n_jobs=kwargs['NPARALLEL'])\
            (delayed(rho_stellar_z)(zval,pop,**kwargs) for zval in zaxis))
        else:
            for znum,zval in enumerate(zaxis):
                rho_halos[znum]=rho_stellar_z(zval,pop,**kwargs)
        rho_halos[rho_halos<=1e-20]=1e-20
        SPLINE_DICT[splkey+('integral','stellar')]\
        =interp.UnivariateSpline(taxis,np.log(rho_halos),ext=3)
        SPLINE_DICT[splkey+('derivative','stellar')]\
        =interp.UnivariateSpline(taxis,np.log(rho_halos),ext=3).derivative()
    if mode=='integral':
        output=np.exp(SPLINE_DICT[splkey+(mode,'stellar')](COSMO.age(z)-kwargs['TAU_DELAY_POP'+pop]))
    else:
        output=SPLINE_DICT[splkey+(mode,'stellar')](COSMO.age(z)-kwargs['TAU_DELAY_POP'+pop])\
        *rho_stellar(z,pop=pop,mode='integral',**kwargs)
    if fractional:
        output=output/(COSMO.Om(0.)*COSMO.rho_b(0.)/COSMO.Ob(0.)*(1e3)**3.)

    return output

def rho_stellar(z,pop,mode='derivative',fractional=False,verbose=False,**kwargs):
    if kwargs['DENSITYMETHOD']=='RUNGEKUTTA':
        return rho_stellar_runge_kutta(z,pop,mode=mode,fractional=fractional,verbose=verbose,**kwargs)
    elif kwargs['DENSITYMETHOD']=='ANALYTIC':
        return rho_stellar_analytic(z,pop,mode=mode,fractional=fractional,verbose=verbose,**kwargs)


def rho_bh_analytic(z,quantity='accreting',verbose=False,**kwargs):
    '''
    Analytic alternative to rho_bh_runge_kutta which relies on interpolation
    and may introduce significant numerical errors in high-growth rate scenarios
    Args:
        z, redshift
        quantity, specify whether you want accreting or quiescent black holes
    '''
    splkey=('rho_bh','analytic')+dict2tuple(kwargs)
    if not splkey in SPLINE_DICT:
        #define integrand
        taxmin=np.max([.9*COSMO.age(kwargs['ZMAX'])-kwargs['TAU_DELAY'],.0015])
        taxis=np.linspace(taxmin,
        1.1*COSMO.age(kwargs['ZMIN']),kwargs['NTIMES'])
        fb_factor=np.exp(kwargs['TAU_FEEDBACK']/kwargs['TAU_GROW'])
        t_seed_max=COSMO.age(kwargs['Z_SEED_MIN'])

        #define integrand to compute accreting black hole density
        def bh_accreting_integrand(t,y):
            tfb=t-kwargs['TAU_FEEDBACK']
            zval=COSMO.age(t,inverse=True)
            mmin,mmax=get_m_minmax(zval,**kwargs)
            output=0.
            if t<=t_seed_max:
                output=output\
                +rho_collapse_eps(mmin,mmax,zval,derivative=True)*kwargs['FS']
            if tfb<=t_seed_max:
                output=output+y/kwargs['TAU_GROW']
                if tfb>taxis[0]:
                    zfb=COSMO.age(tfb,inverse=True)
                    mmin_fb,mmax_fb=get_m_minmax(zfb,**kwargs)
                    output=output-rho_collapse_eps(mmin_fb,mmax_fb,zfb,
                    derivative=True)*fb_factor*kwargs['FS']
            return output
        #define integrand to compute quiescent black hole density
        def bh_quiescent_integrand(t,y):
            tfb=t-kwargs['TAU_FEEDBACK']
            zfb=COSMO.age(tfb,inverse=True)
            mmin,mmax=get_m_minmax(zfb,**kwargs)
            output=0.
            if tfb>taxis[0] and tfb<=t_seed_max:
                output=output+rho_collapse_eps(mmin,mmax,zfb,
                derivative=True)*fb_factor*kwargs['FS']
            return output

        #Now compute integrals
        rho_bh_accreting=np.zeros_like(taxis)
        rho_bh_quiescent=np.zeros_like(taxis)
        dt=taxis[1]-taxis[0]
        qlist=['accreting','quiescent']
        t0list=[taxis[0],taxis[0]+kwargs['TAU_FEEDBACK']]
        intlist=[bh_accreting_integrand,bh_quiescent_integrand]
        SPLINE_DICT[splkey]={}
        ylist=[rho_bh_accreting,rho_bh_quiescent]
        for integrand,y,quant,tstart in zip(intlist,ylist,qlist,t0list):
            integrator=integrate.ode(integrand)
            integrator.set_initial_value(y[0],tstart)
            tnum=int((tstart-taxis[0])/dt)+1
            while integrator.successful and tnum<len(taxis):
                integrator.integrate(integrator.t+dt)
                y[tnum]=integrator.y[0]
                tnum+=1
            y[y<=0]=np.exp(-90.)
            if quant=='accreting':
                y[taxis>=COSMO.age(kwargs['Z_SEED_MIN'])\
                +kwargs['TAU_FEEDBACK']]=np.exp(-90.)
            SPLINE_DICT[splkey][quant]=interp.interp1d(taxis,np.log(y),
            bounds_error=False,
            fill_value=-90.)
        #generate seeds spline
        rho_bh_seeds=np.zeros_like(taxis)
        for tn,t in enumerate(taxis):
            zval=COSMO.age(t,inverse=True)
            mmin,mmax=get_m_minmax(zval,**kwargs)
            rho_bh_seeds[tn]=rho_collapse_eps(mmin,mmax,zval)\
            *kwargs['FS']
        rho_bh_seeds[rho_bh_seeds<=0.]=np.exp(-90.)

        SPLINE_DICT[splkey]['seed']\
        =interp.interp1d(taxis,np.log(rho_bh_seeds),bounds_error=False,
        fill_value=-90)
        SPLINE_DICT[splkey]['total']\
        =interp.interp1d(taxis,
        np.log(np.exp(SPLINE_DICT[splkey]['accreting'](taxis))\
        +np.exp(SPLINE_DICT[splkey]['quiescent'](taxis))),
        bounds_error=False,fill_value=-90)
    return np.exp(SPLINE_DICT[splkey][quantity]\
    (COSMO.age(z)-kwargs['TAU_DELAY']))












def rho_bh_runge_kutta(z,quantity='accreting',verbose=False,**kwargs):
    splkey=('rho_bh','rk')+dict2tuple(kwargs)
    if not splkey in SPLINE_DICT:
        if verbose: print('Growing Black Holes')
        taxmin=np.max([.9*COSMO.age(kwargs['ZMAX'])-kwargs['TAU_DELAY'],.0015])
        taxis=np.linspace(taxmin,
        1.1*COSMO.age(kwargs['ZMIN']),kwargs['NTIMES'])

        taxmin=np.max([.8*COSMO.age(kwargs['ZMAX'])-kwargs['TAU_DELAY'],.0015])
        taxis_seeds=np.linspace(taxmin,
        1.2*COSMO.age(kwargs['ZMIN']),kwargs['NTIMES'])
        zaxis=COSMO.age(taxis,inverse=True)
        zaxis_seeds=COSMO.age(taxis_seeds,inverse=True)
        #compute density of halos in mass range
        rho_seeds=np.zeros_like(zaxis)
        rho_bh=np.zeros_like(zaxis)
        rho_bh_quiescent=np.zeros_like(zaxis)
        rho_bh_accreting=np.zeros_like(zaxis)
        t_seed_max=COSMO.age(kwargs['Z_SEED_MIN'])
        #allow for multiprocessing in computing black hole seed density.
        if kwargs['MULTIPROCESS']:
            rho_seeds=np.array(Parallel(n_jobs=kwargs['NPARALLEL'])\
            (delayed(rho_bh_seeds_z)(zval,**kwargs) for zval in zaxis_seeds))
        else:
            for znum,zval in enumerate(zaxis_seeds):
                rho_seeds[znum]=rho_bh_seeds_z(zval,**kwargs)
        seed_spline=interp.UnivariateSpline(taxis_seeds,rho_seeds,ext=2)
        #define black hole integrand
        rho_bh_accreting[0]=0.
        rho_bh[0]=0.
        fb_factor=np.exp(kwargs['TAU_FEEDBACK']/kwargs['TAU_GROW'])
        if not kwargs['FEEDBACK']:
            def bh_accreting_integrand(t,y):
                output=y/kwargs['TAU_GROW']
                if t<=t_seed_max:
                    output=output+seed_spline.derivative()(t)
                return output
            def bh_quiescent_integrand(t,rho_bh):
                return 0.
        else:
            def bh_accreting_integrand(t,y):
                t0=t-kwargs['TAU_FEEDBACK']
                output=0.
                if t<=t_seed_max:
                    output=output+seed_spline.derivative()(t)
                if t0<=t_seed_max:
                    output=output+y/kwargs['TAU_GROW']
                    #print 'before=%.2e'%output
                    if t0>taxis[0]:
                        output=output-seed_spline.derivative()(t0)*fb_factor
                #print 'after=%.2e'%output
                return output
            def bh_quiescent_integrand(t,y):
                t0=t-kwargs['TAU_FEEDBACK']
                if t0>taxis[0] and t0<=t_seed_max:
                    #print seed_spline.derivative()(t0)*fb_factor
                    return seed_spline.derivative()(t0)*fb_factor
                else:
                    #print('not executing')
                    return 0.

        dt=taxis[1]-taxis[0]
        intlist=[bh_accreting_integrand,bh_quiescent_integrand]
        ylist=[rho_bh_accreting,rho_bh_quiescent]
        qlist=['accreting','quiescent']
        t0list=[taxis[0],taxis[0]+kwargs['TAU_FEEDBACK']]
        SPLINE_DICT[splkey]={}
        for integrand,y,quant,tstart in zip(intlist,ylist,qlist,t0list):
            integrator=integrate.ode(integrand)#.set_integrator('zvode',
            #method='bdf', with_jacobian=False)
            integrator.set_initial_value(y[0],tstart)
            tnum=int((tstart-taxis[0])/dt)+1
            #print tnum
            while integrator.successful and tnum<len(taxis):
                integrator.integrate(integrator.t+dt)
                y[tnum]=integrator.y[0]
                tnum+=1
            #print tnum
            SPLINE_DICT[splkey][quant]=interp.UnivariateSpline(taxis,y,ext=1)
            #print y
        SPLINE_DICT[splkey]['seed']\
        =interp.UnivariateSpline(taxis,rho_seeds,ext=1)
        SPLINE_DICT[splkey]['total']\
        =interp.UnivariateSpline(taxis,
        SPLINE_DICT[splkey]['accreting'](taxis)
        +SPLINE_DICT[splkey]['quiescent'](taxis),ext=1)
    return SPLINE_DICT[splkey][quantity](COSMO.age(z)-kwargs['TAU_DELAY'])



def xray_integral_norm(alpha,emin,emax):
    return (1-alpha)/(emax**(1-alpha)-emin**(1-alpha))

def log_normal_moment(mu,sigma,pow,base=10.):
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
    '''
    if z<=kwargs['ZMAX'] and z>=kwargs['ZMIN']:
        output=2.5e49*((1.+kwargs['FLOUD']\
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
            output=output*np.exp(-10.**kwargs['LOG10_N']*(F_H*sigma_HLike(E_x)\
            +F_HE/F_H*sigma_HeI(E_x)))
        return output
    else:
        return 0.

def emissivity_uv(z,E_uv,mode='energy',obscured=True,**kwargs):
    '''
    emissivity in UV-photons from accreting black holes at redshift z
    in (eV)/sec/eV/(h/Mpc)^3
    Args:
        z, redshift
        E_uv, energy of uv photon (eV)
        kwargs, model parameter dictionary
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
        if mode=='number':
            output=output/E_uv
        if obscured:
            output=output*kwargs['F_ESC']
    else:
        output=0.
    return output

def emissivity_lyalpha_stars(z,E_uv,pop,mode='number',**kwargs):
    '''
    Number of photons per second per eV emitted from stars
    between 912 Angstroms and Ly-alpha transition
    Usese Barkana 2005 emissivity model with interpolation table from 21cmFAST
    Args:
        E_uv,photon energy (eV)
    '''
    output=rho_stellar(z,pop=pop,mode='derivative',**kwargs)\
    *kwargs['F_STAR_POP'+pop]*COSMO.Ob(0.)/COSMO.Om(0.)*MSOL/MP\
    *(1.-.75*YP)*stellar_spectrum(E_uv,pop=pop,**kwargs)/YR/1e9\
    *kwargs['N_ION_POP'+pop]#convert from per Gyr to
    if mode=='energy':
        output=output*E_uv
    return output

def emissivity_xrays_stars(z,E_x,pop,obscured=True,**kwargs):
    '''
    X-ray emissivity for stellar mass black holes in XRB at redshift z
    in (keV)/sec/keV/(h/Mpc)^3

    use 3e39 erg/sec *f_X*(msolar/yr)^-1 X-ray emissivit (see mesinger 2011).
    '''
    if z<=kwargs['ZMAX'] and z>=kwargs['ZMIN_POP'+pop]:
        output=rho_stellar(z,pop=pop,mode='derivative',**kwargs)\
        *kwargs['F_STAR_POP'+pop]*COSMO.Ob(0.)/COSMO.Om(0.)\
        *kwargs['FX_POP'+pop]\
        *ERG*3e39/KEV*1e-9/LITTLEH\
        *xray_integral_norm(kwargs['ALPHA_X_POP'+pop],0.5,8)
        if obscured:
            output=output*np.exp(-10.**kwargs['LOG10_N_POP'+pop]\
            *(F_H*sigma_HLike(E_x)+F_HE/F_H*sigma_HeI(E_x)))
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
    '''
    return background_intensity(z,freq,mode='radio',**kwargs)*(C*1e3/freq)**2.\
    /2./KBOLTZMANN

def ndot_uv(z,E_low=13.6,E_high=13.6e3,**kwargs):
    '''
    number of photons per Gyr per (h/Mpc)^3 at redshift z
    emitted between E_low and E_high
    Args:
        z, redshift
        E_low, lower photon energy (eV)
        E_high, higher photon energy (eV)
    '''
    return (emissivity_uv(z,E_low,**kwargs)\
    -emissivity_uv(z,E_high,**kwargs))/(kwargs['ALPHA_O2'])\
    *YR*1e9

def ndot_uv_stars(z,pop,**kwargs):
    '''
    number of ionizing photons per Gyr per (h/Mpc)^3
    emitted at redshift z
    Args:
        z, redshift
        kwargs, model parameters
    '''
    return kwargs['N_ION_POP'+pop]*kwargs['F_STAR_POP'+pop]\
    *kwargs['F_ESC_POP'+pop]\
    *rho_stellar(z,pop=pop,mode='derivative',**kwargs)\
    *MSOL/MP/LITTLEH*(1.-.75*YP)

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
    dt=(taxis[1]-taxis[0])#dt in Gyr
    zaxis=COSMO.age(taxis,inverse=True)
    qvals=np.zeros_like(taxis)
    qvals_He=np.zeros_like(qvals)
    tau_vals=np.zeros_like(qvals)
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
        dq=dq+zeta_popii*rho_stellar(zval,pop='II',mode='derivative',
        fractional=True,**kwargs)\
        +zeta_popiii*rho_stellar(zval,pop='III',mode='derivative',
        fractional=True,**kwargs)
        #print('dq=%e'%dq)
        #print('ndot_uv=%e'%(ndot_uv(zval,**kwargs)/NH0))
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
    tau_vals=np.zeros(len(taxis))
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
    print(qspline(taxis[-1]))
    def tau_integrand(t,tau):
        zval=COSMO.age(t,inverse=True)
        return DH*1e3*KPC*1e2*NH0_CM*SIGMAT*(1.+zval)**2./COSMO.Ez(zval)*\
        qspline(t)*(1.+chi)*(-TH/COSMO.Ez(zval))**-1.
    for tnum in range(1,len(taxis)):
        tau_vals[tnum]=tau_vals[tnum-1]\
        +dt*tau_integrand(taxis[tnum-1],tau_vals[tnum-1])
    #tnum=1
    #integrator=integrate.ode(tau_integrand)
    #integrator.set_initial_value(0.,taxis[0])
    #while integrator.successful and tnum<len(taxis):
    #    integrator.integrate(integrator.t+dt)
    #    tau_vals[tnum]=integrator.y[0]
    #    tnum+=1
        '''
        dq=-qvals[tnum-1]*trec_inv*dt
        dq_He=-qvals_He[tnum-1]*trec_He_inv*dt
        if zval>=kwargs['ZMIN'] and zval<=kwargs['ZMAX']:
            dq=dq+ndot_uv(zval,E_low=13.6,E_high=4.*13.6,**kwargs)/NH0*dt
            dq_He=dq_He+ndot_uv(zval,E_low=13.6*4.,E_high=np.inf,**kwargs)/NHE0
        if zval>=kwargs['ZMIN_POPII'] and zval<=kwargs['ZMAX']:
            dq=dq+ndot_uv_stars(zval,**kwargs)/NH0*dt
        dtau=DH*1e3*KPC*1e2*NH0_CM*SIGMAT*(1.+zval)**2./COSMO.Ez(zval)*\
        (qvals[tnum-1]*(1.+chi)+qvals_He[tnum-1]*chi)*dz
        tau_vals[tnum]=tau_vals[tnum-1]+dtau
        qvals[tnum]=qvals[tnum-1]+dq
        qvals_He[tnum]=qvals_He[tnum-1]+dq_He
        '''
    return taxis,zaxis,qvals,tau_vals

def Jalpha_summand(n,z,mode='agn',pop=None,**kwargs):
    if mode=='agn':
        return integrate.quad(lambda x:
        emissivity_uv(x,e_ly_n(n)*(1.+x)/(1.+z),mode='number',
        obscured=False,**kwargs)/COSMO.Ez(x),z,zmax(z,n))[0]*pn_alpha(n)
    elif mode=='stars':
        return integrate.quad(lambda x:
        emissivity_lyalpha_stars(x,e_ly_n(n)*(1.+x)/(1.+z),
        mode='number',pop=pop,**kwargs)/COSMO.Ez(x),z,zmax(z,n))[0]*pn_alpha(n)

def J_Xrays_obs(Eobs,**kwargs):
    '''
    Compute the X-ray background from black holes and stars.
    Args:
        Eobs, observed X-ray energy
    Returns:
        X-ray flux in keV/sec/keV/m^2/Sr
    '''
    if isinstance(Eobs,np.ndarray):
        jfactor=DH/4./PI/(1e3*KPC)**2.*LITTLEH**3.
        return np.vectorize(lambda y: integrate.quad(lambda x: \
        (emissivity_xrays(x,y*(1.+x),**kwargs)\
        +emissivity_xrays_stars(x,y*(1.+x),pop='II',**kwargs)\
        +emissivity_xrays_stars(x,y*(1.+x),pop='III',**kwargs))/(1.+x)/COSMO.Ez(x),
        kwargs['ZLOW'],kwargs['ZHIGH'])[0])(Eobs)*jfactor
    else:
        raise(ValueError('Must provide frequencies as numpy array or float.'))

def T_radio_obs(fobs,**kwargs):
    '''
    Compute the radio background from models of black holes and stars.
    Args:
        fobs, observed radio frequency
    Returns:
        radio background in Kelvin at frequency fobs at redshift 0.
    '''

    if isinstance(fobs,np.ndarray) or isinstance(fobs,float):
        tfactor=DH/4./PI/(1e3*KPC)**2.*LITTLEH**3.*(C*1e3/fobs)**2./2./KBOLTZMANN
        g=lambda y: integrate.quad(lambda x: emissivity_radio(x,y*(1.+x),
        **kwargs)/(1.+x)/COSMO.Ez(x),
        kwargs['ZLOW'],kwargs['ZHIGH'])[0]
        return np.vectorize(g)(fobs)*tfactor
    else:
        raise(ValueError('Must provide frequencies as numpy array or float.'))



def delta_Tb(zlow,zhigh,ntimes=int(1e3),T4_HII=1.,verbose=False,diagnostic=False
,**kwargs):
    '''
    Compute the kinetic temperature and electron fraction in the neutral IGM
    (xe) as a function of redshift.
    Args:
        zlow, minimum redshift to evolve calculation to
        zhigh, maximum redshift to evolve calculation to
        ntimes, number of redshift steps
        T4_HII, temperature of HII regions
        kwargs, dictionary of model parameters.
    '''
    #begin by calculating HII history
    if verbose: print('Computing Ionization History')
    taxis,zaxis,q_ion,_=q_ionize(zlow,zhigh,ntimes,T4_HII,**kwargs)
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
        #Compute Ly-alpha flux
        #jalphas[tnum]=np.array([integrate.quad(lambda x:
        #emissivity_uv(x,e_ly_n*(1.+x)/(1.+zaxis[tnum]),mode='number')\
        #/COSMO.Ez(x),zval,zmax(zaxis[tnum],n))[0]*pn_alpha(n)\
        #*DH/aval**2./4./PI/(1e3*KPC*1e2)*LITTLEH**3.
            if verbose: print('Computing Ly-alpha flux')
            if kwargs['MULTIPROCESS']:
                if verbose: print('computing AGN Ly-Alpha')
                Jalphas[tnum]=np.array(Parallel(n_jobs=kwargs['NPARALLEL'])\
                (delayed(Jalpha_summand)(n,zaxis[tnum],**kwargs)\
                 for n in range(2,31))).sum()
                if verbose: print('Computing Stars Ly-Alpha')
                for pop in ['II','III']:
                    Jalphas_stars[tnum]\
                    =Jalphas_stars[tnum]+np.array(Parallel(n_jobs=kwargs['NPARALLEL'])
                    (delayed(Jalpha_summand)(n,zaxis[tnum],mode='stars',
                    pop=pop,**kwargs) for n in range(2,31))).sum()
            else:
                for n in range(2,NSPEC_MAX):
                    Jalphas[tnum]=Jalphas[tnum]\
                    +Jalpha_summand(n,zaxis[tnum],**kwargs)
                    for pop in ['II','III']:
                        Jalphas_stars[tnum]=Jalphas_stars[tnum]\
                        +Jalpha_summand(n,zaxis[tnum],mode='stars',
                        pop=pop,**kwargs)
            #kwargs['ALPHA_FACTOR']\
            Jalphas[tnum]=1.\
            *Jalphas[tnum]*DH/aval**2./4./PI\
            /(1e3*KPC*1e2)**2.*LITTLEH**3.*HPLANCK_EV
            Jalphas_stars[tnum]=Jalphas_stars[tnum]*DH/aval**2./4./PI\
            /(1e3*KPC*1e2)**2.*LITTLEH**3.*HPLANCK_EV
            dtaus=-dz*(DH*1e3*KPC*1e2)/aval**2./COSMO.Ez(zval)\
            *(1.-(qval+(1.-qval)*xe))\
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
            #subtract emissivity since dz is negative
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
    eaxis_xb=np.logspace(np.log10(2.),1.,100)
    if kwargs['COMPUTEBACKGROUNDS']:
        tradio_obs=T_radio_obs(faxis_rb,**kwargs)
        jx_obs=J_Xrays_obs(eaxis_xb,**kwargs)
    else:
        tradio_obs=np.zeros_like(faxis_rb)
        jx_obs=np.zeros_like(eaxis_xb)
    #compute soft XRB between 2-10 keV
    output={'T':taxis,'Z':zaxis,'Tk':Tks,'Xe':xes,'Q':q_ion,'Trad':Trads,
    'Ts':Tspins,'Tb':tb,'Xalpha':xalphas,'Tc':Talphas,
    'Jalpha':Jalphas,'Xcoll':xcolls,'Jalpha*':Jalphas_stars,'Talpha':Talphas,
    'rho_bh_a':rho_bh(zaxis,quantity='accreting',**kwargs),
    'rho_bh_q':rho_bh(zaxis,quantity='quiescent',**kwargs),
    'rho_bh_s':rho_bh(zaxis,quantity='seed',**kwargs),
    'rb_obs':np.vstack([faxis_rb,tradio_obs]).T,
    'ex_obs':np.vstack([eaxis_xb,jx_obs])}
    if diagnostic:
        output['jxs']=np.array(jx_matrix)
        output['xrays']=np.array(xray_matrix)
        output['taus']=np.array(dtau_matrix)
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
        '''
        self.config_file=config_file
        with open(config_file,'r') as yamlfile:
            yamldict=yaml.load(yamlfile)
            self.config=yamldict['CONFIG']
            self.params=yamldict['PARAMS']
            del yamldict
            self.param_vals={}
            for key in self.params:
                self.param_vals[key]=self.params[key]['P0']
        self.param_vals['MULTIPROCESS']=self.config['MULTIPROCESS']
        self.param_vals['NPARALLEL']=self.config['NPARALLEL']
        self.param_vals['MASSLIMUNITS']=self.config['MASSLIMUNITS']
        self.param_vals['FEEDBACK']=self.config['FEEDBACK']
        self.param_vals['MFMODEL']=self.config['MFMODEL']
        self.param_vals['MFDEFF']=self.config['MFDEFF']
        self.param_vals['NTIMESGLOBAL']=self.config['NTIMESGLOBAL']
        self.param_vals['DENSITYMETHOD']=self.config['DENSITYMETHOD']
        self.param_vals['ZLOW']=self.config['ZLOW']
        self.param_vals['ZHIGH']=self.config['ZHIGH']
        self.param_vals['COMPUTEBACKGROUNDS']=self.config['COMPUTEBACKGROUNDS']
        self.param_history=[]#list of parameters for each calculation
        self.global_signals={}#list of global signals to store
    def set(self,key,value):
        '''
        set a parameter value
        Args:
            key, name of the parameter ot set.
            value: value to set the parameter to.
        '''
        if key in self.param_vals:
            self.param_vals[key]=value
            self.param_history.append(copy.deepcopy(self.param_vals))
        else:
            print('Warngin: Invalid Parameter Supplied')
    def increment(self,key,dvalue,log=False,base=10.):
        '''
        increment the value of a parameter
        Args:
            key, name of parameter to set
            value, value to set the parameter to.
        '''
        assert key not in ['MASSLIMUNITS','NPARALLEL','MULTIPROCESS','FEEDBACK']
        if not log:
            self.param_vals[key]=self.param_vals[key]+dvalue
        else:
            self.param_vals[key]=self.param_vals[key]*10.**dvalue
        self.param_history.append(copy.deepcopy(self.param_vals))
    def calculate_global(self):
        self.global_signals[dict2tuple(self.param_vals)]=delta_Tb(\
        zlow=self.config['ZLOW'],zhigh=self.config['ZHIGH'],
        ntimes=self.param_vals['NTIMESGLOBAL'],**self.param_vals,diagnostic=True)
