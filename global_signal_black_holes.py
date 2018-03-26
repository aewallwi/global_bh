import scipy.optimize as op
import numpy as np
import scipy.integrate as integrate
from settings import COSMO,TEDDINGTON,MBH_INTERP_MAX,MBH_INTERP_MIN,SPLINE_DICT
from settings import M_INTERP_MIN,LITTLEH,PI,JY,DH,MP,MSOL,TH,KBOLTZMANN
from settings import N_INTERP_Z,N_INTERP_MBH,Z_INTERP_MAX,Z_INTERP_MIN,ERG
from settings import M_INTERP_MAX,KPC,F_HE,F_H,YP,BARN,YR,EV,ERG,F21
from settings import N_TSTEPS,E_HI_ION,E_HEI_ION,E_HEII_ION,SIGMAT
from settings import KBOLTZMANN_KEV,NH0,NH0_CM,NHE0_CM,NHE0,C,LEDD
from settings import N_INTERP_X,TCMB0,ARAD,ME,TCMB0,HPLANCK_EV,NB0_CM
from cosmo_utils import *
import scipy.interpolate as interp
import copy
import matplotlib.pyplot as plt
import radio_background as RB
import camb
import os
from settings import DEBUG

def rho_stellar(z,mode='derivative',**kwargs):
    '''
    time-derivative of collapsed matter density in msolar h^2/Mpc^3
    at redshift z for **kwargs model
    '''
    splkey=('rho','coll')+dict2tuple(kwargs)
    if not SPLINE_DICT.has_key(splkey+('integral','stellar')):
        print('Calculating Collapsed Fraction')
        taxis=np.linspace(.9*COSMO.age(kwargs['ZMAX']),
        1.1*COSMO.age(kwargs['ZMIN_STARS']),kwargs['NTIMES'])
        zaxis=COSMO.age(taxis,inverse=True)
        rho_halos=np.zeros_like(zaxis)
        for tnum in range(len(taxis)):
            g=lambda x: massfunc(10.**x,zaxis[tnum])*10.**x
            if kwargs['MASSLIMUNITS']=='KELVIN':
                limlow=np.log10(tvir2mvir(kwargs['TMIN_STARS'],
                zaxis[tnum]))
            else:
                limlow=np.log10(kwargs['MMIN_STARS'])
            rho_halos[tnum]=integrate.quad(g,limlow,20.)[0]
        SPLINE_DICT[splkey+('integral','stellar')]\
        =interp.UnivariateSpline(taxis,np.log(rho_halos))
        SPLINE_DICT[splkey+('derivative','stellar')]\
        =interp.UnivariateSpline(taxis,np.log(rho_halos)).derivative()
    if mode=='integral':
        return np.exp(SPLINE_DICT[splkey+(mode,'stellar')](COSMO.age(z)))
    else:
        return SPLINE_DICT[splkey+(mode,'stellar')](COSMO.age(z))\
        *rho_stellar(z,mode='integral',**kwargs)


def rho_bh_runge_kutta(z,quantity='rho_bh_accreting',**kwargs):
    splkey=('rho_bh','rk')+dict2tuple(kwargs)
    if not SPLINE_DICT.has_key(splkey):
        print('Growing Black Holes')
        taxis=np.linspace(.9*COSMO.age(kwargs['ZMAX']),
        1.1*COSMO.age(kwargs['ZMIN']),kwargs['NTIMES'])
        taxis_seeds=np.linspace(.8*COSMO.age(kwargs['ZMAX']),
        1.2*COSMO.age(kwargs['ZMIN']),kwargs['NTIMES'])
        zaxis=COSMO.age(taxis,inverse=True)
        zaxis_seeds=COSMO.age(taxis_seeds,inverse=True)
        #compute density of halos in mass range
        rho_seeds=np.zeros_like(zaxis)
        rho_bh=np.zeros_like(zaxis)
        rho_bh_quiescent=np.zeros_like(zaxis)
        rho_bh_accreting=np.zeros_like(zaxis)
        t_seed_max=COSMO.age(kwargs['Z_SEED_MIN'])
        for tnum in range(len(taxis_seeds)):
            g=lambda x: massfunc(10.**x,zaxis[tnum])*10.**x
            if kwargs['MASSLIMUNITS']=='KELVIN':
                limlow=np.log10(tvir2mvir(kwargs['TMIN_HALO'],
                zaxis_seeds[tnum]))
                limhigh=np.log10(tvir2mvir(kwargs['TMAX_HALO'],
                zaxis_seeds[tnum]))
            else:
                limlow=np.log10(kwargs['MMIN_HALO'])
                limhigh=np.log10(kwargs['MMAX_HALO'])
            rho_seeds[tnum]=integrate.quad(g,limlow,limhigh)[0]*kwargs['FS']
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
        qlist=['rho_bh_accreting','rho_bh_quiescent']
        t0list=[taxis[0],taxis[0]+kwargs['TAU_FEEDBACK']]
        SPLINE_DICT[splkey]={}
        for integrand,y,quantity,tstart in zip(intlist,ylist,qlist,t0list):
            integrator=integrate.ode(integrand)#.set_integrator('zvode',
            #method='bdf', with_jacobian=False)
            integrator.set_initial_value(y[0],tstart)
            tnum=int((tstart-taxis[0])/dt)+1
            print tnum
            while integrator.successful and tnum<len(taxis):
                integrator.integrate(integrator.t+dt)
                y[tnum]=integrator.y[0]
                tnum+=1
            print tnum
            SPLINE_DICT[splkey][quantity]=interp.UnivariateSpline(taxis,y,ext=2)
            #print y
        SPLINE_DICT[splkey]['rho_bh_seed']\
        =interp.UnivariateSpline(taxis,rho_seeds,ext=2)
        SPLINE_DICT[splkey]['rho_bh']\
        =interp.UnivariateSpline(taxis,
        SPLINE_DICT[splkey]['rho_bh_accreting'](taxis)
        +SPLINE_DICT[splkey]['rho_bh_quiescent'](taxis),ext=2)
    return SPLINE_DICT[splkey][quantity](COSMO.age(z))








def rho_bh(z,mode='both',quantity='rho_bh_accreting',derivative=False,**kwargs):
    '''
    density of black holes in msolar h^2/Mpc^3
    at redshift z given model in **kwargs
    '''
    assert mode in ['accretion','seeding','both']
    splkey=('rho_bh',mode)+dict2tuple(kwargs)
    if not SPLINE_DICT.has_key(splkey):
        SPLINE_DICT[splkey]={}
        print('Growing Black Holes')
        taxis=np.linspace(.9*COSMO.age(kwargs['ZMAX']),
        1.1*COSMO.age(kwargs['ZMIN']),kwargs['NTIMES'])
        zaxis=COSMO.age(taxis,inverse=True)
        #compute density of halos in mass range
        rho_seeds=np.zeros_like(zaxis)
        rho_bh=np.zeros_like(zaxis)
        rho_bh_quiescent=np.zeros_like(zaxis)
        rho_bh_accreting=np.zeros_like(zaxis)
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
            rho_seeds[tnum]=integrate.quad(g,limlow,limhigh)[0]*kwargs['FS']
        rho_bh[0]=0.
        rho_bh_accreting[0]=0.
        dt=(taxis[tnum]-taxis[tnum-1])
        seed_spline=interp.UnivariateSpline(taxis,np.log(rho_seeds),ext=2)
        for tnum in range(1,len(taxis)):
            rho_bh_accreting[tnum]=rho_bh_accreting[tnum-1]
            rho_bh_quiescent[tnum]=rho_bh_quiescent[tnum-1]
            d_rho_bh_accreting=0.
            if mode=='seeding' or mode=='both':
                d_rho_bh_accreting=d_rho_bh_accreting\
                +seed_spline.derivative()(taxis[tnum-1])\
                *np.exp(seed_spline(taxis[tnum-1]))*dt
            if mode=='accretion' or mode=='both':
                d_rho_bh_accreting=d_rho_bh_accreting\
                +rho_bh_accreting[tnum-1]*dt/kwargs['TAU_GROW']
            t0=taxis[tnum]-kwargs['TAU_FEEDBACK']
            if kwargs['FEEDBACK'] and t0>=taxis[0]:
                #delay of bholes shutting off
                d_rho_bh_quiescent=np.min([seed_spline.derivative()(t0)\
                *np.exp(seed_spline(t0))\
                *np.exp(kwargs['TAU_FEEDBACK']/kwargs['TAU_GROW']),
                rho_bh_accreting[tnum-1]/dt])*dt
            else:
                d_rho_bh_quiescent=0.
            d_rho_bh_accreting=d_rho_bh_quiescent+d_rho_bh_accreting
            rho_bh_accreting[tnum]=rho_bh_accreting[tnum-1]+d_rho_bh_accreting
            rho_bh_quiescent[tnum]=rho_bh_quiescent[tnum-1]+d_rho_bh_quiescent
            rho_bh[tnum]=rho_bh_quiescent[tnum]+rho_bh_accreting[tnum]


        for rho_vec,label in \
        zip([rho_bh,rho_bh_accreting,rho_bh_quiescent,rho_seeds],
        ['rho_bh','rho_bh_accreting','rho_bh_quiescent','rho_seeds']):
            #tfunc=interp.interp1d(taxis,rho_vec)
            #zv=np.linspace(zaxis.min(),zaxis.max(),N_INTERP_Z)#[1:-1]
            #rhoz=tfunc(np.hstack([taxis.max(),COSMO.age(zv[1:-1]),taxis.min()]))
            if np.any(rho_vec<=0.):
                if np.any(rho_vec>0.):
                    rho_vec[rho_vec<=0.]=rho_vec[rho_vec>0.].min()
                else:
                    rho_vec[rho_vec<=0.]=1e-99
            SPLINE_DICT[splkey][label]=interp.UnivariateSpline(taxis,
            np.log(rho_vec),ext=2)
    if not derivative:
        return np.exp(SPLINE_DICT[splkey][quantity](COSMO.age(z)))
        #return np.exp(SPLINE_DICT[splkey][quantity](COSMO.age(z)))
    else:
        return SPLINE_DICT[splkey][quantity].derivative()(COSMO.age(z))\
        *np.exp(SPLINE_DICT[splkey][quantity](COSMO.age(z)))
        #return np.exp(SPLINE_DICT[splkey][quantity](COSMO.age(z)))\
        #*SPLINE_DICT[splkey][quantity].derivative()(COSMO.age(z))




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
        *(rho_bh_runge_kutta(z,**kwargs)/1e4)*(freq/1.4e9)**(-kwargs['ALPHA_R'])\
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
        *(rho_bh_runge_kutta(z,**kwargs)/1e4)*(1.-kwargs['ALPHA_X'])\
        /(10.**(1.-kwargs['ALPHA_X'])-2.**(1.-kwargs['ALPHA_X']))\
        *np.exp(-E_x/300.)#include 300 keV exponential cutoff typical of AGN
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
    power_select=np.sqrt(np.sign(E_uv-13.6),dtype=complex)
    output=3.5e3*emissivity_xrays(z,2.,obscured=False,**kwargs)*(2500./912.)**(-.61)\
    *(E_uv/13.6)**(-0.61*np.imag(power_select)-1.71*(np.real(power_select)))
    #add stellar contribution
    if mode=='number':
        output=output/E_uv
    if obscured:
        output=output*kwargs['F_ESC']
    return output

def emissivity_lyalpha_stars(z,E_uv,mode='number',**kwargs):
    '''
    Number of photons per second per frequency interval emitted from stars
    between 912 Angstroms and Ly-alpha transition
    Usese Barkana 2005 emissivity model with interpolation table from 21cmFAST
    Args:
        E_uv,photon energy (eV)
    '''
    output=rho_stellar(z,mode='derivative',**kwargs)\
    *kwargs['F_STAR']*COSMO.Ob(0.)/COSMO.Om(0.)*MSOL/MP\
    *(1.-.75*YP)*stellar_spectrum(E_uv,**kwargs)/YR/1e9\
    *kwargs['N_ION_STARS']#convert from per Gyr to
    #pwer second
    if mode=='energy':
        output=output*E_uv
    return output

def emissivity_xrays_stars(z,E_x,obscured=True,**kwargs):
    '''
    X-ray emissivity for stellar mass black holes in XRB at redshift z
    in (keV)/sec/keV/(h/Mpc)^3

    use 3e39 erg/sec *f_X*(msolar/yr)^-1 X-ray emissivit (see mesinger 2011).
    '''
    if z<=kwargs['ZMAX'] and z>=kwargs['ZMIN_STARS']:
        output=rho_stellar(z,mode='derivative',**kwargs)/1e9\
        *kwargs['F_STAR']*COSMO.Ob(0.)/COSMO.Om(0.)*kwargs['FX_STARS']*6.242e8*3e39/LITTLEH\
        *(1.-kwargs['ALPHA_X_STARS'])/(8.**(1.-kwargs['ALPHA_X_STARS'])\
        -0.5**(1.-kwargs['ALPHA_X_STARS']))
        if obscured:
            output=output*np.exp(-10.**kwargs['LOG10_N_STARS']*(F_H*sigma_HLike(E_x)\
            +F_HE/F_H*sigma_HeI(E_x)))
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

def ndot_uv_stars(z,**kwargs):
    '''
    number of ionizing photons per second per (h/Mpc)^3
    emitted at redshift z
    Args:
        z, redshift
        kwargs, model parameters
    '''
    return kwargs['N_ION_STARS']*kwargs['F_STAR']*kwargs['F_ESC_STARS']\
    *rho_stellar(z,mode='derivative',**kwargs)\
    *MSOL/MP/LITTLEH/1e9/YR*(1.-.75*YP)
    #return kwargs['ZETA_ION']*rho_stellar(z,mode='derivative',**kwargs)\
    #*MSOL/MP/LITTLEH/1e9/YR*(1.-.75*YP)

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
        if zval>=kwargs['ZMIN_STARS'] and zval<=kwargs['ZMAX']:
            dq=dq+ndot_uv_stars(zval,**kwargs)/NH0*dt
        dtau=DH*1e3*KPC*1e2*NH0_CM*SIGMAT*(1.+zval)**2./COSMO.Ez(zval)*\
        (qvals[tnum-1]*(1.+chi)+qvals_He[tnum-1]*chi)*dz
        tau_vals[tnum]=tau_vals[tnum-1]+dtau
        qvals[tnum]=qvals[tnum-1]+dq
        qvals_He[tnum]=qvals_He[tnum-1]+dq_He
    return taxis,zaxis,qvals,tau_vals



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
    print('Computing Ionization History')
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
    dirname,filename=os.path.split(os.path.abspath(__file__))
    recfast=np.loadtxt(dirname+'/recfast_LCDM.dat')
    if verbose: print('Initialzing xe and Tk with recfast values.')
    xes[0]=interp.interp1d(recfast[:,0],recfast[:,1])(zhigh)
    Tks[0]=interp.interp1d(recfast[:,0],recfast[:,-1])(zhigh)
    xcolls[0]=x_coll(Tks[0],xes[0],zaxis[0])
    Talphas[0]=TCMB0/aaxis[0]
    Tspins[0]=tspin(xcolls[0],xalphas[0],Tks[0],Talphas[0],TCMB0/aaxis[0])
    if verbose: print('Initializing Interpolation.')
    init_interpolation_tables()
    print('Starting Evolution at z=%.2f, xe=%.2e, Tk=%.2f'%(zaxis[0],xes[0],Tks[0]))
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
            for n in range(2,31):
                Jalphas[tnum]=Jalphas[tnum]+integrate.quad(lambda x:
                emissivity_uv(x,e_ly_n(n)*(1.+x)/(1.+zaxis[tnum]),mode='number',
                obscured=False,**kwargs)/COSMO.Ez(x),zaxis[tnum],
                zmax(zaxis[tnum],n))[0]*pn_alpha(n)
                Jalphas_stars[tnum]=Jalphas_stars[tnum]+integrate.quad(lambda x:
                emissivity_lyalpha_stars(x,e_ly_n(n)*(1.+x)/(1.+zaxis[tnum]),
                mode='number',**kwargs)/COSMO.Ez(x),zaxis[tnum],
                zmax(zaxis[tnum],n))[0]*pn_alpha(n)

            Jalphas[tnum]=kwargs['ALPHA_FACTOR']\
            *Jalphas[tnum]*DH/aval**2./4./PI\
            /(1e3*KPC*1e2)**2.*LITTLEH**3.*HPLANCK_EV
            Jalphas_stars[tnum]=Jalphas_stars[tnum]*DH/aval**2./4./PI\
            /(1e3*KPC*1e2)**2.*LITTLEH**3.*HPLANCK_EV
            dtaus=-dz*(DH*1e3*KPC*1e2)/aval**2./COSMO.Ez(zval)\
            *(1.-(qval+(1.-qval)*xe))\
            *((1.-xe)*NH0_CM*sigma_HLike(xrays)\
            +(1.-xe)*NHE0_CM*sigma_HeI(xrays)\
            +xe*NHE0_CM*sigma_HLike(xrays,z=2.))
            dtaus[xrays<kwargs['EX_MIN']]=9e99
            #subtract emissivity since dz is negative
            jxs=(jxs*aval**3.\
            -(emissivity_xrays(zval,xrays,**kwargs)+
              emissivity_xrays_stars(zval,xrays,**kwargs))\
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
            -alpha_A(tk/1e4)*xe**2.*NH0_CM/aval**3.*clumping_factor(zval))*dt
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
            xalphas[tnum]=xalpha_over_jalpha(Tks[tnum],ts,zval,xes[tnum])\
            *(Jalphas[tnum]+Jalphas_stars[tnum])
            Talphas[tnum]=tc_eff(Tks[tnum],ts)
            xcolls[tnum]=x_coll(Tks[tnum],xes[tnum],zaxis[tnum])
            Tspins[tnum]=tspin(xcolls[tnum],xalphas[tnum],
            Tks[tnum],Talphas[tnum],TCMB0/aaxis[tnum]+Trads[tnum])#include
            # CMB coupling to radio background
            #ts[tnum]=Tks[tnum]
            if verbose print('z=%.2f,Tk=%.2f,xe=%.2e,QHII=%.2f'%(zaxis[tnum],Tks[tnum],
            xes[tnum],q_ion[tnum]))
    for tnum in range(ntimes):
        tb[tnum]=27.*(1.-xes[tnum])*np.max([(1.-q_ion[tnum]),0.])\
        *(1.-(TCMB0/aaxis[tnum]+Trads[tnum])/Tspins[tnum])\
        *(COSMO.Ob(0.)*LITTLEH**2./0.023)*(0.15/COSMO.Om(0.)/LITTLEH**2.)\
        *(1./10./aaxis[tnum])**.5
    output={'T':taxis,'Z':zaxis,'Tk':Tks,'Xe':xes,'Q':q_ion,'Trad':Trads,
    'Ts':Tspins,'Tb':tb,'Xalpha':xalphas,'Tc':Talphas,
    'Jalpha':Jalphas,'Xcoll':xcolls,'Jalpha*':Jalphas_stars}
    if diagnostic:
        output['jxs']=np.array(jx_matrix)
        output['xrays']=np.array(xray_matrix)
        output['taus']=np.array(dtau_matrix)
    return output
