'''
MCMC driver for global-signal black-holes model.
To run:
mpirun -np 2 python global_signal_black_holes_mcmc.py -i <config_file>
'''
import scipy.signal as signal
import scipy.interpolate as interp
import numpy as np
import yaml, emcee, argparse, yaml
from settings import F21
import global_signal_black_holes as GSBH
import copy,sys,os


def delta_Tb_analytic(freq,**kwargs):
    '''
    Analytic function describing delta T_b
    '''

    B=4.*((freq-kwargs['NU0'])/kwargs['W'])**2.\
    *np.log(-1./kwargs['TAU']*\
    np.log((1.+np.exp(-kwargs['TAU']))/2.))
    return -kwargs['A']*(1-np.exp(-kwargs['TAU']*np.exp(B)))\
    /(1.-np.exp(-kwargs['TAU']))


def var_resid(resid_array,window_length=20):
    '''
    estimate rms noise in residuals (resid_array) by taking the running average
    of abs(resid-<resid>)**2.
    Args:
        resid_array: array of residuals (preferably mean zero)
        window_length: number of points to estimate rms from
    Returns:
        array of standard deviations at each position in array, estimated by
        taking rms of surrounding window_lenth points.
    '''
    window=np.zeros_like(resid_array)
    nd=len(resid_array)
    if np.mod(nd,2)==1:
        nd=nd-1
    iupper=int(nd/2+window_length/2)
    ilower=int(nd/2-window_length/2)
    window[ilower:iupper]=1./window_length
    return signal.fftconvolve(window,
    np.abs(resid_array-np.mean(resid_array))**2.,mode='same')

def lnlike(params,x,y,yvar,param_template,param_list,analytic):
    '''
    log-likelihood of parameters
    Args:
        params, instance of parameters defined in params_vary
        x, measured frequencies
        y, measured dTb
        yvar, measured error bars
    '''
    param_instance=copy.deepcopy(param_template)
    for param,param_key in zip(params,param_list):
        param_instance[param_key]=param
    #run heating
    if not analytic:
        signal_model=GSBH.delta_Tb(param_instance['ZLOW'],param_instance['ZHIGH'],
        param_instance['NTIMES_TB'],param_instance['T4_HII'],verbose=False,
        diagnostic=True,**param_instance)
        x_model=F21/(signal_model['Z']+1.)/1e6
        y_model=interp.interp1d(x_model,signal_model['Tb'])(x)
    else:
        y_model=delta_Tb_analytic(x,**param_instance)
        #interpolate model to measured frequencies
    return -np.sum(0.5*(y_model-y)**2./yvar)

#Construct a prior for each parameter.
#Priors can be Gaussian, Log-Normal, or Uniform
def lnprior(params,param_list,param_priors):
    '''
    Compute the lnprior distribution for params whose prior-distributions
    are specified in the paramsv_priors dictionary (read from input yaml file)
    Priors supported are Uniform, Gaussian, or Log-Normal. No prior specified
    will result in no prior distribution placed on a given parameter.
    '''
    output=0.
    for param,param_key in zip(params,param_list):
        if param_priors[param_key]['TYPE']=='UNIFORM':
            if param <= param_priors[param_key]['MIN'] or \
            param >= param_priors[param_key]['MAX']:
                 output-=np.inf
        elif param_priors[param_key]['TYPE']=='GAUSSIAN':
            var=param_priors[param_key]['VAR']
            mu=param_priors[param_key]['MEAN']
            output+=-.5*((param-mu)**2./var-np.log(2.*PI*var))
        elif param_priors[param_key]['TYPE']=='LOGNORMAL':
            var=param_priors[param_key]['VAR']
            mu=param_priors[param_key]['MEAN']
            output+=-.5*((np.log(param)-mu)**2./var-np.log(2.*PI*var))\
            -np.log(param)
    return output

def lnprob(params,x,y,yvar,param_template,param_list,param_priors,analytic):
    lp=lnprior(params,param_list,param_priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp+lnlike(params,x,y,yvar,param_template,param_list,analytic)


class Sampler():
    '''
    Class for running MCMC and storing output.
    '''
    def __init__(self,config_file,verbose=False,analytic=True):
        '''
        Initialize the sampler.
        Args:
            config_file, string with name of the config file.
        '''
        self.verbose=verbose
        self.analytic=analytic
        with open(config_file, 'r') as ymlfile:
            self.config= yaml.load(ymlfile)
        ymlfile.close()
        #read in measurement file
        #Assume first column is frequency, second column is measured brightness temp
        #and third column is the residual from fitting an empirical model
        #(see Bowman 2018)
        if self.config['DATAFILE'][-3:]=='csv':
            self.data=np.loadtxt(self.config['DATAFILE'],
            skiprows=1,delimiter=',')
        elif self.config['DATAFILE'][-3:]=='npy':
            self.data=np.load(self.config['DATAFILE'])
        self.freqs,self.tb_meas,self.dtb\
        =self.data[:,0],self.data[:,1],self.data[:,2]
        self.var_tb=var_resid(self.dtb,
        window_length=self.config['NPTS_NOISE_EST'])#Calculate std of residuals
        #read list of parameters to vary from config file,
        #and set all other parameters to default starting values
        self.params_all=self.config['PARAMS']
        self.params_vary=self.config['PARAMS2VARY']
        self.params_vary_priors=self.config['PRIORS']

    def sample(self):
        '''
        Run the MCMC.
        '''
        ndim,nwalkers=len(self.params_vary),self.config['NWALKERS']

        #perturb initial conditions
        #draw from prior to seed walkers.
        p0=np.zeros((nwalkers,len(self.params_vary)))
        for pnum,pname in enumerate(self.params_vary):
            if self.params_vary_priors[pname]['TYPE']=='UNIFORM':
                p0[:,pnum]=np.random.rand(nwalkers)\
                *(self.params_vary_priors[pname]['MAX']\
                -self.params_vary_priors[pname]['MIN'])\
                +self.params_vary_priors[pname]['MIN']
            elif self.params_vary_priors[pname]['TYPE']=='GAUSSIAN':
                p0[:,pnum]=np.random.randn(nwalkers)*sefl.params_vary_priors[pname]['STD']\
                +self.params_vary_priors[pname]['MEAN']
            elif self.params_vary_priors[pname]['TYPE']=='LOGNORMAL':
                p0[:,pnum]=np.random.lognormal(mean=self.params_vary_priors[pname]['MEAN'],
                sigma=self.params_vary_priors[pname]['STD'],size=nwalkers)
            else:
                p0[:,pnum]=(np.randn(nwalkers)*self.params_vary_priors['DEFAULT_STD']\
                +1.)*params_all[pname]#if prior not listed, start walkers randomly
                #distributed.
        args=(self.freqs,self.tb_meas,self.var_tb,
        self.params_all,self.params_vary,
        self.params_vary_priors,self.analytic)
        if self.config['MPI']:
            from emcee.utils import MPIPool
            pool=MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            self.sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,
            args=args,pool=pool)
            self.sampler.run_mcmc(p0,self.config['NSTEPS'])
            pool.close()
        else:
            self.sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,
            args=args,threads=self.config['THREADS'])
            self.sampler.run_mcmc(p0,self.config['NSTEPS'])
        if not os.path.exists(self.config['PROJECT_NAME']):
            os.makedirs(self.config['PROJECT_NAME'])
        #save output and configuration
            with open(self.config['PROJECT_NAME']+'/config.yaml','w')\
             as yaml_file:
                yaml.dump(self.config,yaml_file,default_flow_style=False)
                np.save(self.config['PROJECT_NAME']+'/chain.npy',
                self.sampler.chain)


'''
Allow execution as a script.
'''
if __name__ == "__main__":

    desc=('MCMC driver for global-signal black-holes model.\n'
          'To run: mpirun -np <num_processes>'
          'python global_signal_black_holes_mcmc.py -c <config_file>')
    parser=argparse.ArgumentParser(description=desc)
    parser.add_argument('-c','--config',
    help='configuration file')
    parser.add_argument('-v','--verbose',
    help='print more output',action='store_true')
    parser.add_argument('-a','--analytic',
    help='test mode for fitting simple analytic test model',
    action='store_true')
    #parser.add_argument('-p','--progress',
    #help='show progress bar',action='store_true')
    args=parser.parse_args()
    my_sampler=Sampler(args.config,
    analytic=args.analytic)
    my_sampler.sample()
