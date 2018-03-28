'''
MCMC driver for global-signal black-holes model.
To run:
mpirun -np 2 python global_signal_black_holes_mcmc.py -i <config_file>
'''
import numpy,scipy.signal
import scipy.interpolate as interp
import yaml, emcee, argparse
from settings import F21
import global_signal_black_holes as GSBH

def run_mcmc(**kwargs):
    '''
    '''
    return
def read_config_file(file_name):
    '''
    read yaml-config file and return dictionary of variables for running MCMC
    '''
    return

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
    window[nd/2-window_length/2:nd/2+window_length/2]=1./window_length
    return signal.fft_convolve(window,
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
        param_instance['NTIMES_TB'],param_instance['T4_HII'],verbose=parser.verbose,
        diagnostic=True,**param_instance)
        x_model=F21/(signal_model['Z']+1.)/1e6
        y_model=interp.interp1d(x_model,signal_model['Tb'])(x)
    else:
        y_model=GSBH.analytic_Tb(x,**param_instance)
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
            output+=.5*((param-mu)**2./var-np.log(2.*PI*var))
        elif param_priors[param_key]['TYPE']=='LOGNORMAL':
            var=param_priors[param_key]['VAR']
            mu=param_priors[param_key]['MEAN']
            output+=.5*((np.log(param)-mu)**2./var-np.log(2.*PI*var))\
            -np.log(param)
    return output

def lnprob(params,x,y,yerr,param_template,param_list,param_priors):
    lp=lnprior(params,param_list,param_priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp+lnlike(params,x,y,yerr,param_template,param_list)




desc=('MCMC driver for global-signal black-holes model.\n'
      'To run: mpirun -np <num_processes>'
      'python global_signal_black_holes_mcmc.py -c <config_file>')
parser=argparse.ArgumentParser(description=desc)
parser.add_argument('-c','--config',
help='configuration file')
parser.add_argument('-v','--verbose',
help='print more output',action='store_true')
parser.add_argument('-a','--analytic',
help='test mode for fitting simple analytic test model'),
action='store_true')
#parser.add_argument('-p','--progress',
#help='show progress bar',action='store_true')
parser.parse_args()
simulation_params=read_config_file(parser.config_file)
#read in measurement file
#Assume first column is frequency, second column is measured brightness temp
#and third column is the residual from fitting an empirical model
#(see Bowman 2018)
freqs,tb_meas,dtb=np.loadtxt(simulation_params['DATAFILE'])
var_tb=var_resid(dtb)#Calculate std of residuals
#read list of parameters to vary from config file, and set all other parameters
#to default starting values
params_all=simulation_params['PARAMS']
params_vary=simulation_params['PARAMS2VARY']
params_vary_priors=simulation_params['PRIORS']


ndim,nwalkers=len(params_vary),simulation_params['NWALKERS']

#perturb initial conditions
#draw from prior to seed walkers.
p0=np.zeros((nwalkers,len(params_vary))
for pnum,pname in enumerate(params_vary):
    if params_vary_priors[pname]['TYPE']=='UNIFORM':
        p0[:,pnum]=np.random.rand(nwalkers)\
        *(params_vary_priors[pname]['MAX']-params_vary_priors[pname]['MIN'])\
        +params_vary_priors[pname]['MIN']
    elif params_vary_priors[pname]['TYPE']=='GAUSSIAN':
        p0[:,pnum]=np.random.randn(nwalkers)*params_vary_priors[pname]['STD']\
        +params_vary_priors[pname]['MEAN']
    elif params_vary_priors[pname]['TYPE']=='LOGNORMAL':
        p0[:,pnum]=np.random.lognormal(mean=params_vary_priors[pname]['MEAN'],
        sigma=params_vary_priors[pname]['STD'],size=nwalkers)
    else:
        p0[:,pnum]=(np.randn(nwalkers)*params_vary_priors['DEFAULT_STD']\
        +1.)*params_all[pname]#if prior not listed, start walkers randomly
        #distributed.



if params_all['MPI']:
    pool=MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(freqs,tb_meas,
    var_tb,params_all,params_vary,params_vary_priors),pool=pool)
    sampler.run_mcmc(p0,simulation_params['NSTEPS'])
    np.savez(simulation_params['OUTPUT_NAME'],chain=sampler.chain)
    pool.close()
else:
    sampler=emcess.EnsembleSampler(nwalkers,ndim,lnprob,args=(freqs,tb_meas,
    var_tb,params_all,params_vary,params_vary_priors),
    threads=simulation_params['THREADS'])
    sampler.run_mcmc(p0,simulation_params['NSTEPS'])
    np.savez(simulation_params['OUTPUT_NAME'],chain=sampler.chain)
