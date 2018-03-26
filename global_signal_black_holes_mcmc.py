'''
MCMC driver for global-signal black-holes model.
To run:
mpirun -np 2 python global_signal_black_holes_mcmc.py -i <config_file>
'''

import yaml, emcee, global_signal_black_holes, argparse

def run_mcmc(**kwargs):
    '''
    '''
    return
def read_config_file(file_name):
    '''
    read yaml-config file and return dictionary of variables for running MCMC
    '''
    return
desc=('MCMC driver for global-signal black-holes model.\n'
      'To run: mpirun -np <num_processes>'
      'python global_signal_black_holes_mcmc.py -c <config_file>')
parser=argparse.ArgumentParser(description=desc)
parser.add_argument('-c','--config',
help='configuration file')
parser.add_argument('-v','--verbose',
help='print more output',action='store_true')
parser.add_argument('-p','--progress',
help='show progress bar',action='store_true')
