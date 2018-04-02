#*********************************************
#hard-coded script to generate config files and submit
#batch jobs to scinet.
#*********************************************

import sys,os,glob,yaml
signal_list=\
glob.glob('/home/b/bmg/aaronew/global_bh/tables/signal_injections/*.npy')
output='/home/b/bmg/aaronew/global_bh/signal_injections/'
if not os.path.exists(output):
    os.mkdir(output)
#load up template .yaml file
config=\
yaml.load('/home/b/bmg/aaronew/global_bh/config/injection_template_scinet.yaml')
for signal_file in signal_list:
    config['DATA_FILE']=signal_file
    config['PROJECT_NAME']=output+signal_file[:-4]
    yaml.dump(config,config['PROJECT_NAME']+'/config.yaml')
    cmd='qsub -v CONFIG=\'%s\' %s\signal_injection_scinet.sh'%(config['PROJECT_NAME'],
    '/home/b/bmg/aaronew/global_bh/scripts/')
    print(cmd)

    #os.system()
