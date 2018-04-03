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
with open('/home/b/bmg/aaronew/global_bh/config/injection_template_scinet.yaml',
'r') as ymlfile:
    config=yaml.load(ymlfile)
ymlfile.close()
for signal_file in signal_list[:1]:
    #print(signal_file)
    config['DATAFILE']=signal_file
    config['PROJECT_NAME']=output+signal_file.split('/')[-1][:-4]
    if not os.path.exists(config['PROJECT_NAME']):
        os.mkdir(config['PROJECT_NAME'])
    with open(config['PROJECT_NAME']+'/config.yaml','w') as ymlfile:
        yaml.dump(config,ymlfile,default_flow_style=False)
    ymlfile.close()
    cmd='qsub -v CONFIG=\'%s\' %ssignal_injection_scinet.sh'\
    %(config['PROJECT_NAME']+'/config.yaml',
    '/home/b/bmg/aaronew/global_bh/scripts/')
    print(cmd)
    os.system(cmd)
