#!/bin/bash
# MOAB/Torque submission script for SciNet GPC
#
#PBS -l nodes=1:ppn=8,walltime=1:00:00
#PBS -N test
# load modules (must match modules used for compilation)
module load anaconda3
source activate blackhole_fitting
module load intel/15.0.2 openmpi/intel/1.6.4


SRCDIR=/home/b/bmg/aaronew/global_bh/

python \
$SRCDIR/src/global_signal_black_holes_mcmc.py -c \
$CONFIG
