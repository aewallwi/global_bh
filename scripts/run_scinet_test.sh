#!/bin/bash
# MOAB/Torque submission script for SciNet GPC
#
#PBS -l nodes=2:ppn=8,walltime=1:00:00
#PBS -N test

module load intel/15.0.2 openmpi/intel/1.6.4
module load anaconda3

source activate blackhole_fitting

cd $PBS_O_WORKDIR

mpirun -np 10 python \
$PBS_O_WORKDIR/src/global_signal_black_holes_mcmc.py -c \
$PBS_O_WORDIR/config/analytic_test_mpi.yaml
