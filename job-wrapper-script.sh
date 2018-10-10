#!/bin/bash

# # Check provided parameters
# if [ ${#*} -ne 1 ]; then
#     echo "usage: `basename $0` <jobnumber>"
#     exit 1
# fi



# Store script parameter in a variable with descriptive name
#JOBNUM=$1

# Source shell profile (needed to run setupATLAS)
source /etc/profile

# Set up desired ROOT version (taken from CVMFS)
setupATLAS
lsetup root
#lsetup python
lsetup "lcgenv -p LCG_93 x86_64-slc6-gcc62-opt matplotlib"
lsetup "lcgenv -p LCG_93 x86_64-slc6-gcc62-opt root_numpy"
lsetup "lcgenv -p LCG_93 x86_64-slc6-gcc62-opt scikitlearn"
lsetup "lcgenv -p LCG_93 x86_64-slc6-gcc62-opt keras"
lsetup "lcgenv -p LCG_93 x86_64-slc6-gcc62-opt ipython"
lsetup "lcgenv -p LCG_93 x86_64-slc6-gcc62-opt cyphon"
lsetup "lcgenv -p LCG_93 x86_64-slc6-gcc62-opt h5py"
#export PATH=$PATH:~/.local/bin

#echo "Job $JOBNUM"

# Do the real thing here
#cd /cephfs/user/s6chkirf/work/
cd work/

cp /cephfs/user/s6chkirf/work/area/run/test_ANNinput.root .


python AdverNet_variable_arg.py 2j2b ${1} ${2} ${3} ${4} ${5} ${6}



