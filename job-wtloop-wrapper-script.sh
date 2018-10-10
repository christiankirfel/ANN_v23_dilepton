#!/bin/bash

# Check provided parameters
if [ ${#*} -ne 1 ]; then
    echo "usage: `basename $0` <jobnumber>"
    exit 1
fi



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
cd /cephfs/user/s6chkirf/work/area/

asetup --restore
source build/$AnalysisTop_PLATFORM/setup.sh

cd run/

mc16d_indicator=".d.root"
mc16a_indicator=".a.root"
atlas_fast_indicator=".fast"
atlas_full_indicator=".full"

var_dsid=$(echo ${1} | cut -f5 -d.)

if [[ ${1} == *"r10201"* ]]; then
    if [[ ${1} == *"a875"* ]]; then
        file_name=$var_dsid$atlas_fast_indicator$mc16d_indicator
    else
        file_name=$var_dsid$atlas_full_indicator$mc16d_indicator
    fi
else
    if [[ ${1} == *"a875"* ]]; then
        file_name=$var_dsid$atlas_fast_indicator$mc16a_indicator
    else
        file_name=$var_dsid$atlas_full_indicator$mc16a_indicator
    fi
fi




#Generate an iteratoe for duplicate file names
iterator=1

while [  -f ./$file_name ]; do
    file_name=${var_dsid}.${iterator}${mc16d_indicator}.
    iterator+=1
done

echo ${1}
echo $file_name
runWtLoop default_config.yaml -d /cephfs/user/s6chkirf/work/v23_rules/${1} -t nominal -o $file_name
#Next I can add systematics to the same file
runWtLoop default_config.yaml -d /cephfs/user/s6chkirf/work/v23_rules/${1} -t  JET_CategoryReduction_JET_BJES_Response__1down  -o $file_name
runWtLoop default_config.yaml -d /cephfs/user/s6chkirf/work/v23_rules/${1} -t  JET_CategoryReduction_JET_BJES_Response__1up  -o $file_name
#runWtLoop config.yaml -d ../../v23_rules/$(1) -t nominal -o $file_name



