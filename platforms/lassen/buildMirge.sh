#!/bin/bash

# install script for mirgecom, will build with the saved lassen version
echo "Building MIRGE-Com as recorded on Lassen"
echo "Run from the top of the driver directory"
versionDir="../platforms/lassen"

if [ -z "$(ls -A emirge)" ]; then
  git clone git@github.com:illinois-ceesd/emirge.git emirge
  #echo "no emirge"
else
  echo "emirge install already present. Remove to build anew"
fi

cd emirge

if [ -z ${CONDA_PATH+x} ]; then
  echo "CONDA_PATH unset, installing new conda with emirge"
  ./install.sh --env-name=mirgeDriver.shock1d --conda-env=${versionDir}/myenv.yml --pip-pkgs=${versionDir}/myreqs.txt
else
  echo "Using existing Conda installation, ${CONDA_PATH}"
  ./install.sh --conda-prefix=$CONDA_PATH --env-name=mirgeDriver.shock1d --conda-env=${versionDir}/myenv.yml --pip-pkgs=${versionDir}/myreqs.txt
fi
