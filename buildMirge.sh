#!/bin/bash


# default install script for mirgecom, will build with the current development head
echo "Building MIRGE-Com from the current development head"
echo "***WARNING*** may not be compatible with this driver ***WARNING"
echo "consider build scripts from <platforms> as appropriate"

if [ -z "$(ls -A emirge)" ]; then
  git clone git@github.com:illinois-ceesd/emirge.git emirge
else
  echo "emirge install already present. Remove to build anew"
fi

cd emirge

if [ -z ${CONDA_PATH+x} ]; then
  echo "CONDA_PATH unset, installing new conda with emirge"
  ./install.sh --env-name=mirgeDriver.shock1d
else
  echo "Using existing Conda installation, ${CONDA_PATH}"
  ./install.sh --conda-prefix=$CONDA_PATH --env-name=mirgeDriver.shock1d
fi

