#!/bin/bash


# update version control information, dump results into current directory
echo "Updating version control information for this driver"

if [ -z "$(ls -A emirge)" ]; then
  echo "missing emirge top level directory. Build emirge before updating version control information"
  exit 1
fi

cd emirge
./version.sh --output-requirements=../myreqs.txt --output-conda-env=../myenv.yml
