#!/bin/bash
export PYOPENCL_CTX="port:tesla"
jsrun_cmd="jsrun -g 1 -a 1 -n 1"
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
$jsrun_cmd js_task_info
#$jsrun_cmd python -O -u -m mpi4py ./shock1d.py
$jsrun_cmd python -u -m mpi4py ./shock1d.py
