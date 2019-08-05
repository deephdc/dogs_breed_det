#!/usr/bin/env bash
#
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
# INFO:
# Script to read host information
#
# VKozlov @18-May-2018
#

echo "[INFO]##### Hostname: $HOSTNAME #####"
echo ""

echo "[INFO]##### Linux release: #####"
cat /etc/os-release
echo ""
echo "[INFO]##### top output: #####"
top -bn3 | head -n 5
echo ""

### info on nvidia cards ###
echo "[INFO]##### NVIDIA card (if installed) #####"
#if [ -f $(command -v nvidia-smi) ]; then
#    nvidia-smi
#else

if command nvidia-smi 2>/dev/null; then
    echo "NVIDIA is present"
else
    echo "!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!"
    echo "no nvidia-smi found on this machine"
    echo "Are you sure that it has GPU(s) and CUDA?"
    echo "!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!"
fi

echo ""
### print all environment settings: ###
echo "[INFO]##### Environment settings: #####"
env
