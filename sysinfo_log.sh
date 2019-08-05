#!/usr/bin/env bash
#
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
# INFO:
# Script to run sysinfo.sh and 
# upload results on the remote share
#
# VKozlov @5-Aug-2019
#

### user defined params
REMOTE_SHARE_NAME="rshare"
REMOTE_DIR="/Datasets/"
# service to run.
# Reminder for deepaas, CPU: --listen-port=5000, GPU: --listen-port=$PORT0"
SERVICE_CMD="deepaas-run --openwhisk-detect --listen-ip=0.0.0.0 --listen-port=$PORT0"

### settings for sysinfo.sh
DATENOW=$(date +%y%m%d_%H%M%S)
SYSINFO_CMD="${PWD}/sysinfo.sh"
SYSINFO_LOG="${PWD}/${DATENOW}_${HOSTNAME}_sysinfo.log"
echo $SYSINFO_LOG

### collect sysinfo
if [ -x $SYSINFO_CMD ]; then
    echo "[INFO] Collecting system information..."
    $SYSINFO_CMD > $SYSINFO_LOG
else
   echo "$SYSINFO_CMD not found!"
fi

### copy sysinfo to the remote share
REMOTE_PATH="$REMOTE_SHARE_NAME:$REMOTE_DIR/"
if [ -x $(which rclone) ]; then
    echo "[INFO] Now upload sysinfo log file to remote share..."
    rclone copy $SYSINFO_LOG $REMOTE_PATH
else
    echo "rclone not found!"
fi

# After collecting sysinfo, start the service
echo "[INFO] Starting the service..."
$SERVICE_CMD
