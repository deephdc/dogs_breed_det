# -*- coding: utf-8 -*-
"""
Info collection about the running host.
Very much based on tensorflow/models/official/utils/logs/logger.py

@author: valentin
"""

import cpuinfo    # pylint: disable=g-import-not-at-top
import psutil
import multiprocessing
import dogs_breed_det.config as cfg

from tensorflow.python.client import device_lib


def get_sys_info(sys_info):
    _collect_cpu_info(sys_info)
    _collect_memory_info(sys_info)
    _collect_gpu_info(sys_info)

# The following code is also in tensorflow/tools/test/system_info_lib
# which is not exposed for import.
def _collect_cpu_info(run_info):
    """Collect the CPU information for the local environment."""
    cpu_info = {}
    
    cpu_info["num_cores"] = multiprocessing.cpu_count()

    info = cpuinfo.get_cpu_info()
    try:
        cpu_info["cpu_info"] = info["brand"]
    except:
        # py-cpuinfo >v5.0.0
        cpu_info["cpu_info"] = info["brand_raw"]

    try:
        cpu_info["mhz_per_cpu"] = info["hz_advertised_raw"][0] / 1.0e6
    except:
        # py-cpuinfo >v5.0.0
        cpu_info["mhz_per_cpu"] = info["hz_advertised"][0] / 1.0e6

    run_info["cpu"] = cpu_info


def _collect_memory_info(run_info):
    """Collect the memory information for the local environment."""

    vmem = psutil.virtual_memory()
    run_info["memory_total"] = vmem.total
    run_info["memory_available"] = vmem.available


def _collect_gpu_info(run_info):
  """Collect local GPU information by TF device library."""
  gpu_info = {}
  local_device_protos = device_lib.list_local_devices()

  gpu_info["count"] = len([d for d in local_device_protos
                           if d.device_type == "GPU"])
  # The device description usually is a JSON string, which contains the GPU
  # model info, eg:
  # "device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0"
  for d in local_device_protos:
    if d.device_type == "GPU":
      gpu_info["model"] = _parse_gpu_model(d.physical_device_desc)
      # Assume all the GPU connected are same model
      break
  run_info["gpu"] = gpu_info


def _parse_gpu_model(physical_device_desc):
  # Assume all the GPU connected are same model
  for kv in physical_device_desc.split(","):
    k, _, v = kv.partition(":")
    if k.strip() == "name":
      return v.strip()
  return None