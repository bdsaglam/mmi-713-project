# Setup nsys
https://stackoverflow.com/questions/76784746/how-to-use-nsys-in-google-colab

```
apt update
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2023.2.3_2023.2.3.1001-1_amd64.deb
apt install ./nsight-systems-2023.2.3_2023.2.3.1001-1_amd64.deb
apt --fix-broken install
```


# System
> g++ --version
g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0

> nvidia-smi
Sat Jun  8 10:41:07 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |


# Profiling
nvprof --print-gpu-trace ./bin/knn_cu --benchmark -numdevices=1 -i=0

ncu --print-details all ./bin/main_cu > profiling-report.txt