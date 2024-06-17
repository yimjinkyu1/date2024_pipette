# [DATE 2024] Pipette: Automatic Fine-grained Large Language Model Training Configurator for Real-World Clusters

It is common to use a cluster of GPUs with 3D parallelism, which splits a model along the data batch, pipeline stage, and intra-layer tensor dimensions. However, the use of 3D parallelism produces the additional challenge of finding the optimal number of ways on each dimension and mapping the split models onto the GPUs. We propose Pipette, which is an automatic fine-grained LLM training configurator for real-world clusters. By devising better performance models along with the memory estimator and fine-grained individual GPU assignment, Pipette achieves faster configurations that satisfy the memory constraints. We evaluated Pipette on large clusters to show that it provides a significant speedup over the prior art.
---
## Motivation
We diagnose that these methods tend to have three main limitations that restrict their practical use in the field.
1) Static Interconnect Assumption. The existing methods sim- ply assume that the interconnects between the servers are static, with a fixed bandwidth and latency. However, the actual communication latency in a real cluster exhibits heterogeneity among the links [9]–[11], which could yield unexpected straggler effects.
2) Hidden Critical Path. The existing methods construct latency models on the 3D parallelism but miss some critical paths. This comes from the discrepancy between the latency model and modern scheduling. While the state-of-the-art latency models [8], [12] assume outdated scheduling methods to achieve maximal throughput, the de facto standard is to use memory-efficient scheduling [5], [13] to relieve the memory capacity requirements.
3) Out-of-Memory Configurations. The configurations rec- ommended by the automated tools often require more memory per GPU than what is physically available. This is because those methods do not consider the memory usage of LLM [8] at all or fail to estimate it [12].
---
## Content
+ Codebases
+ Setups
+ Overall Steps
+ Some Tips
+ Roadmap
---
### Codebases

1. Megatron-LM : https://github.com/NVIDIA/Megatron-LM
2. Varuna : https://github.com/microsoft/varuna
3. AMP : https://github.com/DachengLi1/AMP
4. NCCL-tests: https://github.com/NVIDIA/nccl-tests
---
## Setups

### 1. Hardware


#### Mid-range Cluster
GPU: 8x NVIDIA V100

CPU: 2x Xeon Gold 6142, 16cores

Memory: 768GB DDR4 ECC

Inter-node: Infiniband EDR (100Gbps)

Intra-node: NVLink (300GBps)

#### High-end Cluster
GPU: 8x NVIDIA A100

CPU: 2x EPYC 7543, 32cores

Memory: 1TB DDR4 ECC

Inter-node: Infiniband HDR (200Gbps)

Intra-node: NVLink (600GBps)

### 2. Operating System

   
RHEL 8.4

### 3. Software


CUDA 11.6, PyTorch 1.14.3

### Overall Steps

1. See setup
   

3. See ...
...

4. Run training
..
---
### Some Tips
...
---

