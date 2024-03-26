# [DATE 2024] Pipette: Automatic Fine-grained Large Language Model Training Configurator for Real-World Clusters

It is common to use a cluster of GPUs with 3D parallelism, which splits a model along the data batch, pipeline stage, and intra-layer tensor dimensions. However, the use of 3D parallelism produces the additional challenge of finding the optimal number of ways on each dimension and mapping the split models onto the GPUs. We propose Pipette, which is an automatic fine-grained LLM training configurator for real-world clusters. By devising better performance models along with the memory estimator and fine-grained individual GPU assignment, Pipette achieves faster configurations that satisfy the memory constraints. We evaluated Pipette on large clusters to show that it provides a significant speedup over the prior art.
---
## Motivation
we diagnose that these methods tend to have three main limitations that restrict their practical use in the field.
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

1. Megatron-LM :
2. Varuna :
3. AMP : 
---
### Setups

1. Hardware
...

2. Operating System
...

3. Software
...
---
### Overall Steps

1. See setup
...

2. See ...
...

3. Run training
..
---
### Some Tips
...
---
### Roadmap
...
