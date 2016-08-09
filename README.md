# MESO
The <sub><i>USER</i></sub>**MESO** package of LAMMPS is a fully GPU-accelerated package for Dissipative Particle Dynamics. Instead of being merely a translation of the conventional molecular dynamics, the package integrates several innovations that specifically targets CUDA devices: an atomics-free neighbor list construction algorithm and a locally transposed storage layout; a new seeding scheme for in-situ random number generators; fully overlapped computation/transfer; and specialized transcendental functions. It can achieve tens of times speedup on a single CUDA GPU over 8-16 CPU cores. The work is featured by a NVIDIA Parallel Forall blog article [Accelerating Dissipative Particle Dynamics Simulation on Tesla GPUs](https://devblogs.nvidia.com/parallelforall/accelerating-dissipative-particle-dynamics-simulation-tesla-gpus/).

# License
The package can be freely used and redistributed under the GPL v3 license. However we would greatly appreciate if you could cite the following paper:<br/>
Tang, Yu-Hang, and George Em Karniadakis. "Accelerating dissipative particle dynamics simulations on GPUs: Algorithms, numerics and applications." *Computer Physics Communications* 185.11 (**2014**): 2809-2822.

# Compilation Guide
NVCC and a MPI implementation is required for compilation of the code.
```
cd <working_copy>
make yes-molecule
make yes-user-meso
make meso ARCH=[sm_30|sm_35|sm_52|sm_60|...]
```
