#!/bin/bash

#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --partition=cpar
#SBATCH --time=00:10:00


module load gcc/11.2.0
make profile
perf stat -e instructions,branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores -r 3 ./fluid_sim
