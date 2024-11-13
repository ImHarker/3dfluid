#!/bin/bash

#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --partition=cpar
#SBATCH --time=00:02:00


perf record ./fluid_sim 
perf report -n --stdio > perfreport

