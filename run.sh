#!/bin/bash
#SBATCH --time=5:00
#SBATCH --partition=day
#SBATCH --constraint=k20

time nvprof ./fluid_sim