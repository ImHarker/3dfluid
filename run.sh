#!/bin/bash
#SBATCH --time=2:00
#SBATCH --partition=day
#SBATCH --constraint=k20

time ./fluid_sim
