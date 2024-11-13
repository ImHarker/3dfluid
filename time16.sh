#!/bin/bash

#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --partition=cpar
#SBATCH --time=00:10:00

module load gcc/11.2.0
make


threads=16
    export OMP_NUM_THREADS=$threads
    echo "Running fluid_sim with $threads threads..."

    total_time=0  # Reset total_time for each thread count

    for i in {1..10}; do
        echo "Attempt $i with $threads threads..."

        # Capture the output of fluid_sim, including density
        output=$( { time -p ./fluid_sim ; } 2>&1 )

        # Extract the density value
        density=$(echo "$output" | awk '/Total density after/ {print $NF}')
        
        # Extract the execution time in seconds
        run_time=$(echo "$output" | awk '/^real/ {print $2}')
        echo "Run $i: Density=${density}, Time=${run_time}s"

        # Sum up the time for each run
        total_time=$(echo "$total_time + $run_time" | bc)
    done

    # Calculate and display the average time for the current thread count
    avg_time=$(echo "scale=2; $total_time / 10" | bc)
    echo "Average execution time over 10 runs with $threads threads: ${avg_time}s"
    echo "--------------------------------------------"
    
    # Update threads: double if 1, otherwise increment by 2
    if [ $threads -eq 1 ]; then
        threads=2
    else
        threads=$((threads + 2))
    fi
