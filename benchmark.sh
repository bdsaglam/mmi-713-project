#!/bin/bash

mkdir -p results

# Define the array of arguments
args=(10 100 1000 10000)

# Loop through each argument
for arg in "${args[@]}"; do
    # Run the command with the current argument and redirect the output to a file
    make run-gpu ARGS="$arg" > "results/out-gpu-${arg}.txt"
    grep "Elapsed time" "results/out-gpu-${arg}.txt" | awk '{print $3}' > "results/time-gpu-${arg}.txt"

    make run-cpu ARGS="$arg" > "results/out-cpu-${arg}.txt"
    grep "Elapsed time" "results/out-cpu-${arg}.txt" | awk '{print $3}' > "results/time-cpu-${arg}.txt"
done
