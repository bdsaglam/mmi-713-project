#!/bin/bash

mkdir -p results

# Define the array of arguments
args=(10 100 1000 10000)

# Loop through each argument
for arg in "${args[@]}"; do
    for i in {1..5}; do
        # Run the command with the current argument and redirect the output to a file
        make run-gpu ARGS="$arg" > "results/out-gpu-${i}-${arg}.txt"
        make run-cpu ARGS="$arg" > "results/out-cpu-${i}-${arg}.txt"
    done
done
