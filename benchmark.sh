#!/bin/bash

min() {
    printf "%s\n" "$@" | sort -g | head -n1
}

rm -rf tmp/results
mkdir -p tmp/results

# Define the array of arguments
N_values=(10 100 1000 10000)
Q_values=(10 100 1000 10000)

# Loop through each argument
for N in "${N_values[@]}"; do
    for Q in "${Q_values[@]}"; do
        for i in {1..10}; do
            echo "$i. $N $Q"
            # Run the command with the current argument and redirect the output to a file
            make run-gpu ARGS="--N $N --Q $Q" > "tmp/results/out-gpu-$N-$Q-${i}.txt"
            make run-cpu ARGS="--N $N --Q $Q" > "tmp/results/out-cpu-$N-$Q-${i}.txt"
        done
    done
done
