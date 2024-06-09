#!/bin/bash

min() {
    printf "%s\n" "$@" | sort -g | head -n1
}

rm -rf results
mkdir -p results

# Define the array of arguments
N_values=(10 100 1000 10000)
Q_values=(10 100 1000)

# Loop through each argument
for N in "${N_values[@]}"; do
    for Q in "${Q_values[@]}"; do
        q="$(min $N $Q)"
        for i in {1..3}; do
            echo "$i. $N $Q"
            # Run the command with the current argument and redirect the output to a file
            make run-gpu ARGS="--N $N --Q $q" > "results/out-gpu-$N-$q-${i}.txt"
            make run-cpu ARGS="--N $N --Q $q" > "results/out-cpu-$N-$q-${i}.txt"
        done
    done
done
