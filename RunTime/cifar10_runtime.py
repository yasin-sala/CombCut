# -*- coding: utf-8 -*-
# (The line above is a "magic comment" to prevent encoding errors)

import numpy as np
import graphlearning as gl
import time
from datetime import datetime

# --- Experiment Configuration ---
# The name of the output file where results will be saved
output_filename = "cifar10_runtimes.txt"
# Number of independent trials for each rate
n_runs = 5
# The different training set rates to test
rates_to_test = [1, 1, 1]

# Use a 'with' block to safely open and close the file
# 'w' mode creates a new file (or overwrites an existing one)
with open(output_filename, 'w') as f:
    
    # Custom print function to write to both console and file
    def log(message):
        print(message)       # Print to the console
        f.write(message + '\n') # Write to the file (add a newline)

    # --- Start of Execution ---
    start_time_total = time.time()

    log("="*60)
    log(f"Starting CIFAR-10 Runtime Benchmark")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Results will be saved to: {output_filename}")
    log("="*60)
    log(f"Dataset: CIFAR-10")
    log(f"Rates to test: {rates_to_test}")
    log(f"Runs per rate: {n_runs}")
    log("="*60)

    # --- Setup (Done Once) ---
    log("\nLoading CIFAR-10 labels...")
    labels = gl.datasets.load('cifar10', labels_only=True)
    log("Labels loaded.")

    # Pre-build the graph and class priors
    log("\nBuilding weight matrix (metric='simclr')... This may take a while.")
    W = gl.weightmatrix.knn('cifar10', 10, metric='simclr', kernel='gaussian')
    log("Weight matrix built.")
    class_priors = gl.utils.class_priors(labels)
    log("Class priors built.")

    # Define all your models
    models = [
        gl.ssl.laplace(W),
        gl.ssl.poisson(W),
        gl.ssl.plaplace(W, p=3),
        gl.ssl.amle(W),
        gl.ssl.volume_mbo(W, class_priors),
        gl.ssl.cut_ssl(W, class_priors),
        gl.ssl.stiefel_ssl(W),
        gl.ssl.CombCutSSL(W, class_priors=class_priors),
    ]
    model_names = [model.name for model in models]

    # --- Main Experiment Loop ---
    # Outer loop for iterating through different training set rates
    for rate in rates_to_test:
        log(f"\n--- Starting Test for Rate = {rate} points per class ---")
        
        # Prepare storage for runtimes for the current rate
        runtime_records = {name: [] for name in model_names}

        # Inner loop for repeating the experiment n_runs times
        for run in range(n_runs):
            # Print progress to console only to keep the log file cleaner
            if (run + 1) % 5 == 0:
                print(f"  Running trial {run + 1}/{n_runs}...")
            
            # Generate a fresh training set for each run
            train_ind = gl.trainsets.generate(labels, rate=rate)
            train_labels = labels[train_ind]
            
            for model in models:
                # Measure the runtime for fit_predict
                start_time = time.perf_counter()
                _ = model.fit_predict(train_ind, train_labels)
                end_time = time.perf_counter()
                
                # Calculate and store the elapsed time
                elapsed_time = end_time - start_time
                runtime_records[model.name].append(elapsed_time)
                
        # --- Calculate and Log Results for the current rate ---
        log("\n--- Results for Rate = {} (Mean of {} runs) ---".format(rate, n_runs))
        for name, runtimes in runtime_records.items():
            mean_runtime = np.mean(runtimes)
            log(f"Model: {name:<12} | Mean Runtime: {mean_runtime:.4f} seconds")

    # --- Final Summary ---
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    log("\n" + "="*60)
    log(f"All benchmark runs completed in {total_duration:.2f} seconds.")
    log(f"Results have been saved to '{output_filename}'.")
    log("="*60)

# The 'with' block has ended, so the file is now closed and saved.
print(f"\nProcess finished. Check the file '{output_filename}' for the complete log.")
