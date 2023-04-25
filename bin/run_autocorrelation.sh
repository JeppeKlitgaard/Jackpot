#!/bin/bash
SCRIPT_FULL_PATH=$(dirname "$0")

function run {
    $SCRIPT_FULL_PATH/run_experiment.sh $1
}

run "autocorrelation/N16_wolff"
run "autocorrelation/N16_metropolis_hastings"
run "autocorrelation/N32_wolff"
run "autocorrelation/N32_metropolis_hastings"
run "autocorrelation/N64_wolff"
run "autocorrelation/N64_metropolis_hastings"
run "autocorrelation/N128_wolff"
run "autocorrelation/N128_metropolis_hastings"
run "autocorrelation/N192_wolff"
run "autocorrelation/N192_metropolis_hastings"
run "autocorrelation/N256_wolff"
run "autocorrelation/N256_metropolis_hastings"
