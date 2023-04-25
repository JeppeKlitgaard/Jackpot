#!/bin/bash
SCRIPT_FULL_PATH=$(dirname "$0")

function run {
    $SCRIPT_FULL_PATH/run_experiment.sh $1
}

run "performance/N16_wolff"
run "performance/N16_metropolis_hastings"
run "performance/N32_wolff"
run "performance/N32_metropolis_hastings"
run "performance/N64_wolff"
run "performance/N64_metropolis_hastings"
run "performance/N128_wolff"
run "performance/N128_metropolis_hastings"
run "performance/N192_wolff"
run "performance/N192_metropolis_hastings"
run "performance/N256_wolff"
run "performance/N256_metropolis_hastings"
