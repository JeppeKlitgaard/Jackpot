#!/bin/bash
SCRIPT_FULL_PATH=$(dirname "$0")

function run {
    $SCRIPT_FULL_PATH/run_experiment.sh $1
}

run "d2_spin0.5_N2"
run "d2_spin0.5_N4"
run "d2_spin0.5_N8"
run "d2_spin0.5_N16"
run "d2_spin0.5_N32"
run "d2_spin0.5_N64"
run "d2_spin0.5_N96"
run "d2_spin0.5_N128"
run "d2_spin0.5_N192"
run "d2_spin0.5_N256"
run "d2_spin0.5_N320"
run "d2_spin0.5_N384"
run "d2_spin0.5_N448"
run "d2_spin0.5_N512"