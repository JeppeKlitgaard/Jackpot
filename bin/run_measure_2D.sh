#!/bin/bash
SCRIPT_FULL_PATH=$(dirname "$0")

function run {
    $SCRIPT_FULL_PATH/run_experiment.sh $1
}

run "measure_2D/N16_wolff"
run "measure_2D/N32_wolff"
run "measure_2D/N64_wolff"
run "measure_2D/N96_wolff"
run "measure_2D/N128_wolff"
run "measure_2D/N160_wolff"
run "measure_2D/N192_wolff"
run "measure_2D/N224_wolff"
run "measure_2D/N256_wolff"
