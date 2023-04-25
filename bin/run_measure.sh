#!/bin/bash
SCRIPT_FULL_PATH=$(dirname "$0")

function run {
    $SCRIPT_FULL_PATH/run_experiment.sh $1
}

run "measure/N16_wolff"
run "measure/N32_wolff"
run "measure/N64_wolff"
run "measure/N96_wolff"
run "measure/N128_wolff"
run "measure/N160_wolff"
run "measure/N192_wolff"
run "measure/N224_wolff"
run "measure/N256_wolff"