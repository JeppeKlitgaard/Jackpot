#!/bin/bash
SCRIPT_FULL_PATH=$(dirname "$0")

function run {
    $SCRIPT_FULL_PATH/run_experiment.sh $1
}

run "measure_3D/N8_wolff"
run "measure_3D/N16_wolff"
run "measure_3D/N24_wolff"
run "measure_3D/N32_wolff"
run "measure_3D/N40_wolff"
run "measure_3D/N48_wolff"
run "measure_3D/N56_wolff"
run "measure_3D/N64_wolff"
