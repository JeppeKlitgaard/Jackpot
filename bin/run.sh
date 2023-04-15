#!/bin/bash
SCRIPT_FULL_PATH=$(dirname "$0")

function run {
    $SCRIPT_FULL_PATH/run_experiment.sh $1
}

run test