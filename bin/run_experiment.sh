#!/bin/bash
set +xe

mkdir -p experiments/$1

papermill workbench.ipynb \
    -f "parameters/base.yaml" \
    -f "parameters/$1.yaml" \
    --log-output \
    --stdout-file experiments/$1/papermill.log \
    --stderr-file experiments/$1/papermill.log \
    experiments/$1/experiment.ipynb