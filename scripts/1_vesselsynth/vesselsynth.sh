#!/bin/bash

jobsubmit -A psoct -p rtx8000 -m 10G -t 1-00:00:00 -c 2 -G 1 -o logs/vesselsynth.log python3 scripts/1_vesselsynth/vessels_oct.py;
watch -n 0.1 "squeue -u $USER"