#!/bin/bash

jobsubmit -A psoct -p rtx8000 -m 10G -t 7-00:00:00 -c 8 -G 1 -o logs/vesselsynth.log python3 scripts/vesselsynth/vessels_oct.py;
watch -n 0.1 "squeue -u $USER"