#!/bin/bash

jobsubmit -A psoct -p rtx6000 -m 100G -t 7-00:00:00 -c 32 -G 8 -o logs/vesselsynth.log python3 scripts/vesselsynth/vessels_oct.py;
watch -n 0.1 "squeue | grep $USER"