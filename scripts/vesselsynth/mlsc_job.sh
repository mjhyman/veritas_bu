#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 250G -t 7-00:00:00 -c 70 -G 1 -o logs/vesselsynth.log python3 scripts/vesselsynth/vessels_oct.py;
watch -n 0.1 "squeue | grep $USER"