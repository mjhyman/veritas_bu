#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 100G -t 7-00:00:00 -c 64 -G 1 -o scripts/imagesynth/real.log python3 scripts/imagesynth/real.py;
watch -n 0.1 "squeue | grep epc"
