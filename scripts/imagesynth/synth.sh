#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 500G -t 1-00:00:00 -c 128 -G 1 -o test.log python3 scripts/vessel_synth.py;
watch -n 0.1 "squeue | grep epc"
