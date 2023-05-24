#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 75G -t 7-00:00:00 -c 32 -G 5 -o retrain.log python3 scripts/retrain.py;
watch -n 0.1 "squeue | grep epc"
