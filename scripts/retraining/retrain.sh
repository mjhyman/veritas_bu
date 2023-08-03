#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 20G -t 7-00:00:00 -c 30 -G 9 -o logs/retrain.log python3 scripts/retrain.py;
watch -n 0.1 "squeue | grep epc"
