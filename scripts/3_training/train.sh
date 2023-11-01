#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 100G -t 7-00:00:00 -c 32 -G 5 -o logs/train.log python3 scripts/3_training/train.py;
watch -n 0.1 "squeue -u $USER"