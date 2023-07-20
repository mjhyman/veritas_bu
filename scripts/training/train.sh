#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 50G -t 7-00:00:00 -c 32 -G 5 -o scripts/training/train.log python3 scripts/training/train.py;
watch -n 0.1 "squeue -u epc28"
