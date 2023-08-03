#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 175G -t 7-00:00:00 -c 30 -G 3 -o logs/train.log python3 scripts/training/train.py;
watch -n 0.1 "squeue -u epc28"
