#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 75G -t 7-00:00:00 -c 32 -G 4 -o scripts/training/train.log python3 scripts/training/train.py;
#jobsubmit -A psoct -p rtx8000 -m 50G -t 7-00:00:00 -c 4 -G 7 -o train.log python3 train.py;
watch -n 0.1 "squeue | grep epc"
