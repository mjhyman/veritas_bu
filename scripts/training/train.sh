#!/bin/bash

jobsubmit -A psoct -p rtx8000 -m 50G -t 7-00:00:00 -c 22 -G 6 -o scripts/training/train.log python3 scripts/training/train2.py;
watch -n 0.1 "squeue -u epc28"
