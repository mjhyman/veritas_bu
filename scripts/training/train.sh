#!/bin/bash

jobsubmit -A psoct -p rtx6000 -m 20G -t 1-00:00:00 -c 8 -G 1 -o logs/train.log python3 scripts/training/train.py;
watch -n 0.1 "squeue -u $USER"