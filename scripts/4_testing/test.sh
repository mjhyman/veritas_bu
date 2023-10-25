#!/bin/bash

jobsubmit -A psoct -p rtx8000 -m 10G -t 1-00:00:00 -c 1 -G 1 -o logs/test.log python3 scripts/testing/test.py;
watch -n 0.1 "squeue -u $USER"