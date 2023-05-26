#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 100G -t 1-00:00:00 -c 30 -G 1 -o scripts/testing/test.log python3 scripts/testing/test.py;
watch -n 0.1 "squeue | grep epc"
