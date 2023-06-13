#!/bin/bash

jobsubmit -A psoct -p rtx8000 -m 500G -t 1-00:00:00 -c 20 -G 1 -o scripts/testing/test.log python3 scripts/testing/test.py;
watch -n 0.1 "squeue | grep epc"
